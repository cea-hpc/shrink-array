/*
 * Copyright 2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ============================================================================
// A child process for the validation harnesses, without a shell.
// ============================================================================
//
// Several harnesses check a failure that ends the process (an invariant that
// aborts, a refused team): they run themselves again as a child and read its
// exit status and output. They used std::system() on a command line built by
// concatenation -- paths, arguments, environment assignments, redirections --
// which a shell then parsed: a temporary directory or an executable path
// holding a quote, a `$` or a space would have been read as shell syntax.
// run_process() takes the argument vector as is and starts it with
// posix_spawn(): nothing is parsed, nothing is interpolated.
//
// Header-only: log_validation, built on its own (not with
// validation_driver.cpp), uses it as the grouped harnesses do through
// validation::run_self() (validation_harness.h).
// ============================================================================

#pragma once

#include <cerrno>         // EINTR
#include <chrono>         // std::chrono::steady_clock, milliseconds (the timeout)
#include <csignal>        // SIGKILL, kill
#include <cstdio>         // std::fopen, std::fread, std::fclose, std::remove
#include <cstring>        // std::strerror, std::strchr
#include <string>         // std::string
#include <string_view>    // std::string_view
#include <thread>         // std::this_thread::sleep_for (the timeout's poll)
#include <vector>         // std::vector (argv, envp)
#include <fcntl.h>        // O_WRONLY, O_CREAT, O_TRUNC
#include <spawn.h>        // posix_spawn, posix_spawnp, file actions, attributes
#include <sys/resource.h> // getrlimit, setrlimit, RLIMIT_CORE
#include <sys/wait.h>     // waitpid, WIFEXITED, WEXITSTATUS, WIFSIGNALED, WTERMSIG, WNOHANG
#include <unistd.h>       // environ, pid_t

extern char** environ; // POSIX: the process environment (unistd.h declares it under _GNU_SOURCE only)

namespace validation
{

/// @brief How run_process() starts the child.
struct ProcessOptions
{
    /// `NAME=value` entries, added to this process's environment; an entry
    /// replaces a variable of the same name, and a later entry an earlier one.
    std::vector<std::string> env;
    /// File that receives the child's stdout and stderr (created 0600,
    /// truncated); read into ProcessResult::output and removed afterwards.
    /// Empty: the child writes to this process's streams.
    std::string output;
    /// Seconds before the child's process group is killed (SIGKILL); 0: no
    /// limit. With a limit the child leads a process group of its own, so that
    /// a wrapper's tracee (gdb, strace) goes with it.
    int timeout_s = 0;
    /// The child may not write a core file (RLIMIT_CORE 0).
    bool no_core = false;
};

/// @brief How the child ended.
struct ProcessResult
{
    bool        started   = false; ///< posix_spawn succeeded
    bool        exited    = false; ///< ended by exit(): code is meaningful
    bool        timed_out = false; ///< killed by run_process() at timeout_s
    int         code      = -1;    ///< exit status, when exited
    int         signal    = 0;     ///< the signal that ended it, when not exited
    std::string output;            ///< stdout + stderr (ProcessOptions::output), or why it did not start

    /// @brief Exited with status 0.
    [[nodiscard]] bool ok() const noexcept
    {
        return started && exited && code == 0;
    }
    /// @brief Ran, and did not end with status 0: an error status or a signal.
    [[nodiscard]] bool stopped() const noexcept
    {
        return started && !ok();
    }
    /// @brief The status as a shell reports it: the exit status, or 128 + the
    ///        signal (134 for an abort); -1 when the child did not start.
    [[nodiscard]] int shell_status() const noexcept
    {
        return !started ? -1 : (exited ? code : 128 + signal);
    }
};

namespace process_detail
{

/// @brief `NAME` of a `NAME=value` entry.
inline std::string_view env_name(std::string_view entry) noexcept
{
    return entry.substr(0, entry.find('='));
}

/// @brief This process's environment, then @p extra: an entry of @p extra
///        replaces any earlier one of the same name.
inline std::vector<std::string> merged_environment(const std::vector<std::string>& extra)
{
    std::vector<std::string> env;
    const auto               put = [&env](std::string_view entry)
    {
        const std::string_view name = env_name(entry);
        std::erase_if(env, [name](const std::string& e) { return env_name(e) == name; });
        env.emplace_back(entry);
    };
    for (char** e = environ; e != nullptr && *e != nullptr; ++e)
    {
        put(*e);
    }
    for (const std::string& entry : extra)
    {
        put(entry);
    }
    return env;
}

/// @brief The NULL-terminated char* view of @p strings (which must outlive it).
inline std::vector<char*> c_array(std::vector<std::string>& strings)
{
    std::vector<char*> out;
    out.reserve(strings.size() + 1);
    for (std::string& s : strings)
    {
        out.push_back(s.data());
    }
    out.push_back(nullptr);
    return out;
}

/// @brief The content of @p path (empty when it cannot be read).
inline std::string read_file(const std::string& path)
{
    std::string out;
    if (FILE* file = std::fopen(path.c_str(), "r"))
    {
        char buf[4096];
        for (std::size_t n = 0; (n = std::fread(buf, 1, sizeof buf, file)) > 0;)
        {
            out.append(buf, n);
        }
        std::fclose(file);
    }
    return out;
}

/// @brief waitpid() through EINTR.
inline pid_t wait_for(pid_t pid, int* status, int flags) noexcept
{
    pid_t r = -1;
    do
    {
        r = ::waitpid(pid, status, flags);
    } while (r == -1 && errno == EINTR);
    return r;
}

} // namespace process_detail

/// @brief Run @p argv (argv[0] is the program: looked up in PATH when it holds
///        no '/') and wait for it. No shell: every element reaches the child
///        as one argument, verbatim.
/// @note RLIMIT_CORE is lowered in this process for the duration of the
///       spawn when ProcessOptions::no_core is set (posix_spawn has no
///       attribute for it): a core of this process in that window would be
///       lost, nothing else.
inline ProcessResult run_process(std::vector<std::string> argv, const ProcessOptions& opt = {})
{
    ProcessResult res;
    if (argv.empty())
    {
        res.output = "run_process: empty argv";
        return res;
    }
    std::vector<std::string> env      = process_detail::merged_environment(opt.env);
    std::vector<char*>       argv_c   = process_detail::c_array(argv);
    std::vector<char*>       envp_c   = process_detail::c_array(env);
    const bool               use_path = std::strchr(argv[0].c_str(), '/') == nullptr;

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    if (!opt.output.empty())
    {
        posix_spawn_file_actions_addopen(&actions, 1, opt.output.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        posix_spawn_file_actions_adddup2(&actions, 1, 2);
    }
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    if (opt.timeout_s > 0)
    {
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attr, 0); // the child leads a group: kill(-pid) reaches its tracee too
    }

    rlimit     saved_core{};
    const bool lower_core = opt.no_core && ::getrlimit(RLIMIT_CORE, &saved_core) == 0;
    if (lower_core)
    {
        rlimit none   = saved_core;
        none.rlim_cur = 0;
        ::setrlimit(RLIMIT_CORE, &none);
    }
    pid_t     pid = -1;
    const int err = use_path ? ::posix_spawnp(&pid, argv_c[0], &actions, &attr, argv_c.data(), envp_c.data())
                             : ::posix_spawn(&pid, argv_c[0], &actions, &attr, argv_c.data(), envp_c.data());
    if (lower_core)
    {
        ::setrlimit(RLIMIT_CORE, &saved_core);
    }
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&actions);
    if (err != 0)
    {
        res.output = "run_process: cannot start " + argv[0] + ": " + std::strerror(err) + "\n";
        if (!opt.output.empty())
        {
            std::remove(opt.output.c_str());
        }
        return res;
    }
    res.started = true;

    int status = 0;
    if (opt.timeout_s > 0)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(opt.timeout_s);
        while (process_detail::wait_for(pid, &status, WNOHANG) == 0)
        {
            if (std::chrono::steady_clock::now() >= deadline)
            {
                ::kill(-pid, SIGKILL);
                process_detail::wait_for(pid, &status, 0);
                res.timed_out = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    else
    {
        process_detail::wait_for(pid, &status, 0);
    }
    if (WIFEXITED(status))
    {
        res.exited = true;
        res.code   = WEXITSTATUS(status);
    }
    else if (WIFSIGNALED(status))
    {
        res.signal = WTERMSIG(status);
    }
    if (!opt.output.empty())
    {
        res.output = process_detail::read_file(opt.output);
        std::remove(opt.output.c_str());
    }
    return res;
}

} // namespace validation
