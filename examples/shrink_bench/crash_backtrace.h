#pragma once

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h> // sigaction, siginfo_t (POSIX, not in <csignal>)
#include <unistd.h>
#include <execinfo.h> // backtrace, backtrace_symbols_fd
#include <sys/syscall.h>

namespace crash_bt
{

/// @brief Kernel thread id of the caller.
/// @return syscall(SYS_gettid) when available and positive, getpid() otherwise.
inline long get_thread_id()
{
#if defined(SYS_gettid)
    const long tid = syscall(SYS_gettid);
    if (tid > 0)
        return tid;
#endif
    return static_cast<long>(getpid()); // fallback
}

/// @brief Signal handler: print the signal, fault address, thread id and a backtrace on stderr, run gcore when
///        CRASH_BT_GCORE=1, then restore the default handler and re-raise @p sig.
/// @param sig Signal number.
/// @param info Signal information (si_addr is printed).
inline void handler(int sig, siginfo_t* info, void* /*ucontext*/)
{
    // async-signal-safe-ish: we use write()/backtrace which are acceptable here
    // for a debugging aid (we're crashing anyway).
    char        msg[256];
    const void* addr = info ? info->si_addr : nullptr;
    const int   n =
        std::snprintf(msg, sizeof(msg), "\n=========== CRASH_BT: signal %d (%s), si_addr=%p, tid=%ld ===========\n",
                      sig, (sig == SIGSEGV ? "SIGSEGV" : (sig == SIGABRT ? "SIGABRT" : "?")), addr, get_thread_id());
    if (n > 0)
    {
        const ssize_t w = write(STDERR_FILENO, msg, static_cast<size_t>(n));
        (void)w;
    }

    // Native backtrace of THIS (faulting) thread.
    void*     frames[64];
    const int nframes = backtrace(frames, 64);
    backtrace_symbols_fd(frames, nframes, STDERR_FILENO);

    const char*   tail = "=========== CRASH_BT end — re-raising ===========\n";
    const ssize_t w2   = write(STDERR_FILENO, tail, std::strlen(tail));
    (void)w2;

    // Optional: dump a full core via gcore (works even when ulimit -c = 0,
    // because gcore uses ptrace on ourselves from a forked child). Enable by
    // setting env CRASH_BT_GCORE=1. The core lands in CWD as core.<pid>.
    if (const char* g = std::getenv("CRASH_BT_GCORE"); g != nullptr && g[0] == '1')
    {
        char cmd[256];
        std::snprintf(cmd, sizeof(cmd), "gcore -o core %d >/dev/null 2>&1", static_cast<int>(getpid()));
        // system() in a signal handler is not async-signal-safe, but we're
        // already crashing; best-effort postmortem.
        const int r = std::system(cmd);
        (void)r;
    }

    // Restore default handler and re-raise so the process still dies with the
    // right signal (and the shell reports "Segmentation fault").
    signal(sig, SIG_DFL);
    raise(sig);
}

} // namespace crash_bt

/// @brief Install crash_bt::handler for SIGSEGV, SIGABRT and SIGBUS.
inline void install_crash_backtrace()
{
    struct sigaction sa{};
    sa.sa_sigaction = crash_bt::handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    // SIGBUS too (misaligned/mmap faults on aarch64)
    sigaction(SIGBUS, &sa, nullptr);
}
