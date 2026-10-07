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
// The main() of a grouped validation executable (see validation_harness.h).
// ============================================================================
//
// The harnesses of the executable are listed by VALIDATION_HARNESS_LIST, a
// header CMake writes per target, one VALIDATION_HARNESS(name) line each. The
// first argument names the one to run; the rest is its own command line.
// With a single harness (CHUNK_VALIDATION_UNITY=OFF: one executable per
// harness) the name may be left out.
// ============================================================================

#include <cstdio>  // std::fprintf, stderr
#include <cstdlib> // std::getenv
#include <cstring> // std::strcmp

#ifdef __linux__
    #include <sys/prctl.h> // prctl(PR_SET_DUMPABLE)
#endif

#include "validation_harness.h" // validation::self_command (defined here)


#define VALIDATION_HARNESS(name)    \
    namespace validation::name      \
    {                               \
    int run(int argc, char** argv); \
    }
#include VALIDATION_HARNESS_LIST // one VALIDATION_HARNESS(name) per harness: declares its run()
#undef VALIDATION_HARNESS

namespace
{

struct Harness
{
    const char* name;
    int (*run)(int, char**);
};

const Harness kHarnesses[] = {
#define VALIDATION_HARNESS(name) {#name, &validation::name::run},
#include VALIDATION_HARNESS_LIST // one VALIDATION_HARNESS(name) per harness: its table entry
#undef VALIDATION_HARNESS
};
constexpr int kNbHarnesses = static_cast<int>(sizeof kHarnesses / sizeof kHarnesses[0]);

const char* g_running = nullptr; // the harness name given on the command line, if any

} // namespace

std::string validation::self_command(const char* argv0)
{
    // CHUNK_VALIDATION_CHILD=1: the child is one this harness runs to watch it
    // fail -- most of them abort on purpose -- and main() makes it
    // non-dumpable (see there).
    std::string cmd = std::string("CHUNK_VALIDATION_CHILD=1 \"") + argv0 + "\"";
    if (g_running != nullptr)
    {
        cmd += std::string(" ") + g_running;
    }
    return cmd;
}

int main(int argc, char** argv)
{
#ifdef __linux__
    // A child run through self_command() aborts on purpose. Dumpable, every
    // such abort is a core dump: under WSL each one is captured and written to
    // Windows' %TEMP%\wsl-crashes, some 300 MB apiece for an executable whose
    // arenas reserve that much address space. stale_proxy_validation's 1- and
    // 8-thread runs fire dozens back to back -- 261 captures, 7.7 GB, and WSL
    // went down mid-capture. A non-dumpable process writes no core at all.
    if (std::getenv("CHUNK_VALIDATION_CHILD") != nullptr)
    {
        prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
    }
#endif
    if (argc > 1)
    {
        for (const Harness& h : kHarnesses)
        {
            if (std::strcmp(argv[1], h.name) == 0)
            {
                g_running = h.name;
                argv[1]   = argv[0]; // the harness sees argv[0] = the executable, then its own arguments
                return h.run(argc - 1, argv + 1);
            }
        }
    }
    if (kNbHarnesses == 1)
    {
        return kHarnesses[0].run(argc, argv);
    }

    std::fprintf(stderr, "usage: %s <harness> [args...]\nharnesses:\n", argv[0]);
    for (const Harness& h : kHarnesses)
    {
        std::fprintf(stderr, "  %s\n", h.name);
    }
    return 2;
}
