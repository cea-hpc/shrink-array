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
#include <cstring> // std::strcmp

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
    std::string cmd = std::string("\"") + argv0 + "\"";
    if (g_running != nullptr)
    {
        cmd += std::string(" ") + g_running;
    }
    return cmd;
}

int main(int argc, char** argv)
{
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
