# Environment variables and build options

> French version: [variables_environnement_fr.md](variables_environnement_fr.md).

> Status: October 2026. This page lists what a **user** of the library sets: the settings the
> versioning policy of [CHANGELOG.md](../CHANGELOG.md) covers. Everything else -- the tuning of
> the decider and of the allocators, the traces, the benchmark, the test hooks, the debugging
> options, the build scripts -- is in [developer_settings.md](developer_settings.md): development
> tools, which can change or disappear in any release.
>
> A variable is read only once; `0` or an empty value is the same as leaving it unset. A
> variable that is not read is ignored without any message: when in doubt, check with
> `grep -rn '"NAME"' chunk fast_allocators`.

***

## 1. Environment variables

| variable | default | effect |
|---|---|---|
| `CHUNK_LOG` | `warn,rechunk=info` | which messages of the library are written: comma-separated items, left to right; a bare level (`off`, `error`, `warn`, `info`, `debug`, `trace`) sets every channel (`rechunk` included: `CHUNK_LOG=warn` hides the re-chunks), `channel=level` one (`general`, `env`, `alloc`, `arena`, `pool`, `rechunk`, `drain`, `compress`, `recycle`, `ledger`, `ompt`, `affinity`, `advisor`, `memory`). Example: `CHUNK_LOG=warn,rechunk=debug`. The library's reports are levels of their channel: `arena=info` (the buffer arena), `ledger=info` (the block registry, at exit), `ompt=info` (the synchronisation regions, at exit); a bare `info` or higher switches them all on, with the counters of the arena and of OMPT, which cost a little time. An invariant failure is written regardless of the setting. A program can also call `shrink_array::logging::configure()` and hand the messages to its own logger with `shrink_array::logging::set_sink()` (`chunk/log.h`) |
| `CHUNK_LOG_FILE` | stderr | file the messages are appended to; `%p` is replaced by the process id |
| `CHUNK_BREAK_ON_FAILURE` | auto | what an invariant failure (or another fatal failure of the library) does once its report -- message, backtrace, sink -- is written, before it aborts: under a debugger, it raises `SIGTRAP`, so that the debugger stops on the failing thread with its frames live, after a line on stderr saying so; continued, the program aborts as without one. Auto (unset, or any value but `0` and `1`): on Linux, a debugger is the process `TracerPid` names in `/proc/self/status`, when it is `gdb`, `cuda-gdb`, `gdbserver`, `lldb` or `lldb-server`; strace, ltrace, perf or rr, which trace too, never stop the program. TotalView and Linaro DDT / Forge are not detected: they stop on `SIGABRT` already; `CHUNK_BREAK_ON_FAILURE=1` forces the stop if wanted. Elsewhere than on Linux, auto never stops. `0`: never (an exception to the rule above: `0` is not the default). `1`: always, debugger or not -- without one, `SIGTRAP` then kills the program (exit status 133), unless it handles the signal |
| `CHUNK_THREAD_CAPACITY` | max(`omp_get_max_threads()`, cores, 256) | number of ranks the per-thread structures are sized for; can only raise the default (oversubscription). A rank at or past it stops the program with a message naming this variable |
| `CHUNK_REQUIRE_OMPT` | — | OMPT builds: `1` aborts the program if the OMPT tool was not loaded by the runtime (otherwise, a warning in the log), since the end-of-phase events would then never happen |

## 2. CMake options

| option | default | effect |
|---|---|---|
| `OMP` | `OFF` | build with OpenMP |
| `CHUNK_THREADS` | empty | threading backend of the library: `openmp`, `std` (`std::jthread` workers of its own, no OpenMP) or `serial`; empty: `openmp` with `OMP=ON`, `serial` otherwise. Set, it sets `OMP` to match |
| `CHUNK_STATIC_SIZES` | `auto` | static chunk sizes built (targets `_<size>`); `auto`: `256;1024`, `256` alone in a Debug or sanitizer build |
| `CHUNK_BUILD_DYN` | `ON` | also build a dynamic-geometry library (`_dyn`) per variant |
| `CHUNK_STD_ALLOC` | `OFF` | standard allocator for the chunk buffers instead of `fast_allocators` (useful under ASan: each buffer is a real `malloc`) |
| `CHUNK_SECOND_VALUE_INDEXED` | `OFF` | `zero_any`: the second-value slot holds a 4-byte identifier instead of the value (less memory, +3% write time) |
| `CHUNK_STATS` | `OFF` | instrumentation counters and the `ParamChunk::chunk_stats()` report |
| `CHECK_OPTIM` | `OFF` | invariant checks in an optimised build |
| `SANITIZER` | `none` | `thread`, `address` or `token` (AllocToken, Clang ≥ 22), for all build types |

## 3. Variant and geometry

They are not set by the program: they come with the library target it links
(`chunk::chunk_zero_any_256`, `chunk::chunk_zero_dyn`, ...), which carries the macros
`CHUNK_VARIANT_ZERO` / `CHUNK_VARIANT_ZERO_ONE_VALUE` / `CHUNK_VARIANT_ZERO_ANY`, and `CHUNK_DYNAMIC` for a
dynamic geometry (the size is chosen at run time, per array; `DYN_CHUNK_SIZE` is the
benchmark's own macro, not the library's). See [getting_started.md](getting_started.md).

The options shared by every library of a build (§2) are recorded by CMake in the generated
`chunk/build_options.h` and `fast_allocators/build_options.h`, installed with the libraries; the
headers read them there, regardless of what a program defines itself. A file compiled with OpenMP against a
library built without it, or the reverse, does not compile. A program spelling the variant or the
geometry differently from the library it links does not link: the undefined symbol
(`shrink_array::link_check::variant_...`) names the settings it was compiled with.
