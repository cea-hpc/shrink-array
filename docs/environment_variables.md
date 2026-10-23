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
| `CHUNK_LOG` | `warn` | which messages of the library are written: comma-separated items, left to right; a bare level (`off`, `error`, `warn`, `info`, `debug`, `trace`) sets every channel, `channel=level` one (`general`, `env`, `alloc`, `arena`, `pool`, `rechunk`, `drain`, `compress`, `recycle`, `ledger`, `ompt`, `affinity`, `advisor`, `memory`). Example: `CHUNK_LOG=warn,rechunk=debug`. An invariant failure is written whatever the setting. A program can also call `shrink_array::logging::configure()` and hand the messages to its own logger with `shrink_array::logging::set_sink()` (`chunk/log.h`) |
| `CHUNK_LOG_FILE` | stderr | file the messages are appended to; `%p` is replaced by the process id |
| `CHUNK_THREAD_CAPACITY` | max(`omp_get_max_threads()`, cores, 256) | number of ranks the per-thread structures are sized for; can only raise the default (oversubscription). A rank at or past it stops the program with a message naming this variable |
| `CHUNK_REQUIRE_OMPT` | — | OMPT builds: `1` aborts the program if the OMPT tool was not loaded by the runtime (otherwise, a warning of the log), since the end-of-phase events would then never happen |
| `CHUNK_RECHUNK_DRY_RUN` | — | `1`: the adaptive re-chunk decider analyses and logs what it would do (level info of the log's channel `rechunk`, which it switches on), but applies no rechunk (see [adaptive_rechunk.md](adaptive_rechunk.md)) |

## 2. CMake options

| option | default | effect |
|---|---|---|
| `OMP` | `OFF` | build with OpenMP |
| `CHUNK_THREADS` | empty | threading backend of the library: `openmp`, `std` (`std::jthread` workers of its own, no OpenMP) or `serial`; empty: `openmp` with `OMP=ON`, `serial` otherwise. Set, it sets `OMP` to match |
| `CHUNK_STATIC_SIZES` | `auto` | static chunk sizes built (targets `_<size>`); `auto`: `256;1024`, `256` alone in a Debug or sanitizer build |
| `CHUNK_BUILD_DYN` | `ON` | also build a dynamic-geometry library (`_dyn`) per variant |
| `CHUNK_STD_ALLOC` | `OFF` | standard allocator for the chunk buffers instead of `fast_allocators` (useful under ASan: each buffer is a real `malloc`) |
| `CHUNK_SECOND_VALUE_INDEXED` | `OFF` | Chunk2A: the second-value slot holds a 4-byte identifier instead of the value (less memory, +3% write time) |
| `CHUNK_STATS` | `OFF` | instrumentation counters and the `print_chunk_stats()` report |
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
headers read them there, whatever a program defines itself. A file compiled with OpenMP against a
library built without it, or the reverse, does not compile. A program spelling the variant or the
geometry differently from the library it links does not link: the undefined symbol
(`shrink_array::link_check::variant_...`) names the settings it was compiled with.
