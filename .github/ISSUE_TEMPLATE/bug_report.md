<!-- SPDX-Symbol: GitHub issue template; the repository's license (LICENSE) applies -->

## What happened

<!-- A short description of the wrong behaviour. If an invariant fired
     (CHUNK_INVARIANT_FAILED / CHUNK_CHECK_FAILED), paste the whole message:
     it carries the file, the line and the thread. -->

## How to reproduce

<!-- The smallest program or command line that shows it. When it fits, a
     variant of one of the harnesses in examples/validation/ is the easiest
     form: they already build everywhere and compare against std::vector. -->

## Build

- Compiler and version:
- Backend (openmp / pool / serial / TBB):
- Library variant (chunk / chunk2 / chunk2a) and geometry (size / dyn):
- Element type:
- Build type (Release / check / stats):

## For a concurrency report

<!-- Which handshake it looks like -- writer vs compression pass, deferred
     flush, or reader vs reclaim (docs/formal_models.md names the three) --
     and whether it survives the native stress runners (ctest -L formal). -->
