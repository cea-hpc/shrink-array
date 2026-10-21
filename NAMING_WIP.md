# Branch `naming`: work in progress

One naming convention over the whole tree, as written in
[docs/contributing.md](docs/contributing.md) section 3. This file is the hand-over note of the
branch; delete it before merging.

## Commits

| commit | content |
|---|---|
| lot 1 | the names that read alike everywhere, renamed by word over code, docs and scripts (string literals, `#include` lines and file names left alone): camelCase functions, UPPER_CASE / PascalCase enumerators, snake_case aliases and concepts, `chunk2A` in snake_case names, `__debug_flag`. `getChunkList()` and `ParamChunk::ChunkStats()` keep a `[[deprecated]]` alias |
| lot 2 | the names whose meaning depends on scope, through clang-tidy `readability-identifier-naming`, then the uses it missed from the compiler's errors: `_` at the end of private / protected data members, parameters, short UPPER_CASE locals (`N` -> `n`, `S` -> `s`) |
| lot 3 | what only an OpenMP build compiles, UPPER_CASE template parameters, reserved `__` names (-> `g_`), `check_naming` extended, `scripts/clang_tidy_naming.yaml`, the convention in `docs/contributing.md` |

## Verified

- gcc 16, OMP=OFF (serial): build OK, ctest 336 / 337 -- the failure, `simple_array_vector`,
  fails on develop too.
- gcc 16, OMP=ON: build OK. **ctest not run yet.**
- The six CI source checks (`check_docs`, `check_tracked`, `check_raw_reads`,
  `check_includes`, `check_naming`, `check_headers`): OK.

## To do before merging into develop

1. ctest of the OMP=ON build.
2. Build with `-DOMP=ON -DCHUNK_STD_ALLOC=ON` (lot 3 found the std-alloc `ArenaKind` that
   only this build compiles: other such code may remain).
3. Build with clang (OMPT, multi-mat-embedded): `-DOMP=ON -DUSE_LIBCXX=OFF
   -DGCC_TOOLCHAIN_DIR=<gcc 16 root>`, clang run with `LD_LIBRARY_PATH=<gcc 16>/lib64`.
4. `-Wshadow` before / after in an OMP=ON build. The lot 2 commit message says a `-Wshadow`
   build shows no new shadowing: that was checked in the serial build only; amend it if the
   OMP build disagrees. A rename that makes a name shadow another one compiles silently:
   lot 2 met one (`TCSlab::init`, `element_size = element_size`), fixed.
5. Delete this file, then merge into develop. `cpp23` and `threads` then need a merge of
   develop: expect conflicts on the renamed names.

## Tools

- An error left by a rename (`'x' was not declared`, `has no member named 'x'`): the
  compiler usually suggests the new name (`x_`). Fix at the reported position, never by a
  global replace of a short name -- a global `n` / `s` / `value` replace is how a silent
  shadowing gets in.
- `scripts/clang_tidy_naming.yaml` lists every rule for clang-tidy; its regular expressions
  are POSIX (`\w` is not understood).
- OMP is OFF by default in CMake: build with `-DOMP=ON` too, or the parallel paths are not
  compiled at all.
