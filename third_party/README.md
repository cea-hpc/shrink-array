# Vendored third-party code

Two header-only hash tables, copied into this repository and used by the chunk library
only. They live in `third_party/chunk/vendor/`, are included as `"chunk/vendor/..."` (the build
include directory is `third_party/`) and are installed under `include/chunk/vendor/` -- never at
the root of `include/`: a user's own copy of either library is left alone, and a user's
`#include <ankerl/unordered_dense.h>` never finds chunk's copy.

| file | upstream | version | license | local patches |
|---|---|---|---|---|
| `chunk/vendor/ankerl/unordered_dense.h` | [martinus/unordered_dense](https://github.com/martinus/unordered_dense) | v5.3.1 (commit `c164976d`) | MIT, [licenses/unordered_dense-MIT.txt](licenses/unordered_dense-MIT.txt) | 2 |
| `chunk/vendor/ankerl/stl.h` | martinus/unordered_dense | v5.3.1 (commit `c164976d`) | MIT | none |
| `chunk/vendor/flat_hash_map.hpp` | [skarupke/flat_hash_map](https://github.com/skarupke/flat_hash_map) | master, commit `2c468743` (2018-07-15, the last one: unmaintained) | Boost Software License 1.0, [licenses/flat_hash_map-BSL-1.0.txt](licenses/flat_hash_map-BSL-1.0.txt) | 1 |

Each file opens with the SPDX line of its upstream license (`// SPDX-License-Identifier: MIT`,
`BSL-1.0`; ctest `check_headers` checks it), then a banner stating the above. Each local patch is marked
`Local patch (shrink-array)` in the file:

    grep -rn "Local patch (shrink-array)" third_party

[NOTICE](../NOTICE) lists the same for redistribution; it, `LICENSE` and `licenses/` are
installed under `share/doc/chunk/`.

## The local patches

- `unordered_dense.h`, `ANKERL_UNORDERED_DENSE_NOINLINE` / `_FORCEINLINE`: under nvc++, the
  `on_error_*` functions are `inline` (ODR) and noinline (the throw out of the callers), which
  nvc++ reports as `inline_gnu_noinline_conflict`. The noinline hint is dropped on that
  compiler only; the functions are cold. `ANKERL_UNORDERED_DENSE_FLATTEN` keeps its GNU
  spelling there, as on gcc and clang.
- `unordered_dense.h`, the NEON group match: the 64-bit mask is read out with `memcpy` instead
  of `vreinterpret_u64_u8` + `vget_lane_u64`, which nvc++ reports, once per instantiation, as
  `incompatible_vectors_conversion`. gcc and clang compile both to the same `fmov x0, d0`.
- `flat_hash_map.hpp`, `templated_iterator`'s conversion to the const iterator: under clang
  and nvc++ it is a template disabled when the iterator is already const (a self-conversion),
  and clang's `-Wclass-conversion`, emitted at the definition before SFINAE applies, is
  silenced around it.

## Things to know

- The version macros of `unordered_dense.h` are upstream's and are not touched: they name the
  inline namespace (`ankerl::unordered_dense::v5_3_1`).
- `unordered_dense.h` mentions `CLAUDE.md`, `notes/index-design.md` and issue numbers: those are
  files and issues of the upstream repository, kept as they are.
- One translation unit sees one copy of each. Both keep their upstream guard
  (`ANKERL_UNORDERED_DENSE_H`, `#pragma once`): a program that includes its own
  `unordered_dense.h` before chunk's headers compiles chunk against that copy, and
  `chunk_base.h` refuses one whose major version is not 5. A program's own
  `flat_hash_map.hpp` in the same translation unit as chunk's headers does not compile (two
  definitions of `ska::flat_hash_map`).

## Updating a vendored file

1. Take the file from the upstream tag, e.g. in a clone of the upstream repository:
   `git show vX.Y.Z:include/ankerl/unordered_dense.h`.
2. Re-apply the local patches (drop one that upstream made unnecessary), each still marked
   `Local patch (shrink-array)`.
3. Keep the SPDX line on top, update the banner below it, the table above and, if upstream's changed, the
   license text in `licenses/`.
4. Build with gcc, clang and nvc++ (the patches exist for nvc++), on x86-64 and aarch64 (the
   NEON patch), run the check builds and compare the benchmark with the previous version.
