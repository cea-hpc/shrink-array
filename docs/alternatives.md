# Alternatives, and when to prefer them

> French version: [alternatives_fr.md](alternatives_fr.md).

shrink-array is a drop-in replacement for a `std::vector` that the program keeps reading and
writing in place, from OpenMP threads: same index space, same `[]`, no loss, and every repeated
value is compressed chunk by chunk (`zero_any`), not only zero. Other structures save memory on
other data, and are the better choice there:

| alternative | what it does | prefer it when |
|---|---|---|
| sparse formats (CSR / COO, Eigen SparseMatrix, PETSc `Mat`) | stores the non-zero entries and their indices; only zero is implicit | the data is very sparse and the computation is sparse linear algebra (SpMV, solvers); the structure changes rarely, since writing outside it means rebuilding it |
| hash tables (`std::unordered_map`, open-addressing maps) | a key and a value per stored entry, tens of bytes each, scattered in memory | the entries are extremely few, without spatial structure, and their range is not known in advance; with threads it needs a concurrent table |
| compressors with array access: [zfp](https://github.com/LLNL/zfp) compressed arrays, [Blosc2](https://www.blosc.org/) | compress blocks of any data, varied values included; zfp's fixed-rate arrays are lossy and offer `[]` through a block cache; Blosc2 is lossless by default, by chunk and block | the values are varied but compressible, or a controlled loss is acceptable (zfp, SZ), or the array is mostly written once and read, or stored (I/O, checkpoints): a write in the middle decompresses and recompresses a block |
| sparse 3D grids: [OpenVDB](https://www.openvdb.org/) / NanoVDB, SPGrid, AMR libraries | a hierarchy of tiles over a 3D domain, with a background value and constant tiles | the data is a 3D field handled as geometry (narrow bands, level sets, neighbourhoods, topology operations) rather than as an array indexed by the code's own loops |

The idea of a uniform chunk that keeps no buffer is close to OpenVDB's constant tiles; what
shrink-array adds is that it stays a flat array: the loops, the indexing and the parallel
writes of a code written for `std::vector` are kept as they are. Where every chunk holds varied
values, none of these structures helps much except a compressor, and shrink-array costs a little
more than a `std::vector`.
