# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class ShrinkArray(CMakePackage):
    """Compressed arrays for C++ HPC codes: used like a std::vector, they take
    less memory when large parts of them hold the same value -- zero, or a few
    repeated values. An array is cut into chunks of S elements; a uniform chunk
    keeps no buffer of its own. OpenMP, arrays of 10^9 elements.

    Installs one library per variant and chunk size, found with
    find_package(chunk): chunk::chunk_v1_<S>, chunk::chunk_v2_<S>,
    chunk::chunk_v2a_<S> (static S), chunk::chunk_v<x>_dyn (runtime S),
    chunk::chunk_v<x> (the first static size)."""

    homepage = "https://github.com/cea-hpc/shrink-array"
    git = "https://github.com/cea-hpc/shrink-array.git"

    license("Apache-2.0")

    # The releases (tags, from git: the repository is not public yet), then the
    # branches: main (the last release and what follows), cpp23 (C++23),
    # develop (the branch main is fast-forwarded from).
    version("0.1.1", tag="v0.1.1", commit="86c2a5ffe42aa61a86a1f185ad8ea47f597de653")
    version("0.1.0", tag="v0.1.0", commit="6950e39826ce30eccb453b850660ba6d48020102")
    version("main", branch="main")
    version("cpp23", branch="cpp23")
    version("develop", branch="develop")

    variant("openmp", default=True, description="Build the parallel (OpenMP) libraries")
    variant(
        "sizes",
        default="256,1024",
        # What the library accepts (is_valid_chunk_size): a power of two or a multiple of 64.
        values=lambda x: x.isdigit() and int(x) > 0 and (int(x) % 64 == 0 or int(x) & (int(x) - 1) == 0),
        multi=True,
        description="Static chunk sizes, each a power of two or a multiple of 64: one library per variant "
        "and size (chunk::chunk_v2a_256, ...); "
        "the smallest is the one of the bare chunk::chunk_v<x> targets",
    )
    variant(
        "dyn",
        default=True,
        description="Also a runtime-size library per variant (chunk::chunk_v<x>_dyn): adaptive re-chunk",
    )
    variant(
        "fast_alloc",
        default=True,
        description="The fast_allocators slab allocator for the chunk buffers (off: the standard allocator)",
    )
    variant("stats", default=False, description="Collect compression statistics (CHUNK_STATS)")
    variant(
        "second_value_indexed",
        default=False,
        description="Chunk2A: second values as 4-byte identifiers (less memory, about +3% write time)",
    )
    variant(
        "checks",
        default=False,
        description="Runtime invariant checks in an optimised build (CHECK_OPTIM)",
    )
    variant(
        "examples",
        default=False,
        description="Also build the examples, benchmarks and validation harnesses (not installed)",
    )

    depends_on("cxx", type="build")
    depends_on("cmake@3.20:", type="build")

    # Linux only: membarrier, madvise, mincore, /proc.
    requires("platform=linux", msg="shrink-array relies on Linux system calls and /proc")

    # C++20 on main and develop (GCC 13 and Clang 18 are the oldest tested),
    # C++23 on cpp23 (deducing this: GCC 14, Clang 18).
    conflicts("%gcc@:12", msg="shrink-array needs GCC 13 or later")
    conflicts("%gcc@:13", when="@cpp23", msg="the cpp23 branch needs GCC 14 or later (C++23)")
    conflicts("%clang@:17", msg="shrink-array needs Clang 18 or later")
    conflicts("%nvhpc", when="@cpp23", msg="nvc++ has no C++23")

    def cmake_args(self):
        spec = self.spec
        sizes = sorted(spec.variants["sizes"].value, key=int)
        return [
            self.define_from_variant("OMP", "openmp"),
            self.define("CHUNK_STATIC_SIZES", ";".join(sizes)),
            self.define("CHUNK_DEFAULT_SIZE", sizes[0]),
            self.define_from_variant("CHUNK_BUILD_DYN", "dyn"),
            self.define("CHUNK_STD_ALLOC", spec.satisfies("~fast_alloc")),
            self.define_from_variant("CHUNK_STATS", "stats"),
            self.define_from_variant("CHUNK_SECOND_VALUE_INDEXED", "second_value_indexed"),
            self.define_from_variant("CHECK_OPTIM", "checks"),
            self.define_from_variant("SHRINK_ARRAY_BUILD_EXAMPLES", "examples"),
            self.define("SHRINK_ARRAY_BUILD_TESTS", self.run_tests),
            # The compiler and standard library Spack chose: no libc++ switch
            # (the project's default with Clang), no mold, no Intel libm.
            self.define("USE_LIBCXX", False),
            self.define("USE_MOLD", "OFF"),
            self.define("SHRINK_INTEL_LIBM", False),
        ]
