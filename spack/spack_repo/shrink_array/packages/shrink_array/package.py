# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class ShrinkArray(CMakePackage):
    """Compressed arrays for C++ HPC codes: used like a std::vector, they take
    less memory when large parts of them hold the same value -- zero, or a few
    repeated values. An array is cut into chunks of S elements; a uniform chunk
    keeps no buffer of its own. Arrays of 10^9 elements, written from OpenMP
    threads -- or, with threads=pool, from std::threads or TBB.

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
    version("0.1.3", tag="v0.1.3", commit="c6ea28ded4ee48e1c9d7807071b5cdeeaa022d53")
    version("0.1.2", tag="v0.1.2", commit="94b6445820db7459a848a921a86d86574115828d")
    version("0.1.1", tag="v0.1.1", commit="23ff5fef32bcd410073b563db01beedb02010ba0")
    version("0.1.0", tag="v0.1.0", commit="b95605bd4255abf4365fae098a41b6bfffad9ca9")
    version("main", branch="main")
    version("cpp23", branch="cpp23")
    version("develop", branch="develop")

    variant(
        "threads",
        default="openmp",
        values=("openmp", "pool", "serial"),
        multi=False,
        description="Threading backend of the libraries (CHUNK_THREADS): openmp; pool, std::jthread "
        "workers of the library's own, for programs whose threads are std::threads or TBB; serial",
    )
    variant(
        "tbb",
        default=False,
        description="With +examples and threads=pool: also the TBB bench and the TBB adaptor's validation "
        "(CHUNK_WITH_TBB). The libraries never link TBB: a TBB program includes chunk/tbb_adaptor.h "
        "and links TBB itself",
    )
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
    depends_on("tbb", when="+tbb")

    # Linux only: membarrier, madvise, mincore, /proc.
    requires("platform=linux", msg="shrink-array relies on Linux system calls and /proc")

    # C++20 on main and develop (GCC 13 and Clang 18 are the oldest tested),
    # C++23 on cpp23 (deducing this: GCC 14, Clang 18).
    conflicts("%gcc@:12", msg="shrink-array needs GCC 13 or later")
    conflicts("%gcc@:13", when="@cpp23", msg="the cpp23 branch needs GCC 14 or later (C++23)")
    conflicts("%clang@:17", msg="shrink-array needs Clang 18 or later")
    conflicts("%nvhpc", when="@cpp23", msg="nvc++ has no C++23")
    conflicts("threads=pool", when="@:0.1.2", msg="the pool backend comes after 0.1.2")
    conflicts("+tbb", when="~examples", msg="+tbb builds TBB examples only: add +examples")
    conflicts("+tbb", when="threads=openmp", msg="the TBB examples need threads=pool")
    conflicts("+tbb", when="threads=serial", msg="the TBB examples need threads=pool")

    def cmake_args(self):
        spec = self.spec
        sizes = sorted(spec.variants["sizes"].value, key=int)
        threads = spec.variants["threads"].value
        args = [
            # OMP for every version; CHUNK_THREADS (which sets OMP to match)
            # only where the pool backend exists.
            self.define("OMP", threads == "openmp"),
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
        if threads == "pool":
            args.append(self.define("CHUNK_THREADS", "pool"))
        if spec.satisfies("+tbb"):
            args.append(self.define("CHUNK_WITH_TBB", True))
        return args
