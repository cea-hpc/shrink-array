# Measuring memory and time without fooling yourself

> French version: [mesures_performance_fr.md](mesures_performance_fr.md).

> Status: end of September 2026. This document gathers what the September 2026 campaigns on
> Grace (aarch64, 64 KiB pages, RHEL 9.6, 64 threads) taught us, often the hard way:
> several conclusions had to be withdrawn because the measuring instrument was lying.

***

## 1. Memory

### 1.1 `VmRSS` is wrong with several threads

`VmRSS` (`/proc/self/status`), `/proc/self/statm` and `ru_maxrss` (`getrusage`) read the
kernel's resident memory counters. These counters are kept **per thread**, and only folded
into the process total from time to time. With 64 threads taking page
faults, the total lags behind. Measured on `shrink_bench_zero_any_dyn`:

| point | `VmRSS` | exact (`smaps_rollup`) | lag |
|---|---|---|---|
| initialisation | 9,576 MiB | 11,675 MiB | 2,099 MiB |
| start of steady state | 11,820 MiB | 12,436 MiB | 616 MiB |
| end of run | 11,870 MiB | 13,442 MiB | 1,571 MiB |

`VmRSS` had stayed at ~11.8 GiB for the whole run, while the process grew by one
gigabyte: we had wrongly concluded that the rechunks gained nothing.

### 1.2 The exact measurement: `smaps_rollup`

`/proc/self/smaps_rollup` walks the page tables at the time of the read: it is exact.
But its cost grows with resident memory (~10 to 20 ms for a 13 GB process): keep it
for checkpoints, never on a hot periodic path (the 200 reads that
the benchmark's timed loop used to make cost it 1 to 2 s).

In the library: `shrink_array::exact_rss_kib()` (`internal/exact_rss.h`). In the benchmark:

- the "Memory consumption **after init**" and "**end**" lines are exact;
- the periodic lines reread `statm` (lagging, but free), except with
  `SHRINK_BENCH_EXACT_RSS=1`;
- the "peak" is the maximum of `ru_maxrss` and of the exact reads;
  `SHRINK_BENCH_RSS_SAMPLE_MS=<ms>` adds background sampling;
- the memory traces (`CHUNK_LOG=memory=info` / `memory=debug`) read the exact value.

### 1.3 Locating memory

| question | tool |
|---|---|
| how much in chunk buffers, how much elsewhere? | `CHUNK_LOG=memory=info`: at each advisor tick, the exact memory, including the resident part of the buffer arena (`mincore`) |
| what are the slabs doing? | `CHUNK_ARENA_REPORT=1` (slabs handed out, returned to the system) |
| which area of the address space is growing? | two `pmap -x <pid>` at two points in time, compared area by area |
| what does each rechunk return? | `CHUNK_LOG=memory=debug`: `rss` line (memory at each step, `end-entry`) |

A trap: the "materialised" counter of the stats build is a difference of counters
(allocated − freed). It is only correct if every free is counted, which was not the
case for frees without compression (rechunk, destruction) until the end of September 2026: we
had seen 12,982 MB "materialised" in a 12,012 MB process.

**Memory measurements, on the other hand, are reproducible**: from one run to the next and from one build to
the next, to within a megabyte. A single run is enough.

## 2. Time

### 2.1 Three sources of noise, much larger than the effects being sought

- **Bimodality, set at start-up.** Over ten identical runs without the advisor, seven at
  6.4-6.8 s and three at 7.7-8.0 s (+18%). `perf stat`: same frequency (3.26 GHz), same number
  of instructions, same last-level cache misses, but an IPC of 3.45 instead of
  3.95 and +10 points of cycles stalled on memory. Each memory access costs more,
  probably depending on the physical layout of the pages received at start-up. `zero_one_value` has an even
  more pronounced slow mode (~+40%, runs at 16 s instead of 11).
- **Drift between time windows.** The same binary, with no recompilation at all: 10.75 to
  11.53 s in one window, 9.49 to 10.08 s an hour later.
- **The gap between two builds of the same code.** Same commit, same options, no local
  change: one build at 10.5 s, another at 9.7 s, consistently. Probably the
  layout of the code in memory (alignment of the hot loops).

An **A/A test** (the same code, twice, alternating) gives the measure of the noise: on the
dynamic benchmarks at 64 threads, ~10% spread between extremes.

### 2.2 The rule

**Only compare times on the same binary**, changing the behaviour through an
environment variable, **alternating** the modes within the same time window, over **at
least three rounds**, and comparing the **averages**. Only keep a difference clearly
above 10%.

Comparing two builds (even on the same day, with the same options) is not reliable below
±10-15%. When it is unavoidable (measuring a code change), build both
in fresh directories with the same configuration, alternate, and be wary.

### 2.3 The switches made for this

| compare… | switch |
|---|---|
| without advisor / analyses only / with rechunks | `CHUNK_ADVISOR_PERIOD=0` / `CHUNK_RECHUNK_DRY_RUN=1` / nothing |
| lazy / eager affinity | `CHUNK_ALLOC_AFFINITY_EAGER=0` / `=1` |
| maximum size of a coarsening step | `CHUNK_RECHUNK_MAX_COARSEN=<n>` |
| scheduling of the rebuild | `CHUNK_RECHUNK_SCHEDULE=…` |

The full list is in [developer_settings.md](developer_settings.md).
Beware: `CHUNK_ADVISOR_MIN_GAIN_MB=1000000` no longer blocks coarsenings since the
"gain per cost" criterion; use `CHUNK_RECHUNK_DRY_RUN=1`.

## 3. Recipes

**The cost of the advisor on the same binary** (three rounds, averages):

```
rm -f /tmp/modes.txt
for r in 1 2 3; do
  for mode in "CHUNK_ADVISOR_PERIOD=0" "CHUNK_RECHUNK_DRY_RUN=1" "X_NORMAL=1"; do
    t=$(env $mode OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores $BIN --mode perf --no_io 2>/dev/null \
        | grep -oE "Elapsed time computation : [0-9.]+" | awk '{print $NF}')
    echo "$mode $t" >> /tmp/modes.txt
  done
done
awk '{ s[$1] += $2; n[$1]++ } END { for (k in s) printf "%-26s %.2f s\n", k, s[k] / n[k] }' /tmp/modes.txt
```

**The duration of each interval between two advisor ticks**, to see whether a
slowdown comes from the rechunks or from what follows them:

```
tick_times() {
  env "$@" CHUNK_LOG=warn,memory=info,rechunk=info OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores \
    $BIN --mode perf --no_io 2>&1 >/dev/null \
  | while read -r l; do
      case "$l" in
        *"[memory][I]"*)  echo "T $(date +%s.%N) ${l#*tick }" ;;
        *APPLIED*)        echo "R" ;;
      esac
    done \
  | awk '$1 == "R" { nr++; next }
         { t = $2; k = $3; if (p != "") printf "  %2d->%-2d %5.2f s%s\n", pk, k, t - p, nr ? sprintf("   (%d rechunks)", nr) : ""; p = t; pk = k; nr = 0 }'
}
```

**Second-by-second behaviour** (busy cores, IPC, cycles stalled on memory):

```
perf stat -I 1000 -x, -e task-clock,cycles:u,instructions:u,stall_backend_mem:u -o /tmp/pi.csv -- $BIN --mode perf --no_io
```

The number of busy cores is `task-clock / 1000` for each second. This is what showed
that `zero`, after its rechunk, only kept ~40 cores busy instead of ~52.

**User time, system time, page faults**:
`/usr/bin/time -f "%e s real, %U s user, %S s system, %R minor faults" $BIN …`.

**Fragmentation of the node's memory**: `cat /proc/buddyinfo`. On a kernel with
64 KiB pages, the last column counts the free 512 MiB blocks.

**What tells a slow run from a fast one** (the bimodality of 2.1): `python/bimodal_probe.py`
runs the same binary N times and records, for each run, the pages per NUMA node, AnonHugePages,
the free memory and free top-order blocks of each node before the run, and the perf counters
per thousand instructions (TLB walks, L1D / L2 refills, remote accesses). It splits the runs at
the largest time gap and says which quantities separate the two groups. `--mode` alternates
environments (`--mode "" --mode "CHUNK_THP=never"`, `--mode "numactl --membind=0"`).

```
python3 python/bimodal_probe.py --runs 12 -- $BIN --mode perf --no_io
```

## 4. Traps encountered

- **A binary that was not rebuilt**, or a script that was not applied: an
  environment variable that no code reads is ignored without any message. Check with
  `strings $BIN | grep VARIABLE_NAME`, or through the expected effect.
- **An inherited CMake cache**: options from old configurations stay in
  `CMakeCache.txt`. When in doubt, reconfigure in a fresh directory. To compare two
  builds, the command `ninja -C <build> -t commands <target>` shows the options actually
  used.
- **A fresh build does not have the intended options**: `OMP` is `OFF` by default, and a benchmark without
  OpenMP runs on a single core. Check `ldd $BIN | grep gomp`.
- **The assembler is too old** for `-mcpu=native` with GCC 16 ("unknown architecture"):
  load recent `binutils`.
- **A `std::vector` baseline initialised on one thread**: `std::vector<T> v(n)` writes its
  zeros from the constructing thread, so the creation is serial and every page sits on that
  thread's NUMA node; the parallel loops that follow read remote memory, and the chunked
  arrays, which allocate nothing at creation, look faster than they are. `shrink_bench`'s
  baseline zeroes in parallel instead (`examples/shrink_bench/baseline_vector.h`): keep it
  when you add a phase or a type, and do not compare its creation times with runs made before
  that change.
- **An array size computed as `int`**: 8 × 10⁹ overflows and becomes negative. The
  targets with 64-bit indices (`_int64`) do not have this problem.
