#!/bin/bash
# --fresh-cache: drop a build dir's CMakeCache.txt before it is (re)configured,
# so cached variables (SHRINK_BENCH_DYN_SIZE, toolchain, generator, stale option()s) are
# re-read from the -D flags / CMakeLists defaults instead of sticking at their
# old cached value. Scoped to the dir being configured; never fires in --build
# mode (no configure happens then, and wiping would break the ninja rebuild).
maybe_wipe_cache() {  # <build-dir>
    [ "${FRESH_CACHE:-false}" = true ] || return 0
    [ "${BUILD_ONLY:-false}" = true ] && return 0
    local dir=$1
    [ -n "$dir" ] && [ -f "$dir/CMakeCache.txt" ] || return 0
    if [ "${DRYRUN:-false}" = true ]; then
        echo "rm -f $dir/CMakeCache.txt  # --fresh-cache"
    else
        echo "[fresh-cache] rm $dir/CMakeCache.txt"
        rm -f "$dir/CMakeCache.txt"
    fi
    return 0
}

# ─── Hardening (CHUNK_GLIBCXX_ASSERTIONS / CHUNK_UBSAN) ──────────────────────
# Decides, from a cmake configure argument string, which hardening options of
# chunk/CMakeLists.txt to switch on. One rule, applied identically to every
# compiler section (gnu via compile_with_cmake, llvm/oneapi/nvhpc via run_cmd):
#
#   -DCMAKE_BUILD_TYPE=Debug or -DSANITIZER=<x>  →  -DCHUNK_GLIBCXX_ASSERTIONS=ON
#       libstdc++ precondition checks (vector::operator[] bounds...). This is what
#       exposed the out-of-bounds operator[] in allocate_chunk_internal. ABI-safe
#       and cheap, so every Debug / sanitizer flavour gets it -- nvhpc included.
#
#   -DSANITIZER=address                          →  -DCHUNK_UBSAN=ON
#       UBSan pairs with ASan in the same build; never with TSan (the CMake side
#       refuses the combination, we simply never ask for it).
#
# Release / RelWithDebInfo flavours are untouched: the bench compares their
# timings and memory, instrumentation would skew them.
#
# Opt-out: --no-hardening (compile_all.sh / compile_all_gnu.sh) exports
# CHUNK_HARDENING=off. Explicit -DCHUNK_GLIBCXX_ASSERTIONS= / -DCHUNK_UBSAN= in
# the args always win (nothing is added for an option already present).
hardening_opts_for() {  # <cmake-args> → echo extra -D flags ("" if none)
    [ "${CHUNK_HARDENING:-on}" = off ] && return 0
    local args=$1 out=""
    local debug=false asan=false san=false
    case "$args" in *-DCMAKE_BUILD_TYPE=Debug*) debug=true ;; esac
    case "$args" in *-DSANITIZER=address*|*-DSANITIZER=token*)     asan=true; san=true ;;
                    *-DSANITIZER=*)            san=true ;; esac
    if { $debug || $san; }; then
        case "$args" in *CHUNK_GLIBCXX_ASSERTIONS*) ;;
                        *) out="$out -DCHUNK_GLIBCXX_ASSERTIONS=ON" ;; esac
    fi
    if $asan; then
        case "$args" in *CHUNK_UBSAN*) ;;
                        *) out="$out -DCHUNK_UBSAN=ON" ;; esac
    fi
    printf '%s' "${out# }"
}

# Insert the hardening flags into a full `... cmake -S ... -B ... [|| fallback]`
# command string (compile_all.sh style). Flags go before any trailing `||` so
# the fallback branch is preserved; cmake does not care about -D ordering.
__with_hardening() {  # <cmd-string> → echo rewritten command
    local cmd=$1 extra
    extra=$(hardening_opts_for "$cmd")
    [ -z "$extra" ] && { printf '%s' "$cmd"; return 0; }
    case "$cmd" in
        *" || "*) printf '%s %s || %s' "${cmd%% || *}" "$extra" "${cmd#* || }" ;;
        *)        printf '%s %s' "$cmd" "$extra" ;;
    esac
}

run_cmd() {
    # --target applies HERE, not in launch_build at the end of the block: by
    # then the title and the whole cmake line have already been printed, which
    # is what made --dryrun --target=<glob> unreadable. A cmake configure names
    # its build dir with -B, so the filter has everything it needs.
    case "$*" in
        *cmake*" -B "*)
            local _tdir
            _tdir=$(printf '%s' "$*" | sed -nE 's/.* -B[[:space:]]+"?([^"[:space:]]+).*/\1/p')
            if [ -n "$_tdir" ] && ! __dir_wants "$_tdir"; then
                __PENDING_HEADER=""   # this block says nothing at all
                return 0
            fi
            ;;
    esac
    __flush_header
    # If this run_cmd carries a `cmake ... -B <dir>` configure and --fresh-cache
    # is active, wipe that dir's cache first (see maybe_wipe_cache).
    if [ "${FRESH_CACHE:-false}" = true ]; then
        case "$*" in
            *cmake*" -B "*)
                local _bdir
                _bdir=$(printf '%s' "$*" | sed -nE 's/.* -B[[:space:]]+"?([^"[:space:]]+).*/\1/p')
                maybe_wipe_cache "$_bdir"
                ;;
        esac
    fi
    # Hardening: every cmake configure that goes through run_cmd (llvm, oneapi,
    # nvhpc sections of compile_all.sh) gets the same flags as the gnu builds.
    local _cmd="$*"
    case "$_cmd" in
        *cmake*" -B "*) _cmd=$(__with_hardening "$_cmd") ;;
    esac
    if [ "$DRYRUN" = true ]; then
        echo "$_cmd"
    else
        eval "$_cmd"
    fi
}

# ─── Parallel-build tracking (shared by compile_all.sh / compile_all_gnu.sh) ──
# Every Ninja build is launched in the background with its output redirected to
# a dedicated log ($LOG_DIR/<name>.log). launch_build prints a [LAUNCH] line and
# records pid → (name, log, t0); wait_all consumes the jobs as they finish and
# prints [ OK ]/[FAIL] + duration the moment a build completes — no more
# interleaved Ninja output, no more guessing which build is done.
# NB: per-process state — a child script (compile_all_gnu.sh) has its own table
# and must call wait_all before returning (its ninjas would otherwise be
# orphaned, out of reach of the parent's wait).
# ─── Build target filter (--target=<glob>) ───────────────────────────────────
# CHUNK_BUILD_TARGET (exported by compile_all.sh / compile_all_gnu.sh) is a glob
# matched against the readable build name (the cmake-build-<name> component,
# without prefix or $arch). Empty = build everything. Examples:
#   omp-llvm        -> only cmake-build-omp-llvm            (exact)
#   'omp-llvm*'     -> the whole omp-llvm family
#   '*sanitizer*'   -> every sanitizer build
#   '*check*'       -> every check build
__target_wants() {  # <name> -> 0 build it, 1 skip it
    [ -z "${CHUNK_BUILD_TARGET:-}" ] && return 0
    case "$1" in ($CHUNK_BUILD_TARGET) return 0 ;; esac
    return 1
}

# Same predicate, taking a build DIRECTORY instead of a readable name: strips
# $build_rep and the $arch component, exactly as launch_build and
# compile_with_cmake do, so the three agree on what a name is.
__dir_wants() {  # <build-dir> -> 0 build it, 1 skip it
    local __n=${1#"${build_rep:-}"/}; __n=${__n%%/*}; __n=${__n#cmake-build-}
    __target_wants "$__n"
}

# Deferred block title. compile_all.sh announces a configuration before it knows
# whether --target keeps it; storing the line lets run_cmd print it only if it
# emits something. Replaces `echo -e '\nCompiling X:\n'`.
__PENDING_HEADER=""
set_build_header() { __PENDING_HEADER="$1"; }
__flush_header() {
    [ -n "$__PENDING_HEADER" ] || return 0
    printf '\n%s\n\n' "$__PENDING_HEADER"
    __PENDING_HEADER=""
}

# How many configurations --target let through, and how many it saw.
__TARGET_SEEN=0
__TARGET_KEPT=0
target_filter_summary() {
    [ -n "${CHUNK_BUILD_TARGET:-}" ] || return 0
    printf '[TARGET] %s of %s configurations match --target=%s\n' \
           "$__TARGET_KEPT" "$__TARGET_SEEN" "$CHUNK_BUILD_TARGET"
}

# ─── Typo guard for --target / --exec ────────────────────────────────────────
# A --target or --exec that matched nothing used to build nothing and exit 0:
# "0 of 27 configurations match" scrolled by, and the typo went unnoticed until
# the missing binary did. Both are now checked, and a miss is an error naming
# the closest known argument(s) -- Levenshtein distance, as suggest_closest does
# for the bench's flags.
#
#   --target  is checked BEFORE anything runs, against the cmake-build-<name>
#             names written in the scripts themselves (known_build_names): the
#             list is static, so there is no reason to configure twenty build
#             dirs first. A name that exists but is disabled on this node
#             (oneapi on ARM, --no-llvm...) still gets through here and is
#             caught by check_arg_epilogue at the end.
#   --exec    can only be checked against a configured build dir (the Ninja
#             target list). launch_build checks each dir as it goes -- a
#             literal name is looked up exactly as a pattern is now, instead of
#             being handed to cmake to fail in a log -- and records what it saw
#             in the run's slot directory (shared with the gnu child, so the
#             verdict is global: an executable missing from a sanitizer dir but
#             present in the gnu ones is a [WARN] per dir, not an error).
__closest_names() {  # <query> <names...> → the 3 nearest names, nearest first
    local q=$1 sub=0; shift
    # A pattern is compared as a fragment: '*sanitiser*' should find
    # omp-gnu-debug-thread-sanitizer, not the short names a whole-string
    # distance favours. The glob characters themselves carry nothing to compare.
    case "$q" in *[*?[]*) sub=1 ;; esac
    q=${q//[\*\?\[\]]/}
    printf '%s\n' "$@" | sort -u | awk -v q="$q" -v frag="$sub" '
        # Levenshtein distance from q to b (frag=0), or to the closest substring
        # of b (frag=1: free start and end in b).
        function lev(a, b,   la, lb, i, j, c, best, prev, cur) {
            la = length(a); lb = length(b)
            for (j = 0; j <= lb; j++) prev[j] = frag ? 0 : j
            for (i = 1; i <= la; i++) {
                cur[0] = i
                for (j = 1; j <= lb; j++) {
                    c = (substr(a, i, 1) == substr(b, j, 1)) ? 0 : 1
                    best = prev[j] + 1
                    if (cur[j-1] + 1 < best) best = cur[j-1] + 1
                    if (prev[j-1] + c < best) best = prev[j-1] + c
                    cur[j] = best
                }
                for (j = 0; j <= lb; j++) prev[j] = cur[j]
            }
            if (!frag) return prev[lb]
            best = prev[0]
            for (j = 1; j <= lb; j++) if (prev[j] < best) best = prev[j]
            return best
        }
        NF { printf "%d\t%d\t%s\n", lev(q, $0), length($0), $0 }
    ' | sort -t "$(printf '\t')" -k1,1n -k2,2n -k3,3 | head -n 3 | cut -f3-
}

# The build names a script can configure: every cmake-build-<name> it mentions,
# with the run's ICF suffix and --llvm-cs size substituted (<N> when unknown).
known_build_names() {  # <script>... → one name per line
    grep -oh 'cmake-build-[^/"[:space:]]*' "$@" 2>/dev/null \
      | sed -e 's/^cmake-build-//' \
            -e "s/\${ICF_SUFFIX}/${ICF_SUFFIX:-}/g" \
            -e "s/\$llvm_cs_size/${llvm_cs_size:-<N>}/g" \
      | grep -v '[$]' | sort -u
}

check_target_arg() {  # <known names...> → exit 2 when --target=<glob> matches none
    [ -n "${CHUNK_BUILD_TARGET:-}" ] || return 0
    [ "${CHUNK_ARGS_CHECKED_BY_PARENT:-}" = 1 ] && return 0
    local n
    for n in "$@"; do
        __target_wants "$n" && return 0
    done
    echo "ERROR: --target=$CHUNK_BUILD_TARGET matches no build configuration" >&2
    echo "       did you mean: $(__closest_names "$CHUNK_BUILD_TARGET" "$@" | paste -sd' ')" >&2
    echo "       (a glob over the cmake-build-<name> names of the scripts, e.g. --target='omp-llvm*')" >&2
    exit 2
}

# Per-run markers, in the slot directory both scripts share:
#   exec-matched     touched by the first build dir where --exec matched
#   exec-candidates  the target names of every dir where it did not
__exec_record() {  # matched|<candidate names...>
    local d; d="$(__slot_dir)"
    if [ "$1" = matched ]; then
        : > "$d/exec-matched"
    else
        printf '%s\n' "$@" >> "$d/exec-candidates"
    fi
}

# The cmake target names a configured build dir can build, from the Ninja rule
# names (CXX_EXECUTABLE_LINKER__<target>_<Config>): executables and libraries,
# without the object files, the CTest phonies and the per-dir helpers.
__ninja_target_names() {  # <build-dir> → one name per line (empty if unknown)
    [ -f "$1/build.ninja" ] && command -v ninja >/dev/null 2>&1 || return 0
    ninja -C "$1" -t targets all 2>/dev/null \
      | sed -nE 's/^.*: CXX_(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY)_LINKER__(.*)$/\2/p' \
      | sed -E 's/_(Debug|Release|RelWithDebInfo|MinSizeRel)?$//' | sort -u
}

# End of run: the verdict --target / --exec could not give up front. Returns 1
# (and prints the error) when a filter let nothing through at all.
check_arg_epilogue() {
    [ "${CHUNK_ARGS_CHECKED_BY_PARENT:-}" = 1 ] && return 0   # the parent's job
    local d rc=0; d="$(__slot_dir)"
    if [ -n "${CHUNK_BUILD_TARGET:-}" ] && [ ! -f "$d/target-matched" ]; then
        echo "ERROR: --target=$CHUNK_BUILD_TARGET is a known name but no configuration was selected in this run" >&2
        echo "       (family disabled by --no-<c>/--only=<c>, --debug-only, or not built on this architecture?)" >&2
        rc=1
    fi
    if [ -n "${CHUNK_BUILD_EXEC:-}" ] && [ "${DRYRUN:-false}" != true ] && [ ! -f "$d/exec-matched" ]; then
        echo "ERROR: --exec=$CHUNK_BUILD_EXEC matches no target in any selected build dir" >&2
        if [ -s "$d/exec-candidates" ]; then
            local _cands=()   # the targets of the dirs where it was looked up
            mapfile -t _cands < <(sort -u "$d/exec-candidates")
            echo "       did you mean: $(__closest_names "$CHUNK_BUILD_EXEC" "${_cands[@]}" | paste -sd' ')" >&2
        else
            echo "       (no build dir was configured, so no target list to compare with)" >&2
        fi
        rc=1
    fi
    rm -f "$d/target-matched" "$d/exec-matched" "$d/exec-candidates"
    return $rc
}

declare -A JOB_NAME JOB_LOG JOB_T0 JOB_SLOT
JOB_SLOT_PENDING=""
LOG_DIR=""
N_FAILED=0

# ─── Concurrency cap ─────────────────────────────────────────────────────────
# Without a cap, every compile_with_cmake launched its ninja immediately: 20
# build dirs x CMAKE_BUILD_PARALLEL_LEVEL compilers all resident at once, each
# holding a template-heavy TU in memory. The node ends up thrashing and every
# build crawls in lockstep (observed: 20 builds all at ~55% after 560s).
#
# CHUNK_MAX_JOBS caps how many build DIRS run at a time; launch_build blocks
# until a slot frees. The core budget is then split across those slots instead
# of being multiplied by them, so total compiler processes stay near the core
# count whatever the number of configurations.
#
# Override either knob from the environment: CHUNK_MAX_JOBS=1 serialises
# entirely (useful when a build gets OOM-killed), and an explicitly set
# CMAKE_BUILD_PARALLEL_LEVEL is always respected as-is.
CHUNK_MAX_JOBS="${CHUNK_MAX_JOBS:-4}"

# How many cores to plan for. nproc is the right answer: it already reflects the
# CPU affinity mask, so taskset / scheduler placement are respected.
#
# NOT the cgroup quota. An earlier version divided the cgroup CPU quota by its
# period and used that -- on a 288-core node it reported 1 usable core and gave
# every build a single compile job. The quota throttles CPU TIME on a shared
# node; it is not a bound on how many compilers may run. It is reported below
# for information only, never used to size the build.
__usable_cores() {
    local list n
    # Count the CPUs in the affinity mask, straight from the kernel.
    #
    # NOT nproc: it also honours OMP_NUM_THREADS, so a shell that exported
    # OMP_NUM_THREADS=1 (as every reproduction run here does) makes nproc answer
    # 1 while Cpus_allowed_list says "0-63,128-191" -- 128 CPUs. That is how a
    # 256-core node ended up compiling one TU at a time.
    if [ -r /proc/self/status ]; then
        list=$(awk -F'\t' '/^Cpus_allowed_list:/ {print $2}' /proc/self/status 2>/dev/null)
        if [ -n "$list" ]; then
            n=$(printf '%s' "$list" | awk -F, '{
                    total = 0
                    for (i = 1; i <= NF; i++) {
                        if (split($i, r, "-") == 2) total += r[2] - r[1] + 1
                        else                        total += 1
                    }
                    print total
                 }')
            [ "${n:-0}" -ge 1 ] 2>/dev/null && { echo "$n"; return; }
        fi
    fi
    # Fall back to nproc, ignoring OMP_NUM_THREADS if we can.
    env -u OMP_NUM_THREADS nproc 2>/dev/null || nproc 2>/dev/null || echo 8
}

# Cgroup quota, for the informational line only. Empty when there is none.
__cgroup_quota() {
    local q p
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null
        if [ "${q:-max}" != "max" ] && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
            echo "$(( q / p ))"; return
        fi
    fi
    if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
        q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)
        p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)
        if [ "${q:--1}" -gt 0 ] 2>/dev/null && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
            echo "$(( q / p ))"; return
        fi
    fi
    echo ""
}

# How many compilers the available MEMORY can hold. On this code base a single
# cc1plus on the heavy template TUs peaks in the gigabytes, so the core count is
# not the binding constraint: 128 concurrent compilers on a node whose free RAM
# fits 40 will swap, and every build stalls before finishing its first TU (the
# status line then sits at "starting..." for ten minutes because ninja has not
# completed a single edge).
#
# Budget CHUNK_MEM_PER_JOB_MB per compiler, default 2 GB. Raise it if a build
# still gets OOM-killed, lower it if you know your TUs are lighter.
CHUNK_MEM_PER_JOB_MB="${CHUNK_MEM_PER_JOB_MB:-2048}"

__mem_cap_jobs() {
    local avail_kb n
    [ -r /proc/meminfo ] || { echo ""; return; }
    avail_kb=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null)
    [ -n "$avail_kb" ] || { echo ""; return; }
    n=$(( avail_kb / 1024 / CHUNK_MEM_PER_JOB_MB ))
    [ "$n" -lt 1 ] && n=1
    echo "$n"
}

if [ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
    __ncores="$(__usable_cores)"
    # Take the tighter of the two budgets.
    __memcap="$(__mem_cap_jobs)"
    if [ -n "$__memcap" ] && [ "$__memcap" -lt "$__ncores" ]; then
        __limited_by="memory (~${CHUNK_MEM_PER_JOB_MB} MB/job)"
        __ncores="$__memcap"
    else
        __limited_by="cores"
    fi
    CMAKE_BUILD_PARALLEL_LEVEL=$(( __ncores / CHUNK_MAX_JOBS ))
    [ "$CMAKE_BUILD_PARALLEL_LEVEL" -lt 1 ] && CMAKE_BUILD_PARALLEL_LEVEL=1
    export CMAKE_BUILD_PARALLEL_LEVEL
    __derived="derived, limited by $__limited_by"
else
    __ncores="$(__usable_cores)"
    __derived="from the environment"
fi

# Say it out loud: this is the knob people get wrong, and a build that looks
# hung is usually one compiling with 1-2 cores.
printf '[JOBS] %s usable core(s), %s build(s) at a time, %s compile job(s) each (%s) -> ~%s compiler process(es)\n' \
       "$__ncores" "$CHUNK_MAX_JOBS" "$CMAKE_BUILD_PARALLEL_LEVEL" "$__derived" \
       "$(( CHUNK_MAX_JOBS * CMAKE_BUILD_PARALLEL_LEVEL ))"
__quota="$(__cgroup_quota)"
if [ -n "$__quota" ]; then
    printf '[JOBS] note: cgroup CPU quota is ~%s core-equivalent(s) of CPU TIME on this node.\n' "$__quota"
    printf '[JOBS]       that throttles throughput, not parallelism -- it does not size the build.\n'
fi
if [ "$CMAKE_BUILD_PARALLEL_LEVEL" -le 2 ] && [ "$CHUNK_MAX_JOBS" -gt 1 ]; then
    printf '[JOBS] only %s job(s) per build -- each configuration will take a long time.\n' \
           "$CMAKE_BUILD_PARALLEL_LEVEL"
    printf '[JOBS] fewer, wider builds are usually faster here: try --jobs=1 or --jobs=2.\n'
fi
unset __ncores __derived __quota __memcap __limited_by

launch_build() {  # <build-dir> [cmake --build extra args…]
    local dir=$1; shift
    # Readable name: the cmake-build-* path component, without prefix or $arch.
    local name=${dir#"$build_rep"/}; name=${name%%/*}; name=${name#cmake-build-}
    __TARGET_SEEN=$(( __TARGET_SEEN + 1 ))
    if ! __target_wants "$name"; then
        # Counted, not printed: target_filter_summary() says how many were
        # skipped, once, instead of a line per configuration.
        __PENDING_HEADER=""
        return 0
    fi
    __TARGET_KEPT=$(( __TARGET_KEPT + 1 ))
    : > "$(__slot_dir)/target-matched"   # seen by check_arg_epilogue
    __flush_header
    if [ "$DRYRUN" = true ]; then
        # Trailing shell comment: the echoed line stays copy-pastable as-is.
        local _dry_exec=""
        if [ -n "${CHUNK_BUILD_EXEC:-}" ]; then
            case "$CHUNK_BUILD_EXEC" in
                *[*?[]*) _dry_exec="--target <targets matching '$CHUNK_BUILD_EXEC'> " ;;
                *)       _dry_exec="--target $CHUNK_BUILD_EXEC " ;;
            esac
        fi
        echo "cmake --build $dir ${_dry_exec}$*  # → log $name.log"
        return
    fi
    # --exec=<glob>: build only the Ninja target(s) matching <glob> (e.g. one
    # executable, or a family like '*_dyn' / '*_256') instead of the whole build
    # dir. CHUNK_BUILD_EXEC holds the pattern. A literal name (no glob chars) is
    # passed straight through — cheapest path, unchanged behaviour. A pattern is
    # resolved against the real target list (`ninja -t targets all`) so we can
    # feed cmake several concrete `--target` names (cmake --build --target does
    # NOT accept globs). ${x:+...} expands to nothing when unset, so the default
    # (build everything) is byte-identical.
    # Resolved BEFORE the slot is taken: a dir that turns out to build nothing
    # returns from here, and used to walk off with the slot it had just waited
    # for.
    local _exec_target=()
    if [ -n "${CHUNK_BUILD_EXEC:-}" ]; then
        # The dir's real targets, when it is configured (else we cannot check
        # and a literal name goes through as before).
        local _names=() _t
        mapfile -t _names < <(__ninja_target_names "$dir")
        case "$CHUNK_BUILD_EXEC" in
            *[*?[]*)
                # Pattern: every matching target, fed to cmake as concrete names
                # (cmake --build --target does NOT accept globs).
                local _matches=()
                for _t in "${_names[@]}"; do
                    # shellcheck disable=SC2254
                    case "$_t" in
                        $CHUNK_BUILD_EXEC) _matches+=(--target "$_t") ;;
                    esac
                done
                if [ ${#_matches[@]} -eq 0 ]; then
                    printf '[WARN] %-40s no Ninja target matches --exec=%s (building nothing here)\n' \
                           "$name" "$CHUNK_BUILD_EXEC"
                    # Nothing matched: skip this build dir entirely rather than
                    # falling back to a full build (which would surprise the user).
                    __exec_record "${_names[@]}"
                    return 0
                fi
                _exec_target=("${_matches[@]}")
                ;;
            *)
                # Literal target name: looked up when the list is known, so a
                # typo is a [WARN] here and an error at the end (with the
                # closest name) instead of a ninja "unknown target" in a log.
                if [ ${#_names[@]} -gt 0 ]; then
                    local _found=false
                    for _t in "${_names[@]}"; do
                        [ "$_t" = "$CHUNK_BUILD_EXEC" ] && { _found=true; break; }
                    done
                    if [ "$_found" = false ]; then
                        printf '[WARN] %-40s no Ninja target named --exec=%s (building nothing here)\n' \
                               "$name" "$CHUNK_BUILD_EXEC"
                        __exec_record "${_names[@]}"
                        return 0
                    fi
                fi
                _exec_target=(--target "$CHUNK_BUILD_EXEC")
                ;;
        esac
        __exec_record matched
    fi
    # Wait for a free slot BEFORE starting: this is what keeps the node from
    # running every configuration at once.
    __wait_for_slot

    if [ -z "$LOG_DIR" ]; then
        LOG_DIR="$build_rep/logs/$(date +%Y%m%d-%H%M%S)-$$"
        mkdir -p "$LOG_DIR"
        echo "[INFO] build logs → $LOG_DIR"
    fi
    local log="$LOG_DIR/$name.log"
    # setsid: dedicated session (pgid = pid) → the INT trap can kill the whole
    # tree (cmake + ninja + compilers) via kill -- -pid. Since the child is not
    # a group leader, setsid execs without forking: $! remains our direct child
    # (wait/kill -0 unchanged).
    if command -v setsid >/dev/null 2>&1; then
        setsid cmake --build "$dir" "${_exec_target[@]}" "$@" >"$log" 2>&1 &
    else
        cmake --build "$dir" "${_exec_target[@]}" "$@" >"$log" 2>&1 &
    fi
    local pid=$!
    JOB_NAME[$pid]=$name
    JOB_LOG[$pid]=$log
    JOB_T0[$pid]=$SECONDS
    JOB_SLOT[$pid]="$JOB_SLOT_PENDING"   # released by __report_job
    JOB_SLOT_PENDING=""
    printf '[LAUNCH] %s (pid %d)\n' "$name" "$pid"
}

# ^C: async jobs start with SIGINT ignored (POSIX behavior of non-interactive
# shells) — a Ctrl-C used to kill the script but left every ninja running, now
# silently since their output goes to the logs. cancel_all kills each build
# group (setsid session) then exits. Installed at source time: applies to
# compile_all.sh AND compile_all_gnu.sh (each kills its own jobs, both receive
# the terminal's INT).
cancel_all() {
    trap - INT TERM   # no re-entrance
    __clear_status
    echo
    echo "[INT] interrupted — stopping in-flight builds:"
    local pid
    for pid in "${!JOB_NAME[@]}"; do
        printf '       stop %-50s (pid %d)\n' "${JOB_NAME[$pid]}" "$pid"
        # Whole group when in a dedicated session; fall back to the bare pid.
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        __slot_release "${JOB_SLOT[$pid]:-}"
    done
    wait 2>/dev/null   # reap before returning (no zombie orphans)
    echo "[INT] builds stopped (logs kept in $LOG_DIR)"
    exit 130
}
trap cancel_all INT TERM

# ─── Live progress ───────────────────────────────────────────────────────────
# Ninja writes its "[N/M] ..." progress into each build log; we re-read the log
# tails periodically and render one [RUN ] line per in-flight build. On a real
# terminal (auto-detected, override with BUILD_STATUS=on|off) the block is
# redrawn in place with ANSI cursor moves every ~2s; otherwise a plain snapshot
# is printed every ~30s (safe for redirected output / CI logs).
BUILD_STATUS="${BUILD_STATUS:-auto}"
__STATUS_N=0   # lines currently on screen (ANSI mode)

__status_enabled() {
    case "$BUILD_STATUS" in
        on)  return 0 ;;
        off) return 1 ;;
        *)   [ -t 1 ] && [ "${TERM:-dumb}" != dumb ] ;;
    esac
}

__term_cols() {  # terminal width, re-read every draw (a window can be resized)
    local c
    c=$(tput cols 2>/dev/null) || c=""
    [ -n "$c" ] || c=${COLUMNS:-80}
    [ "$c" -gt 20 ] 2>/dev/null || c=80
    printf '%s' "$c"
}

__clear_status() {  # erase the previously drawn [RUN ] block (ANSI mode only)
    [ "$__STATUS_N" -gt 0 ] || return 0
    printf '\033[%dA\033[J' "$__STATUS_N"
    __STATUS_N=0
}

__job_progress() {  # <log> → "[N/M] (P%)" from the log tail, or "configuring…"
    local prog n m
    prog=$(tail -c 8192 "$1" 2>/dev/null | grep -oE '\[[0-9]+/[0-9]+\]' | tail -1)
    if [ -z "$prog" ]; then printf 'starting…'; return; fi
    n=${prog#[}; m=${n#*/}; m=${m%]}; n=${n%%/*}
    if [ "$m" -gt 0 ] 2>/dev/null; then
        printf '%s (%d%%)' "$prog" $(( 100 * n / m ))
    else
        printf '%s' "$prog"
    fi
}

__draw_status() {  # one [RUN ] line per tracked job, pid-sorted (stable order)
    # Every line is TRUNCATED to the terminal width. __clear_status erases by
    # moving up __STATUS_N ROWS, so a line that wraps makes the block taller
    # than the count and the erase lands in the middle of it: the display then
    # scrolls one block per refresh instead of redrawing in place. The old
    # format was exactly 80 columns wide with the longest config name
    # ("omp-gnu-relwithdebinfo-std-alloc-address-sanitizer", 49 chars), which
    # wrapped on every 80-column terminal. The name field now takes whatever
    # room is left, and the line is cut to width - 1 whatever happens.
    local pid dt line cols name_w
    local cols=$(__term_cols)
    local lines=0
    # "[RUN ] " + name + " " + "9999s" + "  " + "[1234/5678] (100%)" -> the name
    # gets the rest, never less than 12 columns.
    local name
    name_w=$(( cols - 34 ))
    [ "$name_w" -ge 12 ] || name_w=12
    for pid in $(printf '%s\n' "${!JOB_NAME[@]}" | sort -n); do
        dt=$(( SECONDS - JOB_T0[$pid] ))
        # A name too long is cut at the FRONT, not the back: these names share
        # long prefixes (omp-gnu-, omp-llvm-) and differ at the end, so
        # "omp-gnu-relwithdebinfo-std-alloc-address-sanitizer" and its
        # -thread-sanitizer twin are told apart by their last characters.
        name=${JOB_NAME[$pid]}
        [ "${#name}" -le "$name_w" ] || name="…${name: -$(( name_w - 1 ))}"
        line=$(printf '[RUN ] %-*s %4ds  %s' "$name_w" "$name" "$dt" \
                      "$(__job_progress "${JOB_LOG[$pid]}")")
        printf '%.*s\n' "$(( cols - 1 ))" "$line"
        lines=$(( lines + 1 ))
    done
    __STATUS_N=$lines
}

__report_job() {  # <pid> <rc> — print the status and remove the job from the table
    local pid=$1 rc=$2
    __clear_status
    local name=${JOB_NAME[$pid]}
    local dt=$(( SECONDS - JOB_T0[$pid] ))
    if [ "$rc" -eq 0 ]; then
        printf '[ OK ] %-50s %4ds\n' "$name" "$dt"
    else
        N_FAILED=$(( N_FAILED + 1 ))
        printf '[FAIL] %-50s %4ds  rc=%d  → %s\n' "$name" "$dt" "$rc" "${JOB_LOG[$pid]}"
        # Montrer l'ERREUR, pas la fin du fichier. `tail -n 8` etait trompeur :
        # un seul warning gcc sur du template inline occupe dix lignes de chaine
        # « inlined from », si bien que la queue du log affichait un warning
        # anodin pendant que la vraie erreur, plus haut, restait invisible -- on
        # a perdu du temps a soupconner un -Werror qui n'existe pas sur ces
        # configs (il est conditionne a CHUNK_WERROR ET au type Debug).
        # On cherche donc le premier bloc « FAILED: » de ninja, ou a defaut la
        # premiere ligne « error: », et on retombe sur la queue s'il n'y a ni
        # l'un ni l'autre (echec de configure, OOM tue par le noyau, rc=127...).
        local __log=${JOB_LOG[$pid]} __ln
        __ln=$(grep -n -m1 -E '^FAILED:|(^|[ /])[Ee]rror:' "$__log" 2>/dev/null | cut -d: -f1)
        if [ -n "$__ln" ]; then
            echo "       ── premiere erreur (ligne $__ln) ──"
            sed -n "${__ln},$((__ln + 14))p" "$__log" | sed 's/^/       /'
        else
            echo '       ── last lines (aucune ligne d erreur trouvee) ──'
            tail -n 8 "$__log" | sed 's/^/       /'
        fi
    fi
    __slot_release "${JOB_SLOT[$pid]:-}"
    unset "JOB_NAME[$pid]" "JOB_LOG[$pid]" "JOB_T0[$pid]" "JOB_SLOT[$pid]"
}

# Reap exactly ONE finished tracked job and report it. Returns 0 if a job was
# reaped, 1 otherwise (untracked child, or nothing left). Shares the pid-scan
# trick documented in wait_all: `wait -n` tells us a child ended, and the
# tracked pid that no longer answers kill -0 is the one it belongs to.
__reap_one() {
    local rc reaped pid alive
    wait -n; rc=$?
    reaped=""
    for pid in "${!JOB_NAME[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then reaped=$pid; break; fi
    done
    if [ -z "$reaped" ]; then
        if [ "$rc" -eq 127 ]; then
            alive=false
            for pid in "${!JOB_NAME[@]}"; do
                kill -0 "$pid" 2>/dev/null && { alive=true; break; }
            done
            [ "$alive" = true ] && return 1
            return 1
        fi
        return 1
    fi
    __report_job "$reaped" "$rc"
    return 0
}

# ─── Cross-process slot semaphore ────────────────────────────────────────────
# The in-memory job table only knows about THIS process's builds. compile_all.sh
# launches oneapi/llvm itself and then delegates gnu to compile_all_gnu.sh, a
# separate process that sources this file again and starts its own table -- so
# two caps of 4 coexisted and 9 ninjas ran at once, each sized for a node it did
# not have to itself. The gnu builds then sat at "starting..." behind oneapi
# builds that had the machine.
#
# A directory of token files, shared by every process of the run, fixes that:
# taking a slot is `mkdir` (atomic), releasing it is `rmdir`. CHUNK_SLOT_DIR is
# exported so the child script joins the SAME pool instead of opening its own.
__slot_dir() {
    if [ -z "${CHUNK_SLOT_DIR:-}" ]; then
        CHUNK_SLOT_DIR="${build_rep:-/tmp}/.build-slots.$$"
        export CHUNK_SLOT_DIR
    fi
    mkdir -p "$CHUNK_SLOT_DIR" 2>/dev/null
    printf '%s' "$CHUNK_SLOT_DIR"
}

# Drop stale tokens whose owning process is gone (a killed run leaves them).
__slots_gc() {
    local d tok pid
    d="$(__slot_dir)"
    for tok in "$d"/slot-*; do
        [ -d "$tok" ] || continue
        pid=$(cat "$tok/owner" 2>/dev/null)
        case "$pid" in ''|*[!0-9]*) continue ;; esac
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$tok/owner" 2>/dev/null
            rmdir "$tok" 2>/dev/null
        fi
    done
}

__slots_held() {
    local d n
    d="$(__slot_dir)"
    n=$(find "$d" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    printf '%s' "${n:-0}"
}

# Try to take a slot ONCE. Sets JOB_SLOT_PENDING and returns 0 on success, 1 if
# every slot is busy. Prints nothing.
#
# The slot names are FIXED (slot-0 .. slot-N-1) and mkdir is the whole test:
# only one process can create a given name, so acquisition is atomic. Counting
# the tokens first and then creating a uniquely-named one is a check-then-act
# race -- two processes both see "room left" and both succeed, which measured
# cap+1 builds in flight.
#
# Deliberately NOT a function that echoes its result: it used to be called as
# JOB_SLOT_PENDING="$(__slot_acquire)", which ran the whole wait loop in a
# SUBSHELL. The progress display it drew went into the captured string instead
# of the terminal (and corrupted the token path), and __reap_one could not
# update the parent's job table -- so nothing was reaped and the [RUN ] lines
# froze. Setting a global from the current shell avoids both.
__slot_try() {
    local d tok k
    d="$(__slot_dir)"
    __slots_gc
    k=0
    while [ "$k" -lt "${CHUNK_MAX_JOBS:-4}" ]; do
        tok="$d/slot-$k"
        if mkdir "$tok" 2>/dev/null; then
            # Record the owner so __slots_gc can reclaim it if we die.
            printf '%s' "$$" > "$tok/owner" 2>/dev/null
            JOB_SLOT_PENDING="$tok"
            return 0
        fi
        k=$(( k + 1 ))
    done
    return 1
}

__slot_release() {  # <token path>
    [ -n "${1:-}" ] || return 0
    rm -f "$1/owner" 2>/dev/null
    rmdir "$1" 2>/dev/null
    return 0
}

# Block until this process may start another build.
__wait_for_slot() {
    [ "${CHUNK_MAX_JOBS:-0}" -gt 0 ] || return 0
    JOB_SLOT_PENDING=""
    local last_snapshot=$SECONDS ticker=""
    until __slot_try; do
        # Keep the display live and our own finished builds reaped while we wait
        # -- in THIS shell, so the job table and the [RUN ] block are the real
        # ones.
        if __status_enabled; then
            __clear_status
            __draw_status
        elif [ $(( SECONDS - last_snapshot )) -ge 30 ]; then
            __draw_status; __STATUS_N=0
            last_snapshot=$SECONDS
        fi
        if [ "${#JOB_NAME[@]}" -gt 0 ]; then
            # Same refresh ticker as wait_all: __reap_one calls `wait -n`, which
            # blocks until SOME child exits. With only long builds running that
            # is minutes away, and the [RUN ] block would sit frozen the whole
            # time. An UNtracked `sleep 2 &` is a child too, so its exit wakes
            # `wait -n` on schedule and the loop redraws. It is not in JOB_NAME,
            # so __reap_one finds no tracked pid gone and simply returns 1.
            if [ -z "$ticker" ] || ! kill -0 "$ticker" 2>/dev/null; then
                sleep 2 & ticker=$!
            fi
            __reap_one || :
        else
            sleep 2
        fi
    done
    if [ -n "$ticker" ]; then kill "$ticker" 2>/dev/null; wait "$ticker" 2>/dev/null; fi
    __clear_status
}

wait_all() {
    [ "$DRYRUN" = true ] && return 0
    local pid rc reaped alive

    # wait -n -p <var> (bash ≥ 5.1) would be ideal, but not every front-end
    # node has bash 5.1. Portable path (bash ≥ 4.3): wait -n reaps ONE child;
    # we identify which one by scanning our pids — the one that vanished
    # (kill -0 fails). A job that finished but was not waited on yet stays a
    # zombie, so kill -0 still sees it: at most one tracked pid disappears per
    # iteration, and the wait -n rc belongs to it.
    if ! ( : & wait -n 2>/dev/null ); then
        # bash < 4.3: no wait -n — statuses are correct but reported in
        # launch order, not completion order.
        for pid in "${!JOB_NAME[@]}"; do
            wait "$pid"; rc=$?
            __report_job "$pid" "$rc"
        done
        return "$N_FAILED"
    fi

    # Refresh ticker: an UNtracked `sleep 2 &` child whose exit wakes wait -n
    # up even when no build finished, so the [RUN ] block gets redrawn. In
    # non-terminal mode, print a plain snapshot roughly every 30s instead.
    local ticker="" last_snapshot=$SECONDS
    while [ "${#JOB_NAME[@]}" -gt 0 ]; do
        if __status_enabled; then
            __clear_status
            __draw_status
        elif [ $(( SECONDS - last_snapshot )) -ge 30 ]; then
            __draw_status; __STATUS_N=0   # plain snapshot: nothing to erase
            last_snapshot=$SECONDS
        fi
        if [ -z "$ticker" ] || ! kill -0 "$ticker" 2>/dev/null; then
            sleep 2 & ticker=$!
        fi
        wait -n; rc=$?
        reaped=""
        for pid in "${!JOB_NAME[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then reaped=$pid; break; fi
        done
        if [ -z "$reaped" ]; then
            # None of our pids vanished. rc=127 is ambiguous: "no children
            # left" OR an UNtracked child (subshell, module) that exited with
            # 127. If our jobs are still alive (zombies included), keep going.
            if [ "$rc" -eq 127 ]; then
                alive=false
                for pid in "${!JOB_NAME[@]}"; do
                    kill -0 "$pid" 2>/dev/null && { alive=true; break; }
                done
                [ "$alive" = true ] && continue
                echo "[WARN] wait -n: no children left"; break
            fi
            continue  # untracked child reaped: nothing to report
        fi
        # NB: a job exiting with rc=127 (e.g. cmake not found, module not
        # loaded) is reaped, hence detected here and reported [FAIL] rc=127 —
        # it does not go through the ambiguous branch above.
        __report_job "$reaped" "$rc"
    done
    __clear_status
    if [ -n "$ticker" ]; then kill "$ticker" 2>/dev/null; wait "$ticker" 2>/dev/null; fi
    return "$N_FAILED"
}

create_build_rep()
{
# Check if the build directory exists, if not, create it
if [ ! -d "$build_rep" ]; then
    echo "The build directory $build_rep does not exist. Creating it..."
    mkdir -p "$build_rep"
    if [ $? -ne 0 ]; then
        echo "Error creating the build directory $build_rep"
        exit 1
    fi
else
    echo "The build directory $build_rep already exists."
fi
}

# Function to compile with CMake and Ninja.
# Synchronous configure (configure errors stay readable on screen and count
# into N_FAILED), asynchronous build via launch_build (log + tracking).
compile_with_cmake() {
    local build_dir=$1
    local cmake_args=$2
    local build_name=$3
    # Optional 4th argument; defaults to the caller-exported CXX (the old
    # code passed CXX="" to cmake when $4 was absent).
    local compiler=${4:-$CXX}

    local __name=${build_dir#"$build_rep"/}; __name=${__name%%/*}; __name=${__name#cmake-build-}
    if ! __target_wants "$__name"; then
        return 0   # counted and summarised by launch_build / target_filter_summary
    fi

    echo -e "\nCompiling $build_name:\n"

    # Hardening flags derived from the flavour (Debug / sanitizer) -- same rule
    # as run_cmd applies to the other compilers, see hardening_opts_for.
    local _hard
    _hard=$(hardening_opts_for "$cmake_args")
    [ -n "$_hard" ] && cmake_args="$cmake_args $_hard"

    if [ "$DRYRUN" = true ]; then
        if [ "$BUILD_ONLY" = false ]; then
            maybe_wipe_cache "$build_dir"
            echo "CXX=${compiler} cmake -S ~/shrink-array -B \"$build_dir\" $cmake_args -G Ninja"
        fi
        launch_build "$build_dir"
        return 0
    fi

    if [ "$BUILD_ONLY" = false ]; then
        maybe_wipe_cache "$build_dir"
        if ! CXX="${compiler}" cmake -S ~/shrink-array -B "$build_dir" $cmake_args -G Ninja; then
            echo "Error configuring CMake for $build_name"
            N_FAILED=$(( N_FAILED + 1 ))  # visible in the final exit code
            return 1
        fi
    fi

    launch_build "$build_dir"
}
