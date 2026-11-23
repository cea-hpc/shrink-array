#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
# jobserver_broker.sh <weight> <command...> -- runs <command> (a cmake --build
# whose ninja is >= 1.13) on a jobserver of its own, each token of which costs
# <weight> tokens of the run's pool ($CHUNK_JOBSERVER_FIFO).
#
# Why: a sanitizer compile job needs CHUNK_MEM_PER_JOB_MB_SANITIZER, twice a
# normal one, so when the memory is what limits the run it weighs 2 tokens. A
# ninja takes 1 token per job and cannot be told otherwise, so these builds
# used to get a fixed -j at launch and keep it: the last one left, an ASan
# build, compiled on its launch share of the node for twenty minutes while the
# rest of it sat idle. Here ninja sees a pool of its own (MAKEFLAGS, a second
# FIFO), which this script fills one token per <weight> tokens taken from the
# run's pool.
#
# Ninja 1.13 reads its pool only when one of its jobs ends (measured: tokens
# freed in the middle of a job wait for the next end), so its need shows
# there, and that is when this script looks:
#   - after a job of ninja ended (its child processes, read from /proc), a
#     token still in the sub-pool is one ninja had no use for: all of them go
#     back to the run's pool, <weight> each, and the next feed is halved;
#   - the sub-pool empty, and ninja took everything at the last job end (or
#     it is starting): feed it, up to <burst> tokens at once (each <weight>
#     tokens of the run's pool, taken as they come). Emptied again by the next
#     job end, the burst doubles -- a build left alone climbs to the whole
#     node in a few job ends --, up to size / weight. A job end that left
#     tokens unused stops the feed until a job end that leaves none;
#   - tokens left in the sub-pool CHUNK_BROKER_IDLE seconds with no job end
#     (or /proc without the children list): one goes back per second;
#   - at the end (or on TERM/INT), all the sub-pool was given goes back,
#     counted, not read back: a ninja killed holding tokens loses none.
# The job ninja runs without a token is the caller's to cover (launch_build
# holds <weight> tokens for it).
#
# Variables: CHUNK_JOBSERVER_FIFO (the run's pool, required),
# CHUNK_JOBSERVER_SIZE (its size), CHUNK_BROKER_IDLE (30 s).
# Exit status: the command's.
set -u
weight=$1; shift
main=${CHUNK_JOBSERVER_FIFO:?}
size=${CHUNK_JOBSERVER_SIZE:-2}
idle_turns_max=$(( ${CHUNK_BROKER_IDLE:-30} * 10 ))
max=$(( size / weight ))
[ "$max" -ge 1 ] || max=1

tmp=$(mktemp -d "${TMPDIR:-/tmp}/chunk-broker.XXXXXX") || exit 1
sub=$tmp/fifo
mkfifo -m 600 "$sub" || { rmdir "$tmp"; exit 1; }
exec {mfd}<>"$main" {sfd}<>"$sub"

issued=0      # tokens put into the sub-pool and not taken back
held=0        # tokens of the run's pool taken toward the next sub-token
burst=$max     # a build that starts has work for all it can get
feed=true     # ninja took all it was given at the last job end
ninja_pid=""
kids=""       # the pids of ninja's jobs at the last look
end_turns=0   # turns to wait after a job end before reading the sub-pool
idle_turns=0  # turns with tokens in the sub-pool and no job end

give_main() {  # <n>
    [ "$1" -gt 0 ] && printf "%${1}s" "" | tr ' ' '+' >&"$mfd"
    return 0
}

# Take back every token waiting in the sub-pool; sets $n.
drain() {
    local c
    n=0
    while read -r -t 0 -u "$sfd" && IFS= read -r -N 1 -t 0.05 -u "$sfd" c; do
        n=$(( n + 1 ))
    done
}

settle() {
    drain
    give_main $(( issued * weight + held ))
    issued=0 held=0
    exec {sfd}>&- {mfd}>&-
    rm -f "$sub"; rmdir "$tmp" 2>/dev/null
}

child=""
on_signal() {
    trap - TERM INT HUP
    [ -n "$child" ] && kill -TERM "$child" 2>/dev/null && wait "$child" 2>/dev/null
    settle
    exit 143
}
trap on_signal TERM INT HUP

# The children of <pid> (Linux >= 3.5 with CONFIG_PROC_CHILDREN), in $ch.
children_of() {
    ch=()
    [ -r "/proc/$1/task/$1/children" ] && read -r -a ch < "/proc/$1/task/$1/children"
    return 0
}

# Find the ninja: the command itself, or under it (cmake --build → ninja),
# depth <= 3.
find_ninja() {
    local level=("$child") next p comm d
    read -r comm < "/proc/$child/comm" 2>/dev/null &&
        case "$comm" in ninja*) ninja_pid=$child; return 0 ;; esac
    for d in 1 2 3; do
        next=()
        for p in "${level[@]}"; do
            children_of "$p"
            for p in "${ch[@]}"; do
                read -r comm < "/proc/$p/comm" 2>/dev/null || continue
                case "$comm" in ninja*) ninja_pid=$p; return 0 ;; esac
                next+=("$p")
            done
        done
        level=("${next[@]}")
        [ "${#level[@]}" -gt 0 ] || return 1
    done
    return 1
}

# The child gets neither descriptor: ninja opens the sub-pool by its path.
MAKEFLAGS="-j$max --jobserver-auth=fifo:$sub" "$@" {mfd}>&- {sfd}>&- &
child=$!

while kill -0 "$child" 2>/dev/null; do
    # A job of ninja ended? Then, 0.3 s later (ninja has given back that job's
    # token and taken what it could start), what is left was not needed.
    [ -z "$ninja_pid" ] && find_ninja
    if [ -n "$ninja_pid" ]; then
        # A pid of the last look that is gone: a job ended, even when another
        # one started in between (the count alone would miss it).
        children_of "$ninja_pid"
        now=" ${ch[*]} "
        for p in $kids; do
            [[ $now == *" $p "* ]] || { end_turns=3; break; }
        done
        kids=${ch[*]}
    fi
    if [ "$end_turns" -gt 0 ]; then
        end_turns=$(( end_turns - 1 ))
        if [ "$end_turns" -eq 0 ]; then
            drain
            if [ "$n" -gt 0 ]; then
                issued=$(( issued - n ))
                give_main $(( n * weight ))
                burst=$(( burst > 1 ? burst / 2 : 1 ))
                feed=false
            else
                [ "$feed" = true ] && burst=$(( burst * 2 > max ? max : burst * 2 ))
                feed=true
            fi
            idle_turns=0
        fi
        sleep 0.1
        continue
    fi
    if read -r -t 0 -u "$sfd"; then
        # Tokens waiting for the next job end; too long without one: one back.
        idle_turns=$(( idle_turns + 1 ))
        if [ "$idle_turns" -ge "$idle_turns_max" ]; then
            if IFS= read -r -N 1 -t 0.05 -u "$sfd" c; then
                issued=$(( issued - 1 ))
                give_main "$weight"
                burst=1
                feed=false
            fi
            idle_turns=$(( idle_turns_max - 10 ))
        fi
        sleep 0.1
        continue
    fi
    idle_turns=0
    [ "$feed" = true ] || { sleep 0.1; continue; }
    # The sub-pool is empty: feed up to <burst> tokens, each waiting at most
    # 0.2 s for a token of the run's pool (the loop keeps watching).
    fed=0
    while [ "$fed" -lt "$burst" ] && [ "$issued" -lt "$max" ]; do
        IFS= read -r -N 1 -t 0.2 -u "$mfd" c || break
        held=$(( held + 1 ))
        if [ "$held" -ge "$weight" ]; then
            printf '+' >&"$sfd"
            issued=$(( issued + 1 ))
            held=0
            fed=$(( fed + 1 ))
        fi
    done
    [ "$fed" -eq 0 ] && sleep 0.1
done
wait "$child"; rc=$?
settle
exit "$rc"
