# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""--pool: a campaign shared between the batch jobs given the same pool id
(shrink_gui's "Nœuds" field: N one-node jobs, the same command plus
--pool=<id>). The claims themselves are bench_runner/pool.py's (an os.mkdir in
the pool directory, <output-dir>/<arch>/.pools/<id>); this module says what is
claimed and keeps the job's pool for the phases, set once by cli.setup_outputs.

The unit of work is a (phase, configuration) pair, claimed as
<phase>/<configuration> by the first job that reaches it, the others skipping
it: every job walks the phases in the usual order, so the load balances
itself and a job that starts late picks up what is left. Each phase says at
its claim what its unit is when it is not one configuration (perf: a base
configuration and the compression-only ones that inherit its baseline;
seq_omp_compare: a seq/omp pair; stl: a front-end's representative; fuzz: a
case index). A configuration's oracle is generated once, by the job that
claims oracle_<configuration>; the others wait for it (wait_oracle).

Without --pool, `pool` stays None and every function here answers as if this
job owned everything: the phases read the same with or without."""
import os
import sys
import time

from .term import dim, green, yellow

pool = None             # bench_runner.pool.Pool of this job under --pool, else None
units = [0, 0]          # [units this job claimed, units it left to the others]
_mine = set()           # the claim names counted in units[0] (take may be asked twice)
_others = set()         # and those counted in units[1]

# How long a job waits for an oracle another job of the pool generates before
# giving up on it (its checks skipped, an error recorded): a full-size oracle
# takes minutes, and its generator may still be busy with an earlier unit. The
# variable multi_launch_stress.py reads for its references, one setting for both.
ORACLE_WAIT_S = int(os.environ.get("CHUNK_POOL_REF_WAIT", 6 * 3600))
# How often it looks (seconds): the pool directory is on a shared file system,
# read by every waiting job.
POLL_S = float(os.environ.get("CHUNK_POOL_POLL", 2.0))


class PoolStopped(Exception):
    """Another job of the pool stopped it (<pool>/stop: interrupted, or died
    on an exception): this one claims nothing more, writes what it has and
    ends. str(): the stop file's "<job>: <why>"."""


def active():
    return pool is not None and pool.dir is not None


def tag():
    """This job's short name (n<K> under shrink_gui), "" without --pool."""
    return pool.tag if active() else ""


def oracle_root(args):
    """The directory oracle/<config>/ lives under: the architecture's output
    directory, shared by the jobs of a pool (args.output_dir is then the
    job's own sub-directory); args.output_dir itself without --pool."""
    return getattr(args, "shared_dir", None) or args.output_dir


def check_stopped():
    """Raise PoolStopped when another job stopped the pool."""
    if active():
        why = pool.stopped()
        if why:
            raise PoolStopped(why)


def take(phase, unit, label=None):
    """True when this job does @p unit of @p phase: it claimed
    <phase>/<unit> now or earlier (a second call answers the same), or there
    is no pool. False: another job's, said in one dim line. Reads the pool's
    stop file first (PoolStopped)."""
    if not active():
        return True
    check_stopped()
    name = "%s/%s" % (phase, unit)
    if pool.claim(name):
        units[0] += name not in _mine
        _mine.add(name)
        return True
    units[1] += name not in _others
    _others.add(name)
    print(dim("   [pool] %s %s: done by %s" % (phase, label or unit, who(name))))
    return False


def register():
    """After Pool.open: <pool>/jobs/<owner name> holds this job's tag, so
    that the others name it n<K> rather than <host>.<pid> (who)."""
    if active():
        try:
            os.makedirs(pool.path("jobs"), exist_ok=True)
            with open(pool.path("jobs", pool.me), "w") as f:
                f.write(pool.tag + "\n")
        except OSError:
            pass


def who(name):
    """The job holding claim @p name, by its tag when it registered one."""
    owner = pool.owner(name)
    if not owner:
        return "another job"
    try:
        with open(pool.path("jobs", owner)) as f:
            return f.read().strip() or owner
    except OSError:
        return owner


def summary_line():
    """"[pool] <id>: n1 did 7 unit(s), left 5 to the other jobs", "" without --pool."""
    if not active():
        return ""
    return "[pool] %s: %s did %d unit(s), left %d to the other jobs" % (pool.id, pool.tag, units[0], units[1])


def stop(why):
    """This job ends abnormally (interrupt, exception): stop the others too,
    since the units it claimed will not be done and an oracle it was making
    would be waited for in vain."""
    if active():
        pool.stop(why)
        print(yellow("🛑 [pool] %s: stopped for every job (%s)" % (pool.id, why)))


def _oracle_claim(cfg):
    return "oracle_" + cfg


def claim_oracle(cfg):
    """True when this job generates @p cfg's oracle (or there is no pool)."""
    return not active() or pool.claim(_oracle_claim(cfg))


def oracle_done(cfg, ok):
    """After a claimed generation, successful or not: the waiters go on
    (a failed one leaves them the same missing reference it leaves here)."""
    if active():
        pool.mark(_oracle_claim(cfg), "done", "ok\n" if ok else "failed\n")


def check_unit_oracle_settled(cfg):
    """The owner of check/<cfg> is past its oracle step (generated, failed,
    or nothing to do): closes the window between its check claim and its
    oracle claim, in which a waiter would see neither in progress."""
    if active():
        pool.mark("check/" + cfg, "oracle")


def _pending(cfg):
    """The claim another job holds on @p cfg's oracle and has not settled,
    or None: check/<cfg> without its "oracle" marker, oracle_<cfg> without
    its "done"."""
    for name, marker in (("check/" + cfg, "oracle"), (_oracle_claim(cfg), "done")):
        if os.path.isdir(pool.path(name)) and pool.owner(name) != pool.me \
                and not pool.marked(name, marker):
            return name
    return None


def wait_oracle(cfg, label):
    """Wait, polling, until no other job of the pool is making @p cfg's
    oracle; True then (whether it is complete is the caller's test, as
    without --pool). False after ORACLE_WAIT_S: the generator died (time
    limit, node lost) without stopping the pool, or is far too slow -- the
    caller skips what needs the oracle and records why. PoolStopped when the
    pool is stopped meanwhile. Without --pool: True at once."""
    if not active():
        return True
    t0, said = time.time(), False
    while True:
        name = _pending(cfg)
        if name is None:
            break
        check_stopped()
        if time.time() - t0 > ORACLE_WAIT_S:
            return False
        if not said:
            sys.stdout.write("\n" + dim("   ⏳ [pool] oracle of %s made by %s: waiting for it..."
                                        % (label, who(name))) + "\n")
            sys.stdout.flush()
            said = True
        time.sleep(POLL_S)
    if said:
        print(green("   ✅ [pool] oracle of %s ready after %.0fs" % (label, time.time() - t0)))
    return True


def oracle_timeout_entry(cfg, label):
    """The error recorded when wait_oracle gave up."""
    holder = _pending(cfg)
    return ("POOL_ORACLE_TIMEOUT: the oracle of %s, claimed by %s, still not ready after %ds "
            "(no %s marker): what needs it is skipped in this job (CHUNK_POOL_REF_WAIT sets the wait)"
            % (label, who(holder) if holder else "another job", ORACLE_WAIT_S,
               pool.path(holder or _oracle_claim(cfg), "done")))
