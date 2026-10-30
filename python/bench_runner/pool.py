# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""--pool: the work shared between the batch jobs given the same pool id
(shrink_gui's "Nœuds" field submits N one-node jobs <id>-n1..nN, the same
command plus --pool=<id>). The approach of thread_speedup.py's class Pool and
compile_all.sh's __pool_claim (functions_for_compile.sh): a claim is the
directory <pool>/<name>, made by os.mkdir -- atomic on the cluster's shared
file systems, NFS and Lustre included --, and its owner file names the job
(host, pid, Slurm job). Whoever makes the directory does the work; the others
see EEXIST and move on. Used by multi_launch_stress.py (its runs claimed one by
one); bench_and_test.py is the next one.

Python 3.6, standard library only: the scripts run with the cluster's system
python3."""
import errno
import glob
import os
import re
import shutil
import socket
import sys
import time

POOL_DAYS = 7          # the pools older than this, beside the one opened, are removed


class Pool:
    """The pool directory of @p pool_id: <base>/.pools/<id>, or the id itself
    (made absolute) when it holds a slash. Without an id, dir is None and
    every claim succeeds: the caller's code reads the same with or without."""

    def __init__(self, pool_id, base):
        self.id = pool_id
        host = socket.gethostname().split(".")[0]
        job = os.environ.get("SLURM_JOB_ID")
        # me: what the owner and stop files say, unique over the nodes.
        self.me = "%s.%d%s" % (host, os.getpid(), ".job" + job if job else "")
        self.tag = self._tag(pool_id, host)
        self.dir = None
        if not pool_id:
            return
        self.dir = (os.path.abspath(pool_id) if os.sep in pool_id
                    else os.path.join(os.path.abspath(base), ".pools", pool_id))

    @staticmethod
    def _tag(pool_id, host):
        """A short name of this job for its file and directory names: n<K>
        when shrink_gui ran it (run.sh exports CHUNK_BENCH_KEEP_DIR, the run
        directory <id>-n<K> of the job), else <host>.<pid> -- unique, if longer."""
        keep = os.path.basename(os.path.normpath(os.environ.get("CHUNK_BENCH_KEEP_DIR") or ""))
        if pool_id and keep:
            m = re.match(r"^%s-n(\d+)$" % re.escape(os.path.basename(os.path.normpath(pool_id))), keep)
            if m:
                return "n" + m.group(1)
        return "%s.%d" % (host, os.getpid())

    def who(self):
        """This job as the console names it: its tag, and its owner name when that says more."""
        return self.me if self.tag == self.me else "%s (%s)" % (self.tag, self.me)

    def open(self, what):
        """Create the pool directory (a pool that cannot be written to would
        read as "everything taken"), drop the pools POOL_DAYS days old beside
        it, and say so: "[pool] <id>: <what> (<dir>); this job is <who>"."""
        if not self.dir:
            return
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError as e:
            sys.exit("--pool %s: cannot create %s (%s)" % (self.id, self.dir, e))
        if not os.access(self.dir, os.W_OK):
            sys.exit("--pool %s: cannot write %s" % (self.id, self.dir))
        parent = os.path.dirname(self.dir)
        if os.path.basename(parent) == ".pools":
            old = time.time() - POOL_DAYS * 86400
            for d in glob.glob(os.path.join(parent, "*")):
                try:
                    if d != self.dir and os.path.isdir(d) and os.path.getmtime(d) < old:
                        shutil.rmtree(d, ignore_errors=True)
                except OSError:
                    pass
        print("[pool] %s: %s (%s); this job is %s" % (self.id, what, self.dir, self.who()), flush=True)

    def path(self, *names):
        return os.path.join(self.dir, *names)

    def claim(self, name):
        """True when this job made <pool>/<name> (or had made it: its owner
        file names this job), False when another one did. Silent: the caller
        says what the claim was for."""
        if not self.dir:
            return True
        c = self.path(name)
        try:
            os.mkdir(c)
        except FileExistsError:
            return self.owner(name) == self.me
        with open(os.path.join(c, "owner"), "w") as f:
            f.write(self.me + "\n")
        return True

    def owner(self, name):
        """The job that claimed @p name, "" when unknown (its owner file not
        written yet: the claimer is between its mkdir and its write)."""
        try:
            with open(self.path(name, "owner")) as f:
                return f.read().strip()
        except OSError:
            return ""

    def claim_next(self, prefix, start, end):
        """The first index i in [start, end) whose <prefix><i> this job claims
        (EEXIST: another job's, the next one is tried), or None when none is
        left. @p start is the index after the last one this job saw: the ones
        below are taken already, by it or by the others."""
        for i in range(max(0, start), end):
            if self.claim("%s%d" % (prefix, i)):
                return i
        return None

    def stop(self, why):
        """Stop the pool: <pool>/stop, naming this job and @p why. The first
        job to stop it keeps its name there (os.link of a complete file: no
        job ever reads a half-written one); later ones leave it as it is."""
        if not self.dir:
            return
        tmp = self.path(".stop.%s" % self.me)
        try:
            with open(tmp, "w") as f:
                f.write("%s: %s\n" % (self.me, why))
            try:
                os.link(tmp, self.path("stop"))
            except OSError as e:
                if e.errno != errno.EEXIST:          # a file system without hard links
                    if not os.path.exists(self.path("stop")):
                        os.replace(tmp, self.path("stop"))
                        return
            os.remove(tmp)
        except OSError:
            pass

    def stopped(self):
        """The content of <pool>/stop ("<job>: <why>") when a job stopped the
        pool, else None."""
        if not self.dir:
            return None
        p = self.path("stop")
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                return f.read().strip() or "another job"
        except OSError:
            return "another job"

    def mark(self, name, marker, text=""):
        """Write <pool>/<name>/<marker> (a done marker after a claimed piece
        of work), through a temporary file: a waiter sees it complete or not at all."""
        tmp = self.path(name, ".%s.%s" % (marker, self.me))
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, self.path(name, marker))

    def marked(self, name, marker):
        return bool(self.dir) and os.path.exists(self.path(name, marker))
