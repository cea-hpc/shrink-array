#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""
catch_crash.py — Lance shrink_bench_chunk2 (--mode perf) en boucle sous gdb (batch) et
capture automatiquement la stack + le contexte au PREMIER crash (SIGSEGV/SIGABRT)
ou déclenchement de sonde. Conçu pour la traque du use-after-munmap cross-thread
quand le bug devient rare (cf. DEBUG_SESSION_RESUME_3.md).

Usage :
    python3 catch_crash.py [-n NRUNS] [-t THREADS] [--args "ARGS"] [--continue]

    -n         nombre max de runs (défaut 1000)
    -t         OMP_NUM_THREADS fixe (défaut 64) ; "rand" = aléatoire 2..128
    --args     arguments passés à l'exécutable (défaut "--divisor 10")
    --exe      chemin de l'exécutable (défaut = build ledger aarch64)
    --continue ne PAS s'arrêter au 1er crash : archiver et continuer
    --outdir   répertoire d'archivage des crashes (défaut ./crashes)

Sortie : pour chaque crash, un fichier crashes/crash_<i>_<pid>.log contenant
la sortie programme + le backtrace de tous les threads + le contexte allocateur.
"""

import argparse
import datetime
import os
import random
import subprocess
import sys
import time

DEFAULT_EXE = ("$CCCWORKDIR/shrink-array/"
               "cmake-build-omp-llvm-debug/RHEL_8.10__x86_64/examples/shrink_bench/"
               "shrink_bench_chunk2a_dyn")  # --mode perf a passer a l'executable

# Commandes gdb exécutées en batch. On lance le programme ; s'il s'arrête sur un
# signal, on dump. S'il finit normalement, gdb continue après `run` mais $_exitcode
# est défini et il n'y a PAS de stack — on teste donc avant de faire `bt`.
# `set pagination off` est indispensable en batch (sinon gdb bloque sur --More--).
GDB_SCRIPT = r"""
set pagination off
set confirm off
set print pretty on
# Ne s'arrêter que sur les signaux fatals, laisser passer le reste.
handle SIGSEGV stop print
handle SIGABRT stop print
handle SIGFPE stop print
run
# --- On arrive ici soit après un signal (programme arrêté), soit après une fin
# --- normale ($_exitcode défini). On ne dumpe QUE dans le premier cas.
if $_isvoid($_exitcode)
  printf "\n=========== CRASH CONTEXT (signal) ===========\n"
  printf "\n--- faulting thread backtrace ---\n"
  bt
  printf "\n--- bench frame: k / valeurs / tab_complex vs tab1 ---\n"
  frame 11
  print k
  print v_real
  print v_im
  print val_sum
  printf "\n--- faulting address (si SIGSEGV) ---\n"
  print $_siginfo._sifields._sigfault.si_addr
  printf "\n--- all threads backtrace ---\n"
  thread apply all bt
  printf "\n--- registers ---\n"
  info registers x0 x1 x2 x19 x20
  printf "\n=========== END CONTEXT ===========\n"
else
  printf "\n___PROGRAM_EXITED_NORMALLY___ exitcode=%d\n", $_exitcode
end
quit
"""

# Variante interactive : gdb boucle LUI-MÊME sur `run` jusqu'à ce que le
# programme s'arrête sur un signal (SIGSEGV/SIGABRT). À ce moment il imprime le
# contexte PUIS rend le prompt (pas de `quit`) : tu gardes la main pour inspecter
# (p, x, frame, info locals, thread apply all bt...). Si le programme finit
# normalement, la boucle `run` recommence. C'est le mode à utiliser pour un crash
# rare qu'on veut fouiller à la main : lance-le et laisse tourner, il s'arrêtera
# tout seul dans gdb au crash. Sortir avec `quit`.
GDB_SCRIPT_INTERACTIVE = r"""
set pagination off
set confirm off
set print pretty on
handle SIGSEGV stop print
handle SIGABRT stop print
handle SIGFPE stop print
# Boucle: relance tant que le programme sort normalement ($_exitcode défini).
# On s'arrête (et on rend le prompt) dès qu'un signal laisse $_exitcode void.
python
import gdb
while True:
    gdb.execute("run")
    try:
        ec = gdb.parse_and_eval("$_exitcode")
        # exit normal -> relancer
        print("[catch_crash] exit normal, relance...")
        continue
    except gdb.error:
        # $_exitcode indéfini => arrêt sur signal
        print("\n=========== CRASH CONTEXT (signal) ===========")
        gdb.execute("bt")
        for _c in ("frame 11","print k","print v_real","print v_im","print val_sum"):
            try: gdb.execute(_c)
            except gdb.error: pass
        try:
            gdb.execute("print $_siginfo._sifields._sigfault.si_addr")
        except gdb.error:
            pass
        gdb.execute("thread apply all bt")
        print("=========== END CONTEXT — prompt gdb à toi (quit pour sortir) ===========")
        break
end
"""


def build_gdb_command(exe, args, gdb_script_path, interactive=False):
    # -batch : exécute le script et quitte (non-surveillé, archive auto).
    # -iex 'set auto-load safe-path /' : évite le warning auto-load de libstdc++.
    #
    # interactive=True : au lieu de -batch (qui quitte au crash), on charge le
    # même script SANS -batch et SANS le `quit` final. gdb exécute `run`,
    # s'arrête sur le signal, imprime le contexte, PUIS rend le prompt : tu gardes
    # la main pour inspecter (p, x, frame, info locals...). Utile quand un crash
    # rare tombe enfin et qu'on veut fouiller à la main.
    if interactive:
        cmd = ["gdb", "-nx",
               "-iex", "set auto-load safe-path /",
               "-x", gdb_script_path, "--args", exe]
    else:
        cmd = ["gdb", "-batch", "-nx",
               "-iex", "set auto-load safe-path /",
               "-x", gdb_script_path, "--args", exe]
    cmd.extend(args)
    return cmd


def looks_like_crash(returncode, output):
    # gdb -batch renvoie 0 même après un crash du programme inférieur, donc on
    # détecte le crash dans la SORTIE. IMPORTANT : ne PAS matcher la bannière
    # "CRASH CONTEXT" seule — en batch elle peut apparaître ; on exige un vrai
    # signal ou une sonde. Et si le programme est sorti normalement, on ignore.
    if "___PROGRAM_EXITED_NORMALLY___" in output:
        return False
    markers = [
        "Program received signal SIGSEGV",
        "Program received signal SIGABRT",
        "Program received signal SIGFPE",
        "Arithmetic exception",
        "Program terminated with signal",
        "CRASH CONTEXT (signal)",     # n'est imprimée que dans la branche signal
        "ALLOC_PROBE",
        "FLUSH_PROBE",
        "POOL_LEDGER",
        "INVARIANT",
        "ThreadSanitizer",
    ]
    return any(m in output for m in markers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1000, help="nombre max de runs")
    ap.add_argument("-t", default="64", help="OMP_NUM_THREADS (entier ou 'rand')")
    ap.add_argument("--args", default="--divisor 10", help="args de l'exécutable")
    ap.add_argument("--exe", default=DEFAULT_EXE, help="chemin de l'exécutable")
    ap.add_argument("--continue", dest="keep_going", action="store_true",
                    help="continuer après un crash au lieu de s'arrêter")
    ap.add_argument("--interactive", action="store_true",
                    help="au crash, garder la main dans gdb (prompt interactif) "
                         "au lieu d'archiver et quitter")
    ap.add_argument("--outdir", default="./crashes", help="dossier d'archivage")
    args = ap.parse_args()

    if not (os.path.isfile(args.exe) and os.access(args.exe, os.X_OK)):
        print(f"Exécutable introuvable ou non exécutable : {args.exe}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # écrire le script gdb dans un fichier temporaire réutilisé à chaque run
    gdb_script_path = os.path.join(args.outdir, "_gdb_catch.txt")
    with open(gdb_script_path, "w") as f:
        f.write(GDB_SCRIPT_INTERACTIVE if args.interactive else GDB_SCRIPT)

    prog_args = args.args.split() if args.args else []
    start = time.time()
    crashes = 0

    for i in range(1, args.n + 1):
        threads = random.randint(2, 128) if args.t == "rand" else int(args.t)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))

        cmd = build_gdb_command(args.exe, prog_args, gdb_script_path,
                                interactive=args.interactive)

        if args.interactive:
            # Mode interactif : le script gdb boucle LUI-MÊME sur `run` jusqu'au
            # crash, puis rend le prompt. On lance gdb une seule fois en
            # avant-plan (il hérite du terminal) et on laisse la main à l'utilisateur.
            print(f"Mode interactif : gdb va relancer en boucle "
                  f"(threads={threads}) jusqu'au crash, puis rendre le prompt.\n"
                  f"Laisse tourner ; au crash tu pourras inspecter, puis `quit`.\n",
                  flush=True)
            subprocess.run(cmd, env=env)
            print("\ngdb terminé.")
            return

        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""

        if looks_like_crash(proc.returncode, out):
            crashes += 1
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(args.outdir, f"crash_{i:05d}_{threads}thr_{stamp}.log")
            with open(path, "w") as f:
                f.write(f"# run {i}/{args.n}  threads={threads}  "
                        f"rc={proc.returncode}  {stamp}\n")
                f.write(f"# cmd: {' '.join(cmd)}\n\n")
                f.write(out)
            print(f"[{i}/{args.n}] CRASH (threads={threads}) -> {path}")
            if not args.keep_going:
                print(f"\nArrêt au premier crash. {crashes} crash(es). "
                      f"Voir {path}")
                break
        else:
            elapsed = time.time() - start
            rate = elapsed / i
            eta = (args.n - i) * rate
            # Infos utiles extraites de la sortie pour savoir où en est le run.
            n_recycle = out.count("[RECYCLE]")
            exited_ok = "___PROGRAM_EXITED_NORMALLY___" in out
            # dernier "k :" atteint (progression de la boucle du bench)
            last_k = ""
            for line in out.splitlines():
                if line.startswith("k :") or " val_sum " in line:
                    last_k = line.strip()[:60]
            status = "exit0" if exited_ok else "no-exit?"
            extra = f" last[{last_k}]" if last_k else ""
            print(f"[{i}/{args.n}] OK {status}  threads={threads}  "
                  f"recycle={n_recycle}  elapsed={elapsed:.0f}s  eta={eta:.0f}s{extra}",
                  flush=True)

    total = time.time() - start
    print(f"\nTerminé : {i} runs, {crashes} crash(es), {total:.0f}s.")
    if crashes == 0:
        print("Aucun crash capturé — augmenter -n, retirer --divisor "
              "(runs plus gros), ou -t rand pour varier la charge.")


if __name__ == "__main__":
    main()
