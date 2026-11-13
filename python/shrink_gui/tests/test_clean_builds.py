# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
""""Nettoyer les builds…": what remote_helper's clean_builds lists and removes,
and how run.sh's build check ("Builds à jour") judges a cleaned build."""
import base64
import json
import os
import shutil
import subprocess
import time

import pytest

from shrink_gui import runscripts


def _blob(paths):
    return base64.b64encode(json.dumps([str(p) for p in paths]).encode()).decode()


def _fake_build(root, name="omp", arch="RHEL_9.6__x86_64"):
    """A configured build directory with a bit of everything."""
    d = root / f"cmake-build-{name}" / arch
    (d / "chunk" / "CMakeFiles" / "chunk_zero_256.dir" / "Unity").mkdir(parents=True)
    (d / "examples" / "bench" / "CMakeFiles" / "bench.dir").mkdir(parents=True)
    (d / "build.ninja").write_text("# not a real one\n")
    (d / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/nowhere\n")
    (d / "chunk" / "CMakeFiles" / "chunk_zero_256.dir" / "Unity" / "unity_0_cxx.cxx.o").write_bytes(b"o" * 3000)
    (d / "chunk" / "CMakeFiles" / "chunk_zero_256.dir" / "Unity" / "unity_0_cxx.cxx.o.d").write_bytes(b"d" * 100)
    (d / "chunk" / "CMakeFiles" / "chunk_zero_256.dir" / "Unity" / "unity_0_cxx.cxx").write_text("// generated\n")
    (d / "chunk" / "libchunk_zero_256.a").write_bytes(b"a" * 2000)
    (d / "chunk" / "libchunk_dyn.so").write_bytes(b"s" * 500)
    (d / "examples" / "bench" / "CMakeFiles" / "bench.dir" / "bench.cpp.o").write_bytes(b"o" * 1000)
    exe = d / "examples" / "bench" / "bench"
    exe.write_bytes(b"x" * 4000)
    exe.chmod(0o755)
    return d


def test_scan_then_clean(site):
    br = site.home / "builds"
    d = _fake_build(br)
    r = site.helper("clean_builds", str(br), "scan")
    assert r["build_rep"] == str(br)
    (row,) = r["dirs"]
    assert (row["name"], row["arch"], row["path"]) == ("omp", "RHEL_9.6__x86_64", str(d))
    assert row["removable"] == 3000 + 100 + 2000 + 1000 and row["files"] == 4
    assert row["total"] > row["removable"] and row["cleaned"] is None

    c = site.helper("clean_builds", str(br), "clean", _blob([d]))
    (done,) = c["cleaned"]
    assert done["freed"] == 6100 and done["files"] == 4
    assert done["up_to_date"] is True              # what (the fake) ninja -n said before
    assert not list(d.rglob("*.o")) and not list(d.rglob("*.a")) and not list(d.rglob("*.o.d"))
    # what stays: the executable, the shared library, the build files, the generated sources
    for kept in ("examples/bench/bench", "chunk/libchunk_dyn.so", "build.ninja", "CMakeCache.txt",
                 "chunk/CMakeFiles/chunk_zero_256.dir/Unity/unity_0_cxx.cxx"):
        assert (d / kept).exists(), kept
    mark = json.loads((d / ".shrink_gui_cleaned").read_text())
    assert mark["up_to_date"] is True and mark["time"] > 0

    again = site.helper("clean_builds", str(br), "scan")["dirs"][0]
    assert again["removable"] == 0 and again["cleaned"] == pytest.approx(mark["time"])


def test_stale_or_unchecked_before_the_clean(site):
    br = site.home / "builds"
    stale = _fake_build(br, "stale")
    (stale / "STALE").write_text("")                 # the fake ninja: work to do
    bare = _fake_build(br, "bare")
    shutil.rmtree(bare / "examples")                 # nothing built: nothing to check
    c = site.helper("clean_builds", str(br), "clean", _blob([stale, bare]))
    assert [x["up_to_date"] for x in c["cleaned"]] == [False, None]
    # cleaned again, no build since: ninja -n sees the missing objects, the first verdict stays
    (stale / "STALE").unlink()
    c = site.helper("clean_builds", str(br), "clean", _blob([stale]))
    assert c["cleaned"][0]["up_to_date"] is True   # the fake ninja knows no missing object
    # stale per ninja (its objects are gone), cleaned up to date before: judged by the
    # sources, as run.sh's check does -- here a repository that does not exist: not up to date
    (stale / "STALE").write_text("")
    c = site.helper("clean_builds", str(br), "clean", _blob([stale]))
    assert c["cleaned"][0]["up_to_date"] is False


def test_clean_refuses_a_foreign_path(site):
    br = site.home / "builds"
    _fake_build(br)
    other = site.home / "elsewhere"
    other.mkdir()
    (other / "precious.o").write_text("x")
    r = site.helper("clean_builds", str(br), "clean", _blob([other]))
    assert "pas un répertoire de build" in r["error"]
    assert (other / "precious.o").exists()


def test_mark_ends_with_the_next_build(site):
    br = site.home / "builds"
    d = _fake_build(br)
    site.helper("clean_builds", str(br), "clean", _blob([d]))
    assert site.helper("clean_builds", str(br), "scan")["dirs"][0]["cleaned"]
    later = time.time() + 5
    (d / ".ninja_log").write_text("# ninja log v6\n")
    os.utime(d / ".ninja_log", (later, later))
    assert site.helper("clean_builds", str(br), "scan")["dirs"][0]["cleaned"] is None


# --- a real CMake + ninja build, and run.sh's check on it --------------------

def _mine():
    """The architecture directory name run.sh's check looks for first."""
    return subprocess.run(["bash", "-c", 'source /etc/os-release 2>/dev/null && echo "${ID^^}_${VERSION_ID}__$(uname -m)"'],
                          capture_output=True, text=True).stdout.strip() or "X__" + os.uname().machine


def _real_path(site):
    """PATH without the fakes (tests/fakes has a fake ninja)."""
    return os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep) if p != str(site.bin))


def _real(site):
    """The real ninja for cmake, run.sh's check and the helper."""
    site.env = dict(site.env, PATH=_real_path(site))
    return dict(os.environ, PATH=site.env["PATH"])


def _check(site, root, name):
    """gui_check_builds refuse <root> <name>, real ninja: (exit code, output)."""
    script = runscripts.CHECK_BUILDS_SH + f'\ngui_check_builds refuse "{root}" {name}\n'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=dict(os.environ, PATH=_real_path(site)))
    return r.returncode, r.stdout + r.stderr


def _cmake(env, *args):
    subprocess.run(["cmake", *args], check=True, capture_output=True, env=env)


@pytest.mark.skipif(not (shutil.which("cmake") and shutil.which("ninja") and shutil.which("c++")),
                    reason="needs cmake, ninja and a C++ compiler")
def test_real_build_check_after_clean(site):
    src = site.home / "proj"
    (src / "examples").mkdir(parents=True)
    (src / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\nproject(p CXX)\n"
                                        "add_library(l STATIC l.cpp)\nadd_subdirectory(examples)\n")
    (src / "l.cpp").write_text("int f() { return 1; }\n")
    (src / "examples" / "CMakeLists.txt").write_text("add_executable(e e.cpp)\ntarget_link_libraries(e l)\n")
    (src / "examples" / "e.cpp").write_text("int f();\nint main() { return f() - 1; }\n")
    br = site.home / "builds"
    d = br / "cmake-build-t" / _mine()
    env = _real(site)
    _cmake(env, "-S", str(src), "-B", str(d), "-G", "Ninja")
    _cmake(env, "--build", str(d))
    assert _check(site, br, "t") == (0, "[gui] build check: 1 build(s) up to date\n")

    time.sleep(1.1)                                 # mtimes at a second's resolution on some file systems
    c = site.helper("clean_builds", str(br), "clean", _blob([d]))["cleaned"][0]
    assert c["up_to_date"] is True and c["files"] >= 3   # l.cpp.o, e.cpp.o, libl.a
    assert (d / "examples" / "e").exists() and not list(d.rglob("*.o"))
    # cleaned, nothing changed: up to date, although ninja -n would rebuild everything
    code, out = _check(site, br, "t")
    assert code == 0 and "1 build(s) up to date" in out, out

    # cleaned again, nothing changed: still up to date (ninja -n alone would say no)
    assert site.helper("clean_builds", str(br), "clean", _blob([d]))["cleaned"][0]["up_to_date"] is True

    time.sleep(1.1)
    (src / "l.cpp").write_text("int f() { return 2; }\n")
    code, out = _check(site, br, "t")
    assert code == 96 and "cleaned, l.cpp changed since" in out, out

    # the next build ends the mark's reign: ninja -n again
    _cmake(env, "--build", str(d))
    code, out = _check(site, br, "t")
    assert code == 0 and "1 build(s) up to date" in out, out
    assert site.helper("clean_builds", str(br), "scan")["dirs"][0]["cleaned"] is None


@pytest.mark.skipif(not (shutil.which("cmake") and shutil.which("ninja") and shutil.which("c++")),
                    reason="needs cmake, ninja and a C++ compiler")
def test_real_build_cleaned_while_stale(site):
    src = site.home / "proj"
    (src / "examples").mkdir(parents=True)
    (src / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\nproject(p CXX)\n"
                                        "add_subdirectory(examples)\n")
    (src / "examples" / "CMakeLists.txt").write_text("add_executable(e e.cpp)\n")
    (src / "examples" / "e.cpp").write_text("int main() { return 0; }\n")
    br = site.home / "builds"
    d = br / "cmake-build-t" / _mine()
    env = _real(site)
    _cmake(env, "-S", str(src), "-B", str(d), "-G", "Ninja")
    _cmake(env, "--build", str(d))
    time.sleep(1.1)
    (src / "examples" / "e.cpp").write_text("int main() { return 1 - 1; }\n")
    c = site.helper("clean_builds", str(br), "clean", _blob([d]))["cleaned"][0]
    assert c["up_to_date"] is False
    code, out = _check(site, br, "t")
    assert code == 96 and "cleaned while not up to date" in out, out


# --- the dialog -------------------------------------------------------------

def test_dialog_totals(qapp):
    from PySide6.QtCore import Qt

    from shrink_gui.builds_panel import CleanBuildsDialog, fmt_bytes
    scan = {"build_rep": "/b", "dirs": [
        {"name": "omp", "arch": "A__x86_64", "path": "/b/cmake-build-omp/A__x86_64", "removable": 3 << 30,
         "total": 5 << 30, "files": 10, "cleaned": None},
        {"name": "dbg", "arch": "A__x86_64", "path": "/b/cmake-build-dbg/A__x86_64", "removable": 0,
         "total": 1 << 20, "files": 0, "cleaned": 1.7e9}]}
    dlg = CleanBuildsDialog(None, scan)
    assert dlg.selected() == [("/b/cmake-build-omp/A__x86_64", 3 << 30)]     # nothing to free: unchecked
    assert "3.0 Gio à libérer" in dlg.total.text() and dlg.ok.isEnabled()
    dlg.table.item(0, 0).setCheckState(Qt.Unchecked)
    assert dlg.selected() == [] and not dlg.ok.isEnabled()
    assert fmt_bytes(512) == "512 o" and fmt_bytes(1536) == "1.5 Kio"


def test_button_cleans(gui, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from shrink_gui import builds_panel
    w, pump = gui
    site_home = os.environ["FAKE_HOME"]
    br = os.path.join(site_home, "builds")
    import pathlib
    d = _fake_build(pathlib.Path(br))
    monkeypatch.setattr(builds_panel, "BUILD_REP", br)
    monkeypatch.setattr(builds_panel.CleanBuildsDialog, "exec", lambda self: QDialog.Accepted)
    w.bclean.click()
    assert pump(lambda: "nettoyé" in w.statusBar().currentMessage()), w.statusBar().currentMessage()
    assert "1 build(s) nettoyé(s), 6.0 Kio libérés" in w.statusBar().currentMessage()
    assert not list(d.rglob("*.o")) and (d / "examples" / "bench" / "bench").exists()
