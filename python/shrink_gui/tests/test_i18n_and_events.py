"""Translations, and bench_runner/events.py (the other side of events.jsonl)."""
import json

import pytest

from shrink_gui import i18n


@pytest.fixture
def english():
    i18n.set_lang("en")
    yield
    i18n.set_lang("fr")


def test_french_is_identity():
    i18n.set_lang("fr")
    assert i18n.T("en cours") == "en cours"


@pytest.mark.parametrize("fr, en", [
    ("en cours", "running"),
    ("en cours ? (lancé depuis node0000)", "running? (started from node0000)"),
    ("terminé sans code (limite de temps ? voir le log)", "ended without a code (time limit? see the log)"),
    ("bench_and_test.py : 52 options lues (python 3.6.8 sur inti)",
     "bench_and_test.py: 52 options read (python 3.6.8 on inti)"),
    ("4/12 (33 %) · reste ~1 h 20", "4/12 (33 %) · ~1 h 20 left"),
    ("Arrêter le run r1 ?\nle job 12 (ccc_mdel)", "Stop run r1?\njob 12 (ccc_mdel)"),
    ("pas encore de timings_summary.csv", "no timings_summary.csv yet"),
    (" — commits 1a2b3c4d → 5e6f7a8b", " — commits 1a2b3c4d → 5e6f7a8b"),
    ("3 phase(s) — fin : 2 erreur(s), 12.5 min", "3 phase(s) — end: 2 error(s), 12.5 min"),
])
def test_english(english, fr, en):
    assert i18n.T(fr) == en


def test_status_text(english):
    assert i18n.status_text({"status": "fini", "code": 3}) == "done (code 3)"


# ------------------------------------------------------------ bench_runner/events
events = pytest.importorskip("bench_runner.events", reason="bench_runner without events.py "
                             "(run add_bench_events.py)")


def _records(p):
    return [json.loads(l) for l in open(p)]


def test_events_off_without_file(tmp_path, monkeypatch):
    monkeypatch.delenv("CHUNK_BENCH_EVENTS", raising=False)
    events.open_file(None)
    events.emit("start")
    assert not events.enabled()


def test_events_phase_and_progress(tmp_path, monkeypatch):
    p = tmp_path / "ev.jsonl"
    monkeypatch.setenv("CHUNK_BENCH_EVENTS", str(p))
    events.open_file(None)
    errors = []
    with events.phase("check", errors):
        events.progress(1, 3, min_interval=0)
        errors.append("boom")
    with pytest.raises(RuntimeError):
        with events.phase("perf", errors):
            raise RuntimeError
    recs = _records(p)
    kinds = [r["kind"] for r in recs]
    assert kinds == ["phase_start", "progress", "phase_end", "phase_start", "phase_end"]
    assert recs[1]["phase"] == "check" and recs[1]["done"] == 1
    assert recs[2]["status"] == "errors" and recs[2]["errors"] == 1
    assert recs[4]["status"] == "exception"
    events.open_file("")          # off again
    monkeypatch.delenv("CHUNK_BENCH_EVENTS")
    events.open_file(None)


def test_run_counter_reports(tmp_path):
    from bench_runner.progress import RunCounter
    p = tmp_path / "ev.jsonl"
    events.open_file(str(p))
    c = RunCounter(2)
    c.idx += 1
    c.idx += 1                    # the last one always goes through the throttle
    assert c.idx == 2
    done = [r["done"] for r in _records(p) if r["kind"] == "progress"]
    assert done[-1] == 2
    events.open_file("")
