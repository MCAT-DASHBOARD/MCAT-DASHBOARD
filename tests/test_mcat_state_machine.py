"""
MCAT pipeline state-machine test suite (Fix M, 2026-06-11).
Pure-synthetic tests — no network, no KV. Exercises the signal lifecycle,
exit framework, re-entry gates, MQS routing, and the Fix H/I instrumentation.
Run: pytest tests/ -v
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from datetime import date
import mcat_refresh as m



# ── Fix O test infrastructure: controllable calendar date ──
import datetime as _dt

def _set_today(monkeypatch, iso):
    class _FakeDate(_dt.date):
        @classmethod
        def today(cls):
            return _dt.date.fromisoformat(iso)
    monkeypatch.setattr(m, "date", _FakeDate)

def fresh_state(**over):
    s = dict(m.DEFAULT_STATE)
    s.update(over)
    return s


# ── detect_signal ────────────────────────────────────────────────
def test_signal_requires_two_cluster_days(monkeypatch):
    s = fresh_state()
    _set_today(monkeypatch, "2026-06-01")
    s = m.detect_signal(s, 10.0, 11.0, 100.0)
    assert s["cluster_days"] == 1 and not s["signal_active"]
    _set_today(monkeypatch, "2026-06-02")
    s = m.detect_signal(s, 9.0, 10.0, 100.0)
    assert s["signal_active"] and s["entry_price"] == 100.0
    assert s["was_overheated"] is False and s["peak_dpo40"] == 10.0

def test_no_fire_when_only_one_oscillator_compressed():
    s = fresh_state()
    for _ in range(5):
        s = m.detect_signal(s, 9.0, 30.0, 100.0)  # PC40 above line
    assert not s["signal_active"] and s["cluster_days"] == 0

def test_cooldown_blocks_refire_within_30_days(monkeypatch):
    _set_today(monkeypatch, "2026-06-01")
    s = fresh_state(signal_active=False, signal_date="2026-06-01")
    s = m.detect_signal(s, 9.0, 9.0, 100.0)
    _set_today(monkeypatch, "2026-06-02")
    s = m.detect_signal(s, 9.0, 9.0, 100.0)
    assert not s["signal_active"]  # cluster=2 but cooldown blocks

def test_re_entry_gate_blocks_fire(monkeypatch):
    s = fresh_state(re_entry_state="BLOCKED")
    _set_today(monkeypatch, "2026-06-01")
    s = m.detect_signal(s, 9.0, 9.0, 100.0)
    _set_today(monkeypatch, "2026-06-02")
    s = m.detect_signal(s, 9.0, 9.0, 100.0)
    assert not s["signal_active"] and s["cluster_days"] == 2


# ── update_cycle_phase ───────────────────────────────────────────
def test_phase_transitions():
    s = fresh_state()
    s = m.update_cycle_phase(s, 14.0, 40.0);  assert s["cycle_phase"] == "EARLY_SIGNAL"
    s = m.update_cycle_phase(s, 10.0, 11.0);  assert s["cycle_phase"] == "ACCUMULATION"
    s = m.update_cycle_phase(s, 95.0, 50.0)
    assert s["cycle_phase"] == "OVERHEATED" and s["was_overheated"]
    s = m.update_cycle_phase(s, 70.0, 50.0);  assert s["cycle_phase"] == "COOLING"
    s = m.update_cycle_phase(s, 50.0, 50.0)
    assert s["cycle_phase"] == "TOPPED" and not s["was_overheated"]

def test_topped_to_waiting_closes_cycle():
    s = fresh_state(signal_active=True, cycle_phase="TOPPED")
    s = m.update_cycle_phase(s, 50.0, 50.0)
    assert s["cycle_phase"] == "WAITING" and s["signal_active"] is False


# ── update_exit_level ────────────────────────────────────────────
def test_inactive_signal_short_circuits():
    s = fresh_state(signal_active=False, exit_level="TAKE_PROFITS")
    s = m.update_exit_level(s, 50.0, 50.0, "T")
    assert s["exit_level"] is None

def test_stop_loss_at_20_percent():
    s = fresh_state(signal_active=True, entry_price=100.0)
    s = m.update_exit_level(s, 50.0, 50.0, "T", current_price=79.0)
    assert s["exit_level"] == "STOPPED_OUT" and not s["signal_active"]

def test_peak_tracks_continuously_below_oh_threshold():  # Fix F regression
    s = fresh_state(signal_active=True, peak_dpo40=60.0)
    s = m.update_exit_level(s, 50.0, 85.0, "T")
    assert s["peak_dpo40"] == 85.0  # updated though 85 < 90

def test_two_stage_exit():
    s = fresh_state(signal_active=True)
    s = m.update_exit_level(s, 95.0, 92.0, "T")
    assert s["was_overheated"] and s["exit_level"] == "OVERHEATED" and s["signal_active"]
    s = m.update_exit_level(s, 55.0, 50.0, "T")
    assert s["exit_level"] == "TAKE_PROFITS" and not s["signal_active"]

def test_no_tp_without_prior_oh():
    s = fresh_state(signal_active=True)
    s = m.update_exit_level(s, 55.0, 50.0, "T")
    assert s["exit_level"] is None and s["signal_active"]


# ── update_re_entry_state ────────────────────────────────────────
def test_tp_exit_blocks_re_entry():
    s = fresh_state(exit_level="TAKE_PROFITS", re_entry_state="ELIGIBLE")
    s = m.update_re_entry_state(s, 50.0, 50.0)
    assert s["re_entry_state"] == "RESET_PENDING"  # BLOCKED then both>=12 same call

def test_stale_timeout_at_120_days():
    s = fresh_state(signal_active=True, signal_days_ago=121)
    s = m.update_re_entry_state(s, 50.0, 50.0)
    assert s["exit_level"] == "STALE" and not s["signal_active"]

def test_blocked_reset_eligible_cycle():
    s = fresh_state(re_entry_state="BLOCKED")
    s = m.update_re_entry_state(s, 8.0, 9.0)
    assert s["re_entry_state"] == "BLOCKED"      # still compressed
    s = m.update_re_entry_state(s, 20.0, 20.0)
    assert s["re_entry_state"] == "RESET_PENDING"
    s = m.update_re_entry_state(s, 8.0, 9.0)
    assert s["re_entry_state"] == "ELIGIBLE"


# ── Fix H/I instrumentation ──────────────────────────────────────
def test_h_clean_nan_safety():
    assert m._h_clean(float("nan")) is None
    assert m._h_clean(None) is None
    assert m._h_clean(5.234) == 5.2
    assert m._h_clean("x") is None

def test_event_log_cap():
    s = fresh_state(event_log=[{"type": "X"}] * 35)
    s = m.log_signal_events(s, {"signal_active": False, "was_overheated": False}, 50.0, 50.0)
    assert len(s["event_log"]) == 30

def test_near_miss_hysteresis_and_fired():
    pre = {"signal_active": False, "was_overheated": False, "signal_date": None}
    s = m.log_signal_events(fresh_state(), pre, 14.0, 30.0, 50.0)
    assert s["nm_episode"] is not None
    s = m.log_signal_events(s, pre, 16.0, 30.0, 51.0)
    assert s["nm_episode"] is not None            # 16 < 17 stays open
    s = m.log_signal_events(s, pre, 18.0, 30.0, 52.0)
    nm = [e for e in s["event_log"] if e["type"] == "NEAR_MISS_END"]
    assert len(nm) == 1 and nm[0]["fired"] is False and nm[0]["min_dpo20"] == 14.0

def test_oh_episode_without_signal():       # XLM case
    pre = {"signal_active": False, "was_overheated": True, "signal_date": None}
    s = m.log_signal_events(fresh_state(was_overheated=True), pre, 93.7, 93.2, 0.19)
    assert s["oh_episode"] is not None and s.get("nm_episode") is None
    s = m.log_signal_events(s, pre, 55.0, 50.0, 0.15)
    ends = [e for e in s["event_log"] if e["type"] == "OH_EPISODE_END"]
    assert len(ends) == 1 and s["oh_episode"] is None


# ── MQS routing + confidence ─────────────────────────────────────
def test_mqs_rule_routing():
    r = lambda *a: m.classify_mqs(*a)["rule_number"]
    assert r(46.0, False, False, False, False, False) == 1     # capitulation
    assert r(25.0, False, True,  False, False, False) == 2     # co-bottom first
    assert r(25.0, False, False, False, False, False) == 3
    assert r(20.0, False, False, False, False, False) == 3     # boundary: 20 is RED R3
    assert r(25.0, True,  False, False, False, False) == 4
    assert r(35.0, True,  False, False, False, False) == 5
    assert r(18.0, False, False, False, False, False) == 6
    assert r(18.0, True,  False, False, True,  False) == 9     # GREEN + M2 premium
    assert r(18.0, True,  False, True,  False, False) == 8     # yen unwind override

def test_confidence_scoring():
    c = m.compute_confidence(True, True, True, True)
    assert c["score"] == 4
    c = m.compute_confidence(True, True, True, "pending")
    assert c["score"] == 3


# ── Full lifecycle through the orchestrator ──────────────────────
def test_full_signal_lifecycle(monkeypatch):
    # Regression guard for Fix N (2026-06-11): before N, update_cycle_phase ran first
    # and cleared was_overheated on the cool-down day, making TAKE_PROFITS unreachable
    # and leaving the C5 re-entry gate disengaged. This test fails if that ever returns.
    _days = ["2026-06-0%d" % i for i in range(1, 10)]
    _i = [0]
    def day(st, d20, d40, px):
        _set_today(monkeypatch, _days[_i[0]]); _i[0] += 1
        return m.process_asset_automation(
            "TST", 50.0, d20, d40, 50.0, d20, d40, "flat", "flat",
            current_price=px, prev_state=st)
    s = day(None, 10.0, 11.0, 100.0)
    assert not s["signal_active"] and s["cluster_days"] == 1
    s = day(s, 9.0, 10.0, 100.0)
    assert s["signal_active"]                                  # fires on day 2
    assert any(e["type"] == "ENTRY" for e in s["event_log"])
    s = day(s, 40.0, 35.0, 130.0)                              # rising
    s = day(s, 95.0, 92.0, 200.0)                              # overheats
    assert s["was_overheated"] and s["oh_episode"] is not None
    assert any(e["type"] == "OH_TRANSITION" for e in s["event_log"])
    s = day(s, 50.0, 55.0, 180.0)                              # TP exit
    assert not s["signal_active"] and s["re_entry_state"] in ("BLOCKED", "RESET_PENDING")
    exits = [e for e in s["event_log"] if e["type"] == "EXIT"]
    assert len(exits) == 1 and exits[0]["exit_kind"] == "TAKE_PROFITS"
    assert exits[0]["pct_vs_entry"] == 80.0 and exits[0]["peak_dpo40"] == 92.0
    assert any(e["type"] == "OH_EPISODE_END" for e in s["event_log"])
    s = day(s, 9.0, 9.0, 90.0)                                 # compressed again
    assert not s["signal_active"]                              # gated, no cascade re-entry


# ── Fix K descriptors (pure functions only — fetches are graceful-fail) ──
def test_pi_cycle_math():
    closes = [100.0] * 400
    r = m.compute_pi_cycle(closes)
    assert r["ma111"] == 100 and r["ma350x2"] == 200
    assert r["gap_pct"] == 50.0 and r["crossed"] is False
    r2 = m.compute_pi_cycle([100.0] * 100)
    assert r2["gap_pct"] is None and r2["note"] == "insufficient history"

def test_descriptor_bands():
    assert m.classify_mvrv_z(-0.5) == "historic bottom zone"
    assert m.classify_mvrv_z(8.0) == "historic top zone"
    assert m.classify_nupl(-0.1) == "capitulation zone"
    assert m.classify_nupl(0.9) == "euphoria (top zone)"
    assert m.classify_puell(0.3) == "miner capitulation zone"
    assert m.classify_puell(None) == "unavailable"

def test_bg_parser_defensive():
    assert m._bg_latest_value([{"d": "2026-06-10", "nupl": 0.41}], "nupl") == 0.41
    assert m._bg_latest_value({"mvrvZscore": 1.2}, "mvrvZscore") == 1.2
    assert m._bg_latest_value([{"d": "x", "weird": "0.7"}], "missing") == 0.7
    assert m._bg_latest_value([], "v") is None
    assert m._bg_latest_value("garbage", "v") is None

def test_onchain_graceful_without_key(monkeypatch):
    monkeypatch.delenv("BGEOMETRICS_API_KEY", raising=False)
    out = m.fetch_onchain_descriptors()
    assert out["mvrv_z"] is None and out["error"] == "BGEOMETRICS_API_KEY not configured"
    assert out["mvrv_z_band"] == "unavailable"


# ── Fix O: date-aware cluster (12h cadence safety) ──
def test_cluster_one_increment_per_calendar_day(monkeypatch):
    s = fresh_state()
    _set_today(monkeypatch, "2026-06-01")
    s = m.detect_signal(s, 9.0, 9.0, 100.0)
    s = m.detect_signal(s, 9.0, 9.0, 100.0)   # second run, SAME day
    assert s["cluster_days"] == 1 and not s["signal_active"]
    _set_today(monkeypatch, "2026-06-02")
    s = m.detect_signal(s, 9.0, 9.0, 100.0)   # next day -> 2 -> fires
    assert s["cluster_days"] == 2 and s["signal_active"]

def test_cluster_resets_on_intraday_bounce(monkeypatch):
    s = fresh_state()
    _set_today(monkeypatch, "2026-06-01")
    s = m.detect_signal(s, 9.0, 9.0, 100.0)
    assert s["cluster_days"] == 1
    s = m.detect_signal(s, 30.0, 30.0, 100.0)  # same-day bounce above threshold
    assert s["cluster_days"] == 0 and s["cluster_last_date"] is None
    _set_today(monkeypatch, "2026-06-02")
    s = m.detect_signal(s, 9.0, 9.0, 100.0)
    assert s["cluster_days"] == 1 and not s["signal_active"]  # strict: restart
