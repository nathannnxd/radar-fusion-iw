"""tests/test_alerts.py — the ego-motion alert rules of EGO_MOTION.md §5/§7 as implemented in alerts.AlertEngine.

Covers exactly the codes this branch added, which nothing else exercises: 130 IMU_DEGRADED, 131 EGO_LOST, the
121/122 TTC tiers with their closing-speed floor, the STANDING suppression of the collision layer, and 141
OBSTACLE_IN_CORRIDOR. No hardware and no pipeline: the engine is driven with hand-built `fused` / `ego` dicts and
a sink that just collects the sentences.

    python -m pytest -q tests/test_alerts.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alerts                                                          # noqa: E402


class Collector(alerts.AlertSink):
    """Keeps every sentence; codes(event) -> the codes seen since the last clear()."""

    def __init__(self):
        self.lines = []

    def send(self, line):
        self.lines.append(line)

    def codes(self, event=None):
        out = []
        for ln in self.lines:
            f = ln[1:].split("*")[0].split(",")
            if event is None or f[2] == event:
                out.append(int(f[1]))
        return out

    def clear(self):
        self.lines = []


def engine(**kw):
    kw.setdefault("resend_s", 1e9)          # heartbeats off: every test looks at the S/C transitions
    kw.setdefault("radar_only_confirm_s", 1e9)
    sink = Collector()
    return alerts.AlertEngine([sink], **kw), sink


def ego(state="MOVING", raw=2.0, gyro_ok=True, imu_seen=True, ttc_bound=False, bin_=0.272, v=None):
    """The pipeline's carrier dict (iwr1642_live.EgoEstimator.info(), EGO_MOTION.md §4.7) — only the keys the
    alert layer reads."""
    return {"state": state, "v": raw if v is None else v, "raw": raw, "valid": True,
            "moving": abs(raw) >= 0.15, "gyro_ok": gyro_ok, "imu_seen": imu_seen,
            "ttc_bound": ttc_bound, "bin": bin_}


def track(range_m=10.0, radial=-2.0, obstacle=False, state="radar-only", obj_id=7):
    """One `fused` entry as Fusion.on_radar() builds it. radial < 0 = closing."""
    return {"radar_id": obj_id, "absorbed_by": None, "state": state, "range_near_m": range_m,
            "az_from_cam": 0.0, "radial_mps": radial, "obstacle": obstacle}


class FakeCam:
    def range_from_bbox(self, bh, obj_h_m):
        return 99.0


def run(eng, t, fused, ego_dict, radar_stale=False):
    eng.evaluate(t, fused, [], FakeCam(), radar_stale, ego=ego_dict)


# ---------------------------------------------------------------- 130 IMU_DEGRADED
def test_130_only_when_an_imu_is_expected_and_has_proven_itself():
    """§7: 130 is a fault, not the default. gyro_ok is False before the first $EGOVEL sentence arrives — and the
    Pi reaches its first radar frame before the ESP32 finishes booting — so the link has to be seen once first."""
    eng, sink = engine(imu_expected=True)
    run(eng, 0.0, [], ego(gyro_ok=False, imu_seen=False))              # start-up: no sentence yet
    assert alerts.AlertCode.IMU_DEGRADED not in sink.codes()
    run(eng, 1.0, [], ego(gyro_ok=False, imu_seen=True))               # link came up, then went bad
    assert sink.codes("S") == [alerts.AlertCode.IMU_DEGRADED]
    sink.clear()
    run(eng, 2.0, [], ego(gyro_ok=True))                               # recovered -> exactly one clear
    assert sink.codes("C") == [alerts.AlertCode.IMU_DEGRADED]
    # no IMU configured at all: gyro_ok False is the normal no-link state, never an alert
    eng, sink = engine(imu_expected=False)
    run(eng, 0.0, [], ego(gyro_ok=False))
    assert alerts.AlertCode.IMU_DEGRADED not in sink.codes()


# ---------------------------------------------------------------- 131 EGO_LOST
def test_131_fires_after_ego_lost_s_of_unknown_and_clears_on_recovery():
    eng, sink = engine(ego_lost_s=2.0)
    run(eng, 0.0, [], ego(state="UNKNOWN", raw=0.0))
    run(eng, 1.9, [], ego(state="UNKNOWN", raw=0.0))
    assert alerts.AlertCode.EGO_LOST not in sink.codes()               # not yet: 2.0 s not elapsed
    run(eng, 2.1, [], ego(state="UNKNOWN", raw=0.0))
    assert sink.codes("S") == [alerts.AlertCode.EGO_LOST]
    sink.clear()
    run(eng, 2.2, [], ego(state="MOVING"))                             # a valid estimate clears it ...
    assert sink.codes("C") == [alerts.AlertCode.EGO_LOST]
    sink.clear()
    run(eng, 10.0, [], ego(state="UNKNOWN", raw=0.0))                  # ... and the timer restarts
    assert alerts.AlertCode.EGO_LOST not in sink.codes()


# ---------------------------------------------------------------- 121 / 122 TTC tiers
def test_ttc_tiers_at_their_thresholds():
    """§5: 121 at range/closing <= 3.5 s, 122 at <= 1.8 s, with the track's closing speed = -radial."""
    eng, sink = engine(ttc_warn_s=3.5, ttc_critical_s=1.8, closing_speed_mps=99.0, proximity_warn_m=0.0,
                       proximity_critical_m=0.0)
    run(eng, 0.0, [track(range_m=7.2, radial=-2.0)], ego())            # ttc 3.6 s — neither tier
    assert sink.codes("S") == []
    sink.clear()
    run(eng, 1.0, [track(range_m=6.8, radial=-2.0)], ego())            # ttc 3.4 s — warning only
    assert sink.codes("S") == [alerts.AlertCode.TTC_WARNING]
    sink.clear()
    run(eng, 2.0, [track(range_m=3.4, radial=-2.0)], ego())            # ttc 1.7 s — critical joins it
    assert sink.codes("S") == [alerts.AlertCode.TTC_CRITICAL]
    sink.clear()
    run(eng, 3.0, [track(range_m=10.0, radial=+2.0)], ego())           # receding: no TTC at all
    assert sorted(sink.codes("C")) == [alerts.AlertCode.TTC_WARNING, alerts.AlertCode.TTC_CRITICAL]


def test_ttc_ignores_closing_below_the_floor():
    eng, sink = engine()
    run(eng, 0.0, [track(range_m=0.5, radial=-0.2)], ego())            # 0.2 m/s < TTC_MIN_CLOSING_MPS
    assert alerts.AlertCode.TTC_WARNING not in sink.codes()


def test_ttc_bound_raises_the_floor_and_holds_122_back_to_121():
    """§4.3: while CREEPING the estimator sets ttc_bound — a track's radial speed is then mostly ±1 bin of
    quantisation noise, so the closing floor rises to 2·bin and 122 is not emitted."""
    creep = ego(state="CREEPING", raw=0.3, ttc_bound=True, bin_=0.272)
    quiet = dict(proximity_warn_m=0.0, proximity_critical_m=0.0, closing_speed_mps=99.0)
    eng, sink = engine(**quiet)
    run(eng, 0.0, [track(range_m=0.9, radial=-0.54)], creep)           # 0.54 m/s < 2*bin = 0.544: no TTC at all
    assert sink.codes("S") == []
    # a genuinely fast approach while creeping still warns, but never reaches the critical tier
    eng, sink = engine(**quiet)
    run(eng, 0.0, [track(range_m=1.0, radial=-2.0)], creep)            # ttc 0.5 s
    assert sink.codes("S") == [alerts.AlertCode.TTC_WARNING]
    # the same geometry at MOVING (no ttc_bound) does reach 122
    eng, sink = engine(**quiet)
    run(eng, 0.0, [track(range_m=1.0, radial=-2.0)], ego())
    assert sorted(sink.codes("S")) == [alerts.AlertCode.TTC_WARNING, alerts.AlertCode.TTC_CRITICAL]


# ---------------------------------------------------------------- STANDING suppression
def test_standing_suppresses_the_collision_layer_but_not_proximity():
    """§5: while standing, alerts are zone-occupancy only — 120/121/122/141 off, 110/111 still on."""
    eng, sink = engine(proximity_warn_m=3.0, proximity_critical_m=1.5, closing_speed_mps=2.0)
    f = track(range_m=1.2, radial=-3.0, obstacle=True)
    run(eng, 0.0, [f], ego(state="STANDING", raw=0.0, v=0.0))
    started = sink.codes("S")
    for code in (alerts.AlertCode.FAST_APPROACH, alerts.AlertCode.TTC_WARNING,
                 alerts.AlertCode.TTC_CRITICAL, alerts.AlertCode.OBSTACLE_IN_CORRIDOR):
        assert code not in started, code
    assert alerts.AlertCode.PROXIMITY_WARNING in started
    assert alerts.AlertCode.PROXIMITY_CRITICAL in started


def test_standing_with_a_creeping_raw_does_not_suppress():
    """STANDING is a one-bin deadband (0.27 m/s on hangar_v9) and forces v := 0, so the suppression is gated on
    the unclamped fit too: a tractor creeping at 0.25 m/s keeps its collision layer (§4.3/§5)."""
    eng, sink = engine(proximity_warn_m=0.0, proximity_critical_m=0.0, closing_speed_mps=2.0)
    f = track(range_m=2.0, radial=-3.0, obstacle=True)
    run(eng, 0.0, [f], ego(state="STANDING", raw=0.25, v=0.0))
    started = sink.codes("S")
    for code in (alerts.AlertCode.FAST_APPROACH, alerts.AlertCode.TTC_WARNING,
                 alerts.AlertCode.TTC_CRITICAL, alerts.AlertCode.OBSTACLE_IN_CORRIDOR):
        assert code in started, code
    assert alerts.EGO_STANDING_MAX_MPS == 0.15


# ---------------------------------------------------------------- 141 OBSTACLE_IN_CORRIDOR
def test_141_follows_the_obstacle_flag():
    """141, not 140 — 140 is RADAR_SENSOR_STALE in this repo (ALERTS.md, tools/drills.md T5)."""
    assert alerts.AlertCode.OBSTACLE_IN_CORRIDOR == 141 and alerts.AlertCode.RADAR_SENSOR_STALE == 140
    eng, sink = engine(proximity_warn_m=0.0, proximity_critical_m=0.0)
    run(eng, 0.0, [track(range_m=8.0, radial=0.0, obstacle=False)], ego())
    assert alerts.AlertCode.OBSTACLE_IN_CORRIDOR not in sink.codes()
    sink.clear()
    run(eng, 1.0, [track(range_m=8.0, radial=0.0, obstacle=True)], ego())
    assert sink.codes("S") == [alerts.AlertCode.OBSTACLE_IN_CORRIDOR]
    sink.clear()
    run(eng, 2.0, [track(range_m=8.0, radial=0.0, obstacle=True, state="out-of-frame")], ego())
    assert sink.codes("C") == [alerts.AlertCode.OBSTACLE_IN_CORRIDOR]


def test_no_ego_dict_means_no_ego_codes_and_nothing_suppressed():
    """ego=None is 'carrier state unknown': no 130/131, and the collision layer keeps working."""
    eng, sink = engine(imu_expected=True, proximity_warn_m=0.0, proximity_critical_m=0.0)
    run(eng, 0.0, [track(range_m=2.0, radial=-3.0, obstacle=True)], None)
    started = sink.codes("S")
    assert alerts.AlertCode.IMU_DEGRADED not in started and alerts.AlertCode.EGO_LOST not in started
    assert alerts.AlertCode.OBSTACLE_IN_CORRIDOR in started
    assert alerts.AlertCode.TTC_CRITICAL in started
