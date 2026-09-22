"""test_ego.py — checks for the radar-only ego-motion estimate (iwr1642_live.EgoEstimator) and the yaw compensation.

    python test_ego.py <radar.bin recorded with the radar standing still> [--cfg radar_configs/profile_sdk3.cfg]

A. standing:   on a recording of a stationary radar (people may walk in it) the estimate must stay ~0.
B. synthetic:  the same frames with carrier motion injected into every point's Doppler (v' = v - v_ego*cos(az), wrapped
               by the profile's Doppler period exactly as the sensor would) — the estimate must follow the true profile,
               including speeds beyond the profile's unambiguous v_max (Doppler wrap-around).
C. yaw:        a stationary target seen from a turning carrier stays on its track only if the tracker rotates the
               world by -yaw_rate*dt (Track.rotate) — checked on a simulated 30-frame turn.
Exit code 0 = all passed.
"""
import argparse
import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iwr1642_live as radar


def load_frames(path):
    data = open(path, "rb").read()
    return [f for f in radar.frames_from_bytes(data) if f]


def inject_motion(frame, v_ego, period):
    """Carrier moving forward at v_ego: every target's Doppler shifts by -v_ego*cos(az), then wraps like the sensor."""
    fr = copy.copy(frame)
    pts = []
    for p in frame["points"]:
        x, y = p[0], p[1]
        az = math.atan2(x, y)
        v = p[3] - v_ego * math.cos(az)
        if period:
            v -= round(v / period) * period
        pts.append((p[0], p[1], p[2], v) + tuple(p[4:]))
    fr["points"] = pts
    return fr


def run(frames, cfg, v_profile=None):
    radar.Track._next_id = 1
    pipe = radar.Pipeline(cfg, None, 0.0, use_background=False, ego_mode="radar")
    period = cfg.get("doppler_period_mps")
    est, truth = [], []
    for i, fr in enumerate(frames):
        v_true = v_profile(i) if v_profile else 0.0
        out = pipe.process(inject_motion(fr, v_true, period) if v_profile else fr)
        est.append(out["ego"]["v"]); truth.append(v_true)
    return np.array(est), np.array(truth)


def test_standing(frames, cfg):
    est, _ = run(frames, cfg)
    frac_ok = float(np.mean(np.abs(est) < radar.EGO_STATIONARY_MPS))
    print(f"A standing: {len(est)} frames, |v|<{radar.EGO_STATIONARY_MPS} m/s in {frac_ok:.0%}, max |v| = {np.abs(est).max():.2f} m/s")
    return frac_ok >= 0.95


def test_synthetic(frames, cfg):
    period = cfg.get("doppler_period_mps") or 0.0
    vmax = cfg.get("max_doppler_mps") or period / 2
    n = len(frames)
    # ramp up to 1.2 m/s, hold, jump to 1.5 x v_max + 0.7 (aliased), hold, brake to 0
    def profile(i):
        s = i / max(n - 1, 1)
        if s < 0.15: return 1.2 * s / 0.15
        if s < 0.40: return 1.2
        if s < 0.45: return 1.2 + (1.5 * vmax + 0.7 - 1.2) * (s - 0.40) / 0.05
        if s < 0.80: return 1.5 * vmax + 0.7
        return max(0.0, (1.5 * vmax + 0.7) * (1 - (s - 0.80) / 0.15))
    est, truth = run(frames, cfg, profile)
    err = est - truth
    # judge only the steady parts (the rate limit is allowed to lag on the ramps)
    steady = np.array([0.20 <= i / max(n - 1, 1) <= 0.38 or 0.50 <= i / max(n - 1, 1) <= 0.78 for i in range(n)])
    rmse = float(np.sqrt(np.mean(err[steady] ** 2))); worst = float(np.abs(err[steady]).max())
    aliased = truth > vmax
    print(f"B synthetic: period {period:.2f} m/s, v_max {vmax:.2f} m/s, truth up to {truth.max():.2f} m/s "
          f"({aliased.sum()} frames beyond v_max) -> steady-state RMSE {rmse:.3f} m/s, worst {worst:.2f} m/s")
    return rmse < 0.15 and worst < 0.5


def test_yaw():
    """A post 6 m ahead, carrier turning left at 0.3 rad/s: without rotation the prediction drifts sideways by
    6*0.3*dt per frame (~0.12 m at 15 Hz) and the innovation grows; with it the innovation stays at sensor noise."""
    dt, w, r = 1 / 15, 0.3, 6.0
    def measured(k):                                   # the post's true position in the sensor frame after k frames
        th = -w * dt * k
        return r * math.sin(th), r * math.cos(th)      # x right, y forward
    res = {}
    for compensate in (False, True):
        radar.Track._next_id = 1
        trk = radar.Tracker()
        worst = 0.0
        for k in range(30):
            x, y = measured(k)
            obj = {"x_m": x, "y_m": y, "range_m": r, "azimuth_rad": math.atan2(x, y), "doppler_mps": 0.0,
                   "snr_db": 20.0, "n_points": 4, "kind": "static", "kind_conf": 0.5, "is_static": True,
                   "extent_m": 0.3, "cluster_v_std": 0.0}
            trk.step([obj], k * dt, dt, yaw_rate=(w if compensate else 0.0))
            if k > 5 and trk.tracks:
                tr = trk.tracks[0]
                worst = max(worst, math.hypot(tr.x[0] - x, tr.x[1] - y))
        res[compensate] = (worst, len(trk.tracks))
    print(f"C yaw: worst position error over the turn — without compensation {res[False][0]:.2f} m, with {res[True][0]:.2f} m; "
          f"tracks at the end {res[False][1]} / {res[True][1]}")
    return res[True][0] < 0.25 and res[True][0] < res[False][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="radar.bin / radar_dump_*.bin of a stationary radar")
    ap.add_argument("--cfg", default="radar_configs/profile_sdk3.cfg")
    a = ap.parse_args()
    cfg = radar.parse_cfg(a.cfg)
    frames = load_frames(a.dump)
    print(f"{len(frames)} frames from {a.dump}; Doppler period {cfg.get('doppler_period_mps')}, res {cfg.get('doppler_res_mps')}")
    ok = [test_standing(frames, cfg), test_synthetic(frames, cfg), test_yaw()]
    print("PASS" if all(ok) else f"FAIL {ok}")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
