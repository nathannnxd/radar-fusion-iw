"""test_ego.py — checks for the radar ego-motion estimate (iwr1642_live.EgoEstimator, EGO_MOTION.md §4/§8), the
profile parser, the per-profile static threshold, the short-memory ego buffer and the tracker's yaw compensation.

    python -m pytest -q test_ego.py                      # synthetic tests A–E (+ IMU bridge, gyro aiding, cross-check)
    python test_ego.py [radar.bin] [--walk radar_walk.bin] [--cfg radar_configs/xxx.cfg]
                                                          # same, plus the recorded-file tests when the files exist

Synthetic frames are generated the way the sensor sees a stationary world from a moving radar: every point's Doppler is
-(v_fwd*cos(az) + vx*sin(az)) with the lever-arm terms, plus noise, QUANTISED to the profile's Doppler bin and WRAPPED
by its period. Two profiles: hangar_v9 (bin 0.272 m/s, period 17.4 m/s — the tractor's speed never wraps) and the old
config-16.09.26 (bin 0.127, period 4.05 — 4.5 m/s aliases).

A. standing scene                       -> state STANDING, v = 0 exactly (>= 95 % of frames after the hysteresis)
B. straight drive 0.3 / 1 / 4.5 m/s     -> |v - v_true| < 0.5*bin on both profiles (wrapped Doppler on the old one)
C. turn at 0.3 rad/s, lever arm 2 m     -> with the gyro |v - v_true| < 0.5*bin; tracker: a post stays on its track
                                            only when the world is rotated by -yaw_rate*dt (th = w*dt*k, sign fixed)
D. one large mover covering 60 % of pts -> the plausibility gate holds the previous value
E. narrow azimuth spread                -> 'ambiguous', hold, then UNKNOWN after EGO_HOLD_S
+  IMU bridge (source 'imu-bridge' for <= 2 s), gyro cross-check (15 frames of disagreement -> gyro_ok False),
   parse_cfg numbers, static threshold, BackgroundMap persistence.
Recorded (skipped when the file is missing; RADAR_BIN / RADAR_WALK_BIN / RADAR_CFG env vars or the CLI arguments):
   test_stationary_stand.bin -> STANDING >= 95 % of frames; test_stationary_walk.bin -> walker moving, background static.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iwr1642_live as radar

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_HANGAR = os.path.join(HERE, "radar_configs", "hangar_v9.cfg")
CFG_OLD = os.path.join(HERE, "radar_configs", "config-16.09.26-12.43.cfg")
LX, LY = radar.LEVER_ARM_X_M, radar.LEVER_ARM_Y_M


# ---------------------------------------------------------------- synthetic scenes
def imu_sample(yaw_rate=0.0, vy_imu=None, zupt=False, gyro_ok=True, age_s=0.01):
    """The dict ego_velocity.EgoVelocityReader.sample_at(t) returns (EGO_MOTION.md §5)."""
    return {"yaw_rate": yaw_rate, "pitch": 0.0, "roll": 0.0, "acc_fwd": 0.0, "vy_imu": vy_imu,
            "zupt": zupt, "gyro_ok": gyro_ok, "age_s": age_s}


# what EgoVelocityReader.sample_at() returns with no link at all: vy_imu 0.0, age inf — must never bridge
IMU_NO_LINK = imu_sample(vy_imu=0.0, gyro_ok=False, age_s=float("inf"))


class Scene:
    """A stationary world seen from the moving radar. Points persist from frame to frame (the tracker then behaves as
    on a real scene): each frame the world is translated by -(vx, v_fwd)*dt and rotated by -w*dt in the sensor frame,
    points that leave the field of view are re-spawned, and every point's Doppler is -(v_fwd*cos az + vx*sin az) with
    the lever-arm terms, plus noise, quantised to the bin and wrapped by the period."""

    def __init__(self, cfg, rng, n=60, az_deg=(-60.0, 60.0), noise=0.03, r_lim=(1.5, 14.0)):
        self.cfg, self.rng, self.n, self.az_deg, self.noise, self.r_lim = cfg, rng, n, az_deg, noise, r_lim
        self.pts = self._spawn(n)
        self.k = 0

    def _spawn(self, m):
        az = np.radians(self.rng.uniform(self.az_deg[0], self.az_deg[1], m))
        r = self.rng.uniform(self.r_lim[0], self.r_lim[1], m)
        return np.column_stack([r * np.sin(az), r * np.cos(az)])

    def frame(self, v=0.0, w=0.0, mover=None, n_points=None):
        """mover=(frac, doppler_rel, az_lo, az_hi): that share of the points is one large object at relative Doppler
        doppler_rel inside the azimuth band (5..6.5 m). n_points: emit only this many points (a poor scene)."""
        cfg, rng = self.cfg, self.rng
        bin_, period, dt = cfg["doppler_res_mps"], cfg["doppler_period_mps"], cfg["frame_period_s"]
        v_fwd, vx = v - w * LY, -w * LX                                # radar velocity in its own frame (§4 model)
        if self.k > 0:                                                 # the world moves the other way in the sensor frame
            c, s = math.cos(-w * dt), math.sin(-w * dt)
            p = self.pts - np.array([vx, v_fwd]) * dt
            self.pts = np.column_stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c])
        r = np.hypot(self.pts[:, 0], self.pts[:, 1])
        az_now = np.degrees(np.arctan2(self.pts[:, 0], self.pts[:, 1]))
        gone = (r < self.r_lim[0]) | (r > self.r_lim[1]) | (az_now < self.az_deg[0]) | (az_now > self.az_deg[1])
        if gone.any():
            self.pts[gone] = self._spawn(int(gone.sum()))
        pts = self.pts.copy()
        m = 0
        if mover is not None:
            frac, d_rel, lo, hi = mover
            m = int(round(frac * self.n))
            az_m = np.radians(rng.uniform(lo, hi, m)); r_m = rng.uniform(5.0, 6.5, m)
            pts[:m] = np.column_stack([r_m * np.sin(az_m), r_m * np.cos(az_m)])
        if n_points is not None:
            pts = pts[:n_points]
        az = np.arctan2(pts[:, 0], pts[:, 1])
        dop = -(v_fwd * np.cos(az) + vx * np.sin(az)) + rng.normal(0.0, self.noise, len(pts))
        if m:
            dop[:m] += d_rel
        dop = np.round(dop / bin_) * bin_                              # the sensor quantises ...
        dop -= np.round(dop / period) * period                         # ... and wraps
        side = [(float(s_), -90.0) for s_ in rng.uniform(12.0, 30.0, len(pts))]
        fr = {"frame": self.k, "num_obj": len(pts), "sdk_major": 3, "range_profile_db": None, "side": side,
              "points": [[float(x), float(y), 0.0, float(d)] for (x, y), d in zip(pts, dop)]}
        self.k += 1
        return fr


def synth_frame(k, cfg, rng, v=0.0, w=0.0, n=60, az_deg=(-60.0, 60.0), mover=None):
    """One independent frame (no persistence) — for the plumbing tests."""
    fr = Scene(cfg, rng, n=n, az_deg=az_deg).frame(v=v, w=w, mover=mover)
    fr["frame"] = k
    return fr


def make_pipe(cfg, use_background=False):
    radar.Track._next_id = 1
    return radar.Pipeline(cfg, None, 0.0, use_background=use_background, ego_mode="radar")


def ramp(v_start, v_target, dt, k, a=2.0):
    """Speed at frame k when going from v_start to v_target at a m/s² (below the 2.5 m/s² plausibility gate)."""
    step = a * dt * k
    return min(v_start + step, v_target) if v_target >= v_start else max(v_start - step, v_target)


def drive(pipe, scene, v_target, frames, w=0.0, imu=None, v_start=0.0, **kw):
    """Ramp from v_start to v_target then hold; returns (list of ego_info, list of v_true)."""
    dt = scene.cfg["frame_period_s"]
    infos, truth = [], []
    for k in range(frames):
        v_true = ramp(v_start, v_target, dt, k)
        out = pipe.process(scene.frame(v=v_true, w=w, **kw), imu=imu)
        infos.append(out["ego"]); truth.append(v_true)
    return infos, truth


def steady_errors(infos, truth, dt, v_target, v_start=0.0):
    """|v - v_true| on frames after the ramp has ended plus 0.5 s of settling."""
    k_end = int(math.ceil(abs(v_target - v_start) / (2.0 * dt))) + int(0.5 / dt)
    return np.array([abs(i["v"] - vt) for i, vt in zip(infos[k_end:], truth[k_end:])])


# ---------------------------------------------------------------- profile numbers
def test_parse_cfg_start_frequency():
    h, o = radar.parse_cfg(CFG_HANGAR), radar.parse_cfg(CFG_OLD)
    assert h["num_tx"] == 2 and o["num_tx"] == 2                                  # TDM factor kept
    assert abs(h["doppler_res_mps"] - 0.272) < 0.002, h["doppler_res_mps"]        # hangar_v9 header: bin 0.27 m/s
    assert abs(h["doppler_period_mps"] - 17.4) < 0.05, h["doppler_period_mps"]    # 2 x v_max 8.7 m/s
    assert abs(o["doppler_period_mps"] - 4.06) < 0.02, o["doppler_period_mps"]    # old profile
    assert abs(h["frame_period_s"] - 1 / 15) < 1e-3


def test_static_threshold_from_bin():
    """STATIC_DOPPLER_MPS per point = max(0.2, 1.2*bin) (EGO_MOTION.md §5); the module constant stays as fallback."""
    h = radar.parse_cfg(CFG_HANGAR)
    assert abs(make_pipe(h).static_doppler_mps - 1.2 * h["doppler_res_mps"]) < 1e-9
    assert make_pipe(radar.parse_cfg(CFG_OLD)).static_doppler_mps == 0.2
    assert make_pipe({"frame_period_s": 0.1}).static_doppler_mps == radar.STATIC_DOPPLER_MPS


# ---------------------------------------------------------------- A. standing
def test_a_standing():
    for cfg_path in (CFG_HANGAR, CFG_OLD):
        cfg, rng = radar.parse_cfg(cfg_path), np.random.default_rng(1)
        pipe = make_pipe(cfg)
        infos, _ = drive(pipe, Scene(cfg, rng), 0.0, 40)
        states = [i["state"] for i in infos[radar.EGO_STATE_HYST_FRAMES:]]
        assert np.mean([s == "STANDING" for s in states]) >= 0.95, states
        assert all(i["v"] == 0.0 and i["valid"] for i in infos[radar.EGO_STATE_HYST_FRAMES:])
        assert not infos[-1]["moving"] and infos[-1]["source"] == "radar"


def test_a_standing_with_walkers_and_zupt():
    """A standing tractor with people walking (20 % of the points moving) stays STANDING; with the IMU's ZUPT the
    tolerated share is larger (40 %)."""
    cfg, rng = radar.parse_cfg(CFG_HANGAR), np.random.default_rng(2)
    pipe = make_pipe(cfg)
    infos, _ = drive(pipe, Scene(cfg, rng), 0.0, 30, mover=(0.2, 1.4, -30.0, 30.0))
    assert infos[-1]["state"] == "STANDING" and infos[-1]["v"] == 0.0
    pipe = make_pipe(cfg)
    infos, _ = drive(pipe, Scene(cfg, rng), 0.0, 30, imu=imu_sample(zupt=True), mover=(0.4, 1.4, -30.0, 30.0))
    assert infos[-1]["state"] == "STANDING" and infos[-1]["v"] == 0.0


# ---------------------------------------------------------------- B. straight drive
def test_b_straight_drive():
    for cfg_path in (CFG_HANGAR, CFG_OLD):
        cfg = radar.parse_cfg(cfg_path)
        bin_, dt = cfg["doppler_res_mps"], cfg["frame_period_s"]
        for v_target in (0.3, 1.0, 4.5):
            rng = np.random.default_rng(int(v_target * 10))
            pipe = make_pipe(cfg)
            infos, truth = drive(pipe, Scene(cfg, rng), v_target, 60)
            err = steady_errors(infos, truth, dt, v_target)
            worst = float(err.max())
            print(f"B {os.path.basename(cfg_path)} v={v_target}: steady worst |err| {worst:.3f} (0.5 bin = {0.5 * bin_:.3f}), "
                  f"state {infos[-1]['state']}, source {infos[-1]['source']}")
            assert worst < 0.5 * bin_, (cfg_path, v_target, worst)
            assert all(i["valid"] and i["source"] == "radar" for i in infos[-20:])
            # a real fit every frame, not a held value: the grid search must keep finding the stationary world
            assert all(i["n_inliers"] >= radar.EGO_MIN_POINTS for i in infos[-20:]), [i["n_inliers"] for i in infos[-20:]]
            expect = "CREEPING" if abs(v_target) <= 5 * bin_ else "MOVING"          # §4.3 literal: 1..5 bins
            assert infos[-1]["state"] == expect, (infos[-1]["state"], expect)
            assert infos[-1]["ttc_bound"] == (expect == "CREEPING")
    assert 4.5 > radar.parse_cfg(CFG_OLD)["max_doppler_mps"]                    # the old profile really wraps at 4.5


# ---------------------------------------------------------------- C. turn with the gyro
def test_c_turn_gyro_aided():
    w, v_target = 0.3, 1.5
    cfg = radar.parse_cfg(CFG_HANGAR)
    bin_, dt = cfg["doppler_res_mps"], cfg["frame_period_s"]
    rng = np.random.default_rng(3)
    pipe = make_pipe(cfg)
    infos, truth = drive(pipe, Scene(cfg, rng), v_target, 60, w=w, imu=imu_sample(yaw_rate=w))
    err = steady_errors(infos, truth, dt, v_target)
    last = infos[-1]
    print(f"C gyro-aided turn: worst |err| {err.max():.3f} (0.5 bin = {0.5 * bin_:.3f}), vx {last['vx']:+.2f} "
          f"(expect {-w * LX:+.2f}), yaw_rate used {last['yaw_rate']:.3f}, gyro_ok {last['gyro_ok']}")
    assert float(err.max()) < 0.5 * bin_
    assert all(i["n_inliers"] >= radar.EGO_MIN_POINTS for i in infos[-20:]), [i["n_inliers"] for i in infos[-20:]]
    assert last["gyro_ok"] and abs(last["yaw_rate"] - w) < 1e-6
    assert abs(last["vx"] - (-w * LX)) < 1e-6                                       # lateral term from the lever arm
    # without the gyro the 2-parameter fit absorbs the lateral term — still within the bound on a wide scene
    pipe = make_pipe(cfg)
    infos, truth = drive(pipe, Scene(cfg, np.random.default_rng(4)), v_target, 60, w=w)
    err = steady_errors(infos, truth, dt, v_target)
    assert float(err.max()) < 0.5 * bin_, err.max()
    assert all(i["n_inliers"] >= radar.EGO_MIN_POINTS for i in infos[-20:]), [i["n_inliers"] for i in infos[-20:]]
    assert not infos[-1]["gyro_ok"] and abs(infos[-1]["vx"] - (-w * LX)) < 0.25    # fitted lateral ≈ -w*LX


def test_c_tracker_yaw():
    """A post 6 m ahead, carrier turning left at 0.3 rad/s: the post moves RIGHT in the sensor frame (th = w*dt*k).
    Without rotation the prediction drifts by 6*0.3*dt per frame (~0.12 m at 15 Hz) and the innovation grows; with
    Track.rotate(-yaw_rate*dt) it stays at sensor noise."""
    dt, w, r = 1 / 15, 0.3, 6.0

    def measured(k):                                   # the post's true position in the sensor frame after k frames
        th = w * dt * k                                # the carrier turned left by th -> the world turned right
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
    assert res[True][0] < 0.25 and res[True][0] < res[False][0]


# ---------------------------------------------------------------- D. one large mover
def test_d_large_mover_gate_holds():
    cfg, rng = radar.parse_cfg(CFG_HANGAR), np.random.default_rng(5)
    bin_ = cfg["doppler_res_mps"]
    pipe, scene = make_pipe(cfg), Scene(cfg, rng, n=100)
    v = 2.0
    infos, _ = drive(pipe, scene, v, 60)
    assert abs(infos[-1]["v"] - v) < 0.5 * bin_
    # a truck (60 % of the points, +3 m/s relative, 20..30° right) for 10 frames — under EGO_HOLD_S
    infos2, _ = drive(pipe, scene, v, 10, v_start=v, mover=(0.6, 3.0, 20.0, 30.0))
    vs = [i["v"] for i in infos2]
    print(f"D mover: v over the mover frames {min(vs):.2f}..{max(vs):.2f} (truth {v}), "
          f"n_inliers {[i['n_inliers'] for i in infos2]}, valid {[i['valid'] for i in infos2]}")
    assert all(abs(x - v) < 0.5 * bin_ for x in vs), vs
    assert all(i["valid"] for i in infos2)
    # the same frames seen by a fresh estimator (no previous value, so no gate) follow the mover instead — i.e. the
    # gate, not the inlier count, is what held the value: 60 % of the points outvote the 40 % stationary world
    fresh, scene2 = make_pipe(cfg), Scene(cfg, np.random.default_rng(5), n=100)
    infos3, _ = drive(fresh, scene2, v, 10, v_start=v, mover=(0.6, 3.0, 20.0, 30.0))
    v_fresh = infos3[-1]["v"]
    print(f"D fresh estimator on the same scene: v {v_fresh:.2f} (mover is at v - 3 m/s relative)")
    assert abs(v_fresh - v) > 0.5 * bin_, v_fresh


# ---------------------------------------------------------------- E. narrow azimuth spread
def test_e_narrow_spread_ambiguous_hold():
    cfg, rng = radar.parse_cfg(CFG_OLD), np.random.default_rng(6)
    dt = cfg["frame_period_s"]
    pipe = make_pipe(cfg)
    v = 2.0
    infos, _ = drive(pipe, Scene(cfg, rng), v, 60)
    v_last = infos[-1]["v"]
    n_hold = int(radar.EGO_HOLD_S / dt)
    narrow = Scene(cfg, rng, az_deg=(-8.0, 8.0)); narrow.k = 60                  # a wall dead ahead
    infos2, _ = drive(pipe, narrow, v, n_hold + 15, v_start=v)
    assert all(i["ambiguous"] for i in infos2), [i["ambiguous"] for i in infos2]
    held = infos2[:n_hold - 1]
    assert all(i["valid"] and i["v"] == v_last for i in held), [(i["valid"], i["v"]) for i in held]
    assert infos2[-1]["state"] == "UNKNOWN" and not infos2[-1]["valid"] and infos2[-1]["v"] == 0.0


# ---------------------------------------------------------------- IMU bridge
def test_imu_bridge():
    cfg, rng = radar.parse_cfg(CFG_HANGAR), np.random.default_rng(7)
    dt = cfg["frame_period_s"]
    pipe, scene = make_pipe(cfg), Scene(cfg, rng)
    v = 2.0
    drive(pipe, scene, v, 45, imu=imu_sample(vy_imu=v))
    # radar loses the world (3 points) but the IMU still integrates 1.9 m/s and is not in ZUPT
    n_bridge = int(radar.EGO_BRIDGE_S / dt)
    infos, _ = drive(pipe, scene, v, n_bridge + 15, v_start=v, n_points=3, imu=imu_sample(vy_imu=1.9))
    bridged = infos[:n_bridge - 1]
    assert all(i["source"] == "imu-bridge" and i["valid"] and i["v"] == 1.9 for i in bridged), \
        [(i["source"], i["v"]) for i in bridged[:5]]
    assert infos[-1]["source"] == "radar" and not infos[-1]["valid"] and infos[-1]["v"] == 0.0
    assert infos[-1]["state"] == "UNKNOWN"
    # no bridge when the IMU reports ZUPT (the tractor may well be standing): plain hold, then UNKNOWN
    pipe, scene = make_pipe(cfg), Scene(cfg, rng)
    drive(pipe, scene, v, 45, imu=imu_sample(vy_imu=v))
    infos, _ = drive(pipe, scene, v, 5, v_start=v, n_points=3, imu=imu_sample(vy_imu=0.0, zupt=True))
    assert all(i["source"] == "radar" for i in infos)
    # and none with no IMU link at all (vy_imu 0.0, age inf) — that would report "standing" mid-drive
    for imu in (IMU_NO_LINK, None):
        pipe, scene = make_pipe(cfg), Scene(cfg, rng)
        drive(pipe, scene, v, 45, imu=imu)
        infos, _ = drive(pipe, scene, v, 5, v_start=v, n_points=3, imu=imu)
        assert all(i["source"] == "radar" and i["v"] > 0.5 * v for i in infos), [(i["source"], i["v"]) for i in infos]


# ---------------------------------------------------------------- gyro cross-check
def test_gyro_cross_check_flags_bad_gyro():
    """A gyro reading 10 deg/s off the truth on rich, wide frames while MOVING: after 15 frames of disagreement the
    estimator flags gyro_ok False and falls back to the 2-parameter fit; yaw_rate_radar tracks the true rate."""
    cfg, rng = radar.parse_cfg(CFG_HANGAR), np.random.default_rng(8)
    bin_, dt = cfg["doppler_res_mps"], cfg["frame_period_s"]
    w_true, w_gyro, v = 0.3, 0.3 + math.radians(10.0), 3.0
    pipe = make_pipe(cfg)
    infos, truth = drive(pipe, Scene(cfg, rng, n=100, az_deg=(-70.0, 70.0)), v, 60, w=w_true,
                         imu=imu_sample(yaw_rate=w_gyro))
    flags = [i["gyro_ok"] for i in infos]
    yr = [i["yaw_rate_radar"] for i in infos if i["yaw_rate_radar"] is not None]
    print(f"cross-check: gyro_ok first False at frame {flags.index(False) if False in flags else None}, "
          f"yaw_rate_radar median {np.median(yr):.3f} (truth {w_true}), n_inliers {infos[-1]['n_inliers']}, "
          f"spread {infos[-1]['az_spread_deg']}")
    assert not infos[-1]["gyro_ok"]
    assert abs(float(np.median(yr[-30:])) - w_true) < math.radians(3.0)
    assert float(steady_errors(infos, truth, dt, v).max()) < 0.5 * bin_
    # an honest gyro keeps gyro_ok
    pipe = make_pipe(cfg)
    infos, _ = drive(pipe, Scene(cfg, np.random.default_rng(9), n=100, az_deg=(-70.0, 70.0)), v, 60, w=w_true,
                     imu=imu_sample(yaw_rate=w_true))
    assert all(i["gyro_ok"] for i in infos)


# ---------------------------------------------------------------- §4 constants, pinned as literals
def test_ego_constants_match_the_spec():
    """The numbers EGO_MOTION.md §4 states, written out. The behavioural tests size their own frame counts and
    accept bands from these constants, so a wrong value would otherwise still pass everything."""
    assert radar.EGO_MIN_POINTS == 6 and radar.EGO_MIN_INLIER_FRAC == 0.4            # §4.5
    assert radar.EGO_MIN_AZ_SPREAD_DEG == 25.0 and radar.EGO_MAX_ACCEL_MPS2 == 2.5   # §4.5
    assert radar.EGO_CREEP_BINS == 5 and radar.EGO_STATE_HYST_FRAMES == 2            # §4.3
    assert radar.EGO_STANDING_GYRO_DPS == 1.5 and radar.EGO_INLIER_TOL_MPS == 0.06   # §4.3, §4.2
    assert radar.EGO_HOLD_S == 1.0 and radar.EGO_BRIDGE_S == 2.0                     # §4.3, §1
    assert radar.EGO_XCHECK_MIN_INLIERS == 40 and radar.EGO_XCHECK_MIN_SPREAD_DEG == 60.0   # §4.6
    assert radar.EGO_XCHECK_TOL_DPS == 3.0 and radar.EGO_XCHECK_FRAMES == 15         # §4.6
    assert radar.EGO_SLOPE_DEG == 8.0 and radar.EGO_STATIONARY_MPS == 0.15           # §4.1, §5
    assert radar.BACKGROUND_PERSIST_FRAMES == 3                                      # §5


def test_hold_and_bridge_are_measured_in_seconds():
    """EGO_HOLD_S = 1.0 s and EGO_BRIDGE_S = 2.0 s as WALL-CLOCK durations: the other tests derive their frame
    counts from the constants themselves and would pass with either one ten times too large."""
    cfg = radar.parse_cfg(CFG_HANGAR)
    dt, v = cfg["frame_period_s"], 2.0
    pipe, scene = make_pipe(cfg), Scene(cfg, np.random.default_rng(20))
    drive(pipe, scene, v, 45)
    infos, _ = drive(pipe, scene, v, int(1.6 / dt), v_start=v, n_points=3)          # no IMU: plain hold
    t_last_valid = max(k for k, i in enumerate(infos) if i["valid"]) * dt
    assert 1.0 - 2 * dt <= t_last_valid <= 1.0 + dt, t_last_valid
    pipe, scene = make_pipe(cfg), Scene(cfg, np.random.default_rng(20))
    drive(pipe, scene, v, 45, imu=imu_sample(vy_imu=v))
    infos, _ = drive(pipe, scene, v, int(2.7 / dt), v_start=v, n_points=3, imu=imu_sample(vy_imu=1.9))
    t_last_bridge = max(k for k, i in enumerate(infos) if i["source"] == "imu-bridge") * dt
    assert 2.0 - 2 * dt <= t_last_bridge <= 2.0 + dt, t_last_bridge


def test_turning_in_place_is_not_standing():
    """§4.3: a scene that reads zero but |gyro| >= EGO_STANDING_GYRO_DPS (1.5 °/s) is CREEPING, not STANDING --
    turning in place must not force v := 0 and silence the creep flags."""
    cfg = radar.parse_cfg(CFG_HANGAR)
    w = math.radians(5.0)
    pipe = make_pipe(cfg)
    infos, _ = drive(pipe, Scene(cfg, np.random.default_rng(21)), 0.0, 30, w=w, imu=imu_sample(yaw_rate=w))
    assert infos[-1]["state"] == "CREEPING", [i["state"] for i in infos[-6:]]
    assert infos[-1]["ttc_bound"]
    pipe = make_pipe(cfg)                                                            # the same scene, gyro quiet
    infos, _ = drive(pipe, Scene(cfg, np.random.default_rng(21)), 0.0, 30, imu=imu_sample(yaw_rate=0.0))
    assert infos[-1]["state"] == "STANDING"


def test_zupt_alone_does_not_stand_a_moving_tractor():
    """§4.3: on a frame with fewer than EGO_MIN_POINTS returns the IMU's ZUPT flag alone must not declare STANDING.
    At a constant 4 m/s the firmware's ZUPT is true (|a_fwd|, |a_right| and |yaw| all under threshold), and forcing
    v := 0 there would silence every closing-speed / TTC / corridor alert."""
    cfg = radar.parse_cfg(CFG_HANGAR)
    dt, v = cfg["frame_period_s"], 4.0
    pipe, scene = make_pipe(cfg), Scene(cfg, np.random.default_rng(22))
    drive(pipe, scene, v, 50)
    assert pipe.ego_est.state == "MOVING"
    infos, _ = drive(pipe, scene, v, int(2.0 / dt), v_start=v, n_points=4,
                     imu=imu_sample(zupt=True, gyro_ok=False, vy_imu=0.0))
    assert all(i["state"] != "STANDING" for i in infos), [i["state"] for i in infos]
    assert infos[-1]["state"] == "UNKNOWN" and not infos[-1]["valid"]                # -> alert 131, not a fake zero


def test_standing_publishes_this_frames_raw():
    """§4.3 forces v := 0 inside the one-bin STANDING deadband (0.27 m/s on hangar_v9), so the only number left
    that can tell a parked tractor from one creeping inside that band is 'raw' — and it has to be THIS frame's
    fit, not whatever the estimator last saw while moving. alerts.py gates the whole collision layer on it (§5)."""
    cfg = radar.parse_cfg(CFG_HANGAR)
    bin_ = cfg["doppler_res_mps"]
    pipe, scene = make_pipe(cfg), Scene(cfg, np.random.default_rng(23), n=90, az_deg=(-55.0, 55.0))
    infos, _ = drive(pipe, scene, 2.0, 40)
    assert infos[-1]["state"] == "MOVING" and abs(infos[-1]["raw"] - 2.0) < 0.5 * bin_
    infos, _ = drive(pipe, scene, 0.0, 40, v_start=2.0)                              # roll to a stop
    last = infos[-1]
    print(f"stop: state {last['state']}, v {last['v']}, raw {last['raw']} (bin {bin_:.3f})")
    assert last["state"] == "STANDING" and last["v"] == 0.0
    assert abs(last["raw"]) < radar.EGO_STATIONARY_MPS, last["raw"]                  # not the stale 2.0 m/s


def test_turn_compensation_keeps_clutter_static():
    """§5: the per-point compensation uses the full sensor-frame ego velocity -- forward v AND the fit's lateral
    vx = -w*LEVER_ARM_X_M. Forward-only leaves w*LEVER_ARM_X_M*sin(az) on every stationary return, so a headland
    turn labels the wide-azimuth clutter 'moving' and BackgroundMap stops collecting corridor evidence."""
    cfg = radar.parse_cfg(CFG_HANGAR)
    w, v = 0.3, 3.0
    pipe = make_pipe(cfg)
    scene = Scene(cfg, np.random.default_rng(24), n=150, az_deg=(-60.0, 60.0), noise=0.0)
    drive(pipe, scene, v, 40, w=w, imu=imu_sample(yaw_rate=w))
    out = pipe.process(scene.frame(v=v, w=w), imu=imu_sample(yaw_rate=w))
    st = np.array([d["is_static"] for d in out["dets"]])
    wide = np.array([d["is_static"] for d in out["dets"] if abs(d["azimuth_rad"]) > math.radians(40.0)])
    print(f"turn compensation: static {st.sum()}/{len(st)}, beyond 40 deg {wide.sum()}/{len(wide)}, "
          f"vx {out['ego']['vx']:+.2f} (expect {-w * LX:+.2f})")
    assert abs(out["ego"]["vx"] - (-w * LX)) < 0.05
    assert st.mean() > 0.95, st.mean()
    assert len(wide) and wide.mean() > 0.9, wide.mean()


def test_cluster_static_label():
    """§5 cluster-level label: |median doppler_rel| of >= 3 members against 0.8*bin, immune to the +-half-bin
    quantisation that makes single points of one post disagree with each other. Fewer members -> None."""
    bin_ = 0.272
    saved, radar.DOPPLER_PERIOD = radar.DOPPLER_PERIOD, None
    try:
        def cluster(dopplers, bin_mps=bin_):
            dets = [{"x_m": 0.05 * i, "y_m": 6.0, "range_m": 6.0, "azimuth_rad": 0.0, "doppler_mps": d,
                     "doppler_rel_mps": d, "snr_db": 20.0} for i, d in enumerate(dopplers)]
            return radar.cluster_objects(dets, ["target"] * len(dets), [90.0] * len(dets), bin_mps=bin_mps)
        post = cluster([bin_, bin_, 0.0, -bin_])               # one post, quantised: median +0.136 < 0.8*bin
        assert len(post) == 1 and post[0]["is_static"] is True
        walker = cluster([0.28, 0.30, 0.26, 0.29])             # a real mover: median 0.285 > 0.218
        assert len(walker) == 1 and walker[0]["is_static"] is False
        pair = cluster([0.0, 0.0])                             # too few members to say (§5: >= 3)
        assert len(pair) == 1 and pair[0]["is_static"] is None
        no_bin = cluster([0.0, 0.0, 0.0], bin_mps=None)        # no profile bin -> no verdict
        assert len(no_bin) == 1 and no_bin[0]["is_static"] is None
    finally:
        radar.DOPPLER_PERIOD = saved


def test_background_counts_are_capped():
    """§5: nothing decays while STANDING, so the persistence counter is capped -- a normal pre-run standstill must
    not bank hundreds of frames of 'evidence' and bypass BACKGROUND_PERSIST_FRAMES for the first half-minute of
    driving (that counter is what Fusion.on_radar turns into f['obstacle'] and alert 141)."""
    cfg, rng = radar.parse_cfg(CFG_HANGAR), np.random.default_rng(25)
    pipe = make_pipe(cfg, use_background=True)
    post = [[1.0, 6.0, 0.0, 0.0]]
    out = None
    for k in range(120):                                       # 8 s parked at 15 fps
        fr = synth_frame(k, cfg, rng)
        fr["points"] = post + fr["points"]; fr["side"] = [(25.0, -90.0)] + fr["side"]
        out = pipe.process(fr)
    assert pipe.ego_est.state == "STANDING"
    assert float(pipe.bg.counts.max()) == radar.BACKGROUND_COUNT_CAP, float(pipe.bg.counts.max())
    assert out["dets"][0]["persist"] == radar.BACKGROUND_COUNT_CAP


def test_slope_is_logged_not_rejected():
    """§4.1: pitch/roll reach info() with a 'slope' flag over EGO_SLOPE_DEG -- logged, never a point rejection."""
    cfg = radar.parse_cfg(CFG_HANGAR)
    imu = {**imu_sample(yaw_rate=0.0), "pitch": 11.0, "roll": -2.0}
    pipe = make_pipe(cfg)
    infos, _ = drive(pipe, Scene(cfg, np.random.default_rng(26)), 1.0, 30, imu=imu)
    assert infos[-1]["pitch_deg"] == 11.0 and infos[-1]["roll_deg"] == -2.0 and infos[-1]["slope"] is True
    assert infos[-1]["n_points"] >= 40                         # nothing was dropped for the slope
    pipe = make_pipe(cfg)
    infos, _ = drive(pipe, Scene(cfg, np.random.default_rng(26)), 1.0, 30, imu=imu_sample(yaw_rate=0.0))
    assert infos[-1]["slope"] is False


# ---------------------------------------------------------------- pipeline plumbing
def test_pipeline_imu_and_legacy_yaw():
    cfg, rng = radar.parse_cfg(CFG_HANGAR), np.random.default_rng(10)
    pipe = make_pipe(cfg)
    out = pipe.process(synth_frame(0, cfg, rng), yaw_rate=math.radians(5.0))       # legacy caller: yaw for the tracker
    assert abs(out["ego_yaw_rate_dps"] - 5.0) < 1e-9 and abs(out["ego"]["yaw_rate"] - math.radians(5.0)) < 1e-12
    out = pipe.process(synth_frame(1, cfg, rng), yaw_rate=0.0, imu=imu_sample(yaw_rate=0.1))   # the imu dict wins
    assert abs(out["ego_yaw_rate_dps"] - math.degrees(0.1)) < 1e-9 and out["ego"]["gyro_ok"]
    for key in ("ego", "ego_speed_mps", "ego_yaw_rate_dps"):
        assert key in out
    for key in ("state", "bin", "yaw_rate", "yaw_rate_radar", "gyro_ok", "gyro_bias_dps", "time_offset_s",
                "n_inliers", "n_points", "az_spread_deg", "ambiguous", "source", "v", "vx", "valid", "moving"):
        assert key in out["ego"], key
    out = pipe.process(synth_frame(2, cfg, rng), ego_speed=1.0)                    # external speed keeps the schema
    assert out["ego"]["source"] == "external" and out["ego"]["v"] == 1.0 and "state" in out["ego"]


def test_background_persistence():
    """The ego buffer counts frames a static return was seen in a cell (±1 cell) and never suppresses."""
    cfg, rng = radar.parse_cfg(CFG_HANGAR), np.random.default_rng(11)
    pipe = make_pipe(cfg, use_background=True)
    post = [[1.0, 6.0, 0.0, 0.0]]                                                  # one post, standing radar
    for k in range(5):
        fr = synth_frame(k, cfg, rng)
        fr["points"] = post + fr["points"]; fr["side"] = [(25.0, -90.0)] + fr["side"]
        out = pipe.process(fr)
    d0 = out["dets"][0]
    assert d0["persist"] == 5 and d0["is_static"] and not d0["background"], d0
    assert pipe.bg.n_cells >= 1 and not pipe.bg.learning
    # moving: the map is dead-reckoned with the carrier, the post keeps its persistence, a mover gets none
    pipe = make_pipe(cfg, use_background=True)
    for k in range(40):
        fr = synth_frame(k, cfg, rng, v=1.0)
        y = 8.0 - 1.0 * cfg["frame_period_s"] * k
        fr["points"] = [[0.5, y, 0.0, -1.0 * math.cos(math.atan2(0.5, y))], [-2.0, 5.0, 0.0, 1.2]] + fr["points"]
        fr["side"] = [(25.0, -90.0), (25.0, -90.0)] + fr["side"]
        out = pipe.process(fr)
    dets = out["dets"]
    assert dets[0]["is_static"] and dets[0]["persist"] >= radar.BACKGROUND_PERSIST_FRAMES, dets[0]
    assert not dets[1]["is_static"] and dets[1]["persist"] <= 2, dets[1]
    assert not any(d["background"] for d in dets)


# ---------------------------------------------------------------- recorded files (optional)
def _cfg_env():
    return os.environ.get("RADAR_CFG") or os.path.join(HERE, radar.code_config.get("RADAR_CONFIG", "radar_configs/profile_sdk3.cfg"))


def _load_frames(path):
    data = open(path, "rb").read()
    return [f for f in radar.frames_from_bytes(data) if f]


def _skip_unless(path):
    if not path or not os.path.exists(path):
        try:
            import pytest
            pytest.skip(f"recording {path!r} not found")
        except ImportError:
            print(f"skip: recording {path!r} not found")
        return False
    return True


def test_recorded_stationary_stand():
    path = os.environ.get("RADAR_BIN", os.path.join(HERE, "test_stationary_stand.bin"))
    if not _skip_unless(path):
        return
    cfg, frames = radar.parse_cfg(_cfg_env()), _load_frames(path)
    pipe = make_pipe(cfg)
    states = [pipe.process(fr)["ego"]["state"] for fr in frames]
    frac = float(np.mean([s == "STANDING" for s in states[radar.EGO_STATE_HYST_FRAMES:]]))
    print(f"recorded stand: {len(frames)} frames, STANDING in {frac:.0%}")
    assert frac >= 0.95


def test_recorded_stationary_walk():
    path = os.environ.get("RADAR_WALK_BIN", os.path.join(HERE, "test_stationary_walk.bin"))
    if not _skip_unless(path):
        return
    cfg, frames = radar.parse_cfg(_cfg_env()), _load_frames(path)
    pipe = make_pipe(cfg)
    n_static = n_moving = 0
    for fr in frames:
        out = pipe.process(fr)
        n_static += sum(d["is_static"] for d in out["dets"]); n_moving += sum(not d["is_static"] for d in out["dets"])
    print(f"recorded walk: {len(frames)} frames, static points {n_static}, moving {n_moving}, ego v {out['ego']['v']}")
    assert n_moving > 0 and n_static > n_moving                                     # the walker moves, the room does not
    assert abs(out["ego"]["v"]) < radar.EGO_STATIONARY_MPS


# ---------------------------------------------------------------- script entry
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="?", help="radar.bin of a stationary radar (test_stationary_stand.bin)")
    ap.add_argument("--walk", help="radar.bin of a stationary radar with a walker (test_stationary_walk.bin)")
    ap.add_argument("--cfg", help="profile the recordings were made with (default: configs.json RADAR_CONFIG)")
    a = ap.parse_args()
    if a.dump:
        os.environ["RADAR_BIN"] = a.dump
    if a.walk:
        os.environ["RADAR_WALK_BIN"] = a.walk
    if a.cfg:
        os.environ["RADAR_CFG"] = a.cfg
    try:
        import pytest
    except ImportError:
        pytest = None
    if pytest is not None:
        sys.exit(pytest.main(["-q", "-s", __file__]))
    ok = True                                                                      # no pytest on the Pi: plain runner
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                ok = False; print(f"FAIL {name}: {e}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
