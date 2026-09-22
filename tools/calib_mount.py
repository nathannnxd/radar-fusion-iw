"""tools/calib_mount.py — MOUNT_YAW_DEG from two straight drives (EGO_MOTION.md §3, §6).

    python tools/calib_mount.py rec_*_straight_fwd/run.jsonl.gz rec_*_straight_back/run.jsonl.gz [--cfg ...]

On a straight drive the stationary world moves along the vehicle axis; if the radar boresight is rotated by m
(left +) from that axis, the 2-parameter fit sees vy = v·cos m (forward), vx = v·sin m (lateral, right +) — the
vehicle's forward direction lies m to the RIGHT of the boresight. The mounting yaw is therefore the rotation that
zeroes the lateral component: m = atan2(vx, vy). Forward and reverse drives are averaged as vectors after flipping
the reverse ones (vy < 0) so both drives vote for the same angle; the lever-arm/yaw term is excluded by keeping only
frames with |w_gyro| < 2 deg/s (when $EGOVEL lines are present) and |v| >= 0.3 m/s.
Prints MOUNT_YAW_DEG (+ = boresight left of the vehicle forward axis) and the per-log values as a consistency check;
the two drives should agree within ~1°. Sign convention is that of iwr1642_live.points_to_detections
(azimuth = atan2(x, y), right +) — the estimator rotates azimuths by -MOUNT_YAW_DEG before fitting.
"""
import argparse
import math
import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
if not os.path.exists("configs.json"):
    os.chdir(_root)                                      # iwr1642_live reads configs.json from the working directory
import fusion_offline as fo                              # noqa: E402
import iwr1642_live as radar                             # noqa: E402

STRAIGHT_MAX_DPS = 2.0              # frames turning faster than this are not "straight" (lever-arm term would bias vx)
MIN_SPEED_MPS = 0.3                 # below this the lateral component is quantisation noise
MIN_INLIER_FRAC = 0.5               # a frame whose fit explains fewer points than this is skipped (movers, corridor walls)


def straight_vectors(log_path, cfg, gyro_bias_dps=0.0, out=print):
    """One straight-drive log -> (sum_vx, sum_vy, n_frames, per-frame angles deg) with reverse frames flipped."""
    log = fo.read_log(log_path)
    imu = fo.LoggedImu(0.0, gyro_bias_dps)
    for t_rx, line in log["ego"]:
        imu.feed_line(line, t_rx)
    gyro = imu.arrays() if len(imu.t) >= 2 else None
    tol, period = fo.inlier_tol(cfg), cfg.get("doppler_period_mps")
    sx = sy = 0.0; angles = []; v_prev = 0.0; n_turn = n_slow = n_nofit = 0
    for fr in log["radar"]:
        az, dop = fo.frame_points(fr, cfg)
        fit = fo.fit_ego_2p(az, dop, tol, period, v_ref=v_prev) if len(az) else None
        if fit is None or fit["n_inl"] < MIN_INLIER_FRAC * fit["n"]:
            n_nofit += 1; continue
        v_prev = fit["vy"]
        if gyro is not None and abs(float(np.interp(fr["t"], gyro[0], gyro[1] - gyro_bias_dps))) > STRAIGHT_MAX_DPS:
            n_turn += 1; continue
        vx, vy = fit["vx"], fit["vy"]
        if abs(vy) < MIN_SPEED_MPS:
            n_slow += 1; continue
        if vy < 0:                                       # reverse: the same mounting angle, mirrored vector
            vx, vy = -vx, -vy
        sx += vx; sy += vy
        angles.append(math.degrees(math.atan2(vx, vy)))
    angles = np.array(angles)
    m = math.degrees(math.atan2(sx, sy)) if angles.size else float("nan")
    out(f"{log_path}: {len(log['radar'])} frames -> {angles.size} straight moving frames used "
        f"(skipped: no fit {n_nofit}, turning {n_turn}, slow {n_slow}; gyro {'yes' if gyro is not None else 'no'}) · "
        f"mount yaw {m:+.2f}° (per-frame median {np.median(angles) if angles.size else float('nan'):+.2f}°, "
        f"IQR {np.percentile(angles, 75) - np.percentile(angles, 25) if angles.size else float('nan'):.2f}°)")
    return sx, sy, int(angles.size), angles


def estimate(log_fwd, log_back, cfg, gyro_bias_dps=0.0, out=print):
    """Both drives -> dict(mount_yaw_deg, per_log, n_frames)."""
    per = [straight_vectors(p, cfg, gyro_bias_dps, out) for p in (log_fwd, log_back)]
    sx = sum(p[0] for p in per); sy = sum(p[1] for p in per); n = sum(p[2] for p in per)
    if n == 0:
        raise SystemExit("no straight moving frames in either log — check the drill (|v| >= 0.3 m/s, |w| < 2 dps)")
    m = math.degrees(math.atan2(sx, sy))
    per_log = [math.degrees(math.atan2(p[0], p[1])) if p[2] else float("nan") for p in per]
    if all(np.isfinite(per_log)) and abs(per_log[0] - per_log[1]) > 1.0:
        out(f"⚠️ forward and reverse disagree by {abs(per_log[0] - per_log[1]):.2f}° — a side-slip or a curved drive; redo the straights")
    out(f"\nMOUNT_YAW_DEG = {m:+.2f}   ({n} frames; fwd {per_log[0]:+.2f}°, back {per_log[1]:+.2f}°)")
    out(f'configs.json:  "MOUNT_YAW_DEG": {m:.2f}')
    return {"mount_yaw_deg": m, "per_log": per_log, "n_frames": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_fwd", help="run.jsonl.gz of the forward straight drive")
    ap.add_argument("log_back", help="run.jsonl.gz of the reverse straight drive")
    ap.add_argument("--cfg", default=None, help="radar profile .cfg (default: the one stored in the log, else configs.json)")
    ap.add_argument("--gyro-bias", type=float, default=0.0, help="EGO_GYRO_BIAS_DPS (for the |w| < 2 dps straight gate)")
    a = ap.parse_args()
    cfg_path = a.cfg or fo.read_log(a.log_fwd)["meta"].get("cfg_file") or radar.CFG_FILE
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(_root, cfg_path)
    cfg = radar.parse_cfg(cfg_path)
    if not cfg.get("doppler_res_mps"):
        raise SystemExit(f"profile {cfg_path} not parsed — pass --cfg radar_configs/<profile>.cfg")
    print(f"profile {cfg_path}: bin {cfg['doppler_res_mps']:.3f} m/s")
    estimate(a.log_fwd, a.log_back, cfg, a.gyro_bias)


if __name__ == "__main__":
    main()
