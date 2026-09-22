"""tools/calib_time_offset.py — EGO_TIME_OFFSET_S from a step-turn recording (EGO_MOTION.md §3, §6).

    python tools/calib_time_offset.py rec_*_stepturn/run.jsonl.gz [--cfg radar_configs/hangar_v9.cfg]
                                      [--lever-arm-x 2.0] [--lever-arm-y 0.0] [--mount-yaw 0] [--gyro-bias 0.3]

Two independent estimates of the offset between the radar frame timestamps and the IMU stream, both on the
recording's own clock (record_sync.py stamps both with the same perf_counter):
  (a) replay grid search over EGO_TIME_OFFSET_S in [-0.4, +0.1] s, step 10 ms: for every candidate the gyro yaw
      rate is interpolated to t_frame + offset, the lever-arm term is removed from every point's Doppler and the
      single-unknown speed is fitted (fusion_offline.fit_ego_1p_lever, Cauchy IRLS); the offset minimising the
      summed fit residual over the TURNING frames (|w| > 5 deg/s) wins, refined to sub-step by a parabola through
      the three lowest grid points. This is the primary number — it measures exactly what the estimator will use.
  (b) cross-correlation of the gyro yaw rate against the radar's own yaw rate (2-parameter fit's lateral component
      / LEVER_ARM_X_M) as an independent check — insensitive to the lever-arm sign conventions of (a).
Pass criterion: |a - b| < 1 radar frame period. The tool prints both, the frame period and PASS/FAIL.

Sign convention (matches iwr1642_live.points_to_detections and EgoVelocityReader.sample_at): azimuth = atan2(x, y),
right +; yaw rate + = left (CCW from above); LEVER_ARM_X_M forward +, LEVER_ARM_Y_M left +; the IMU time that
belongs to a radar frame is t_frame + EGO_TIME_OFFSET_S (negative: the radar's timestamp is later than the IMU's —
the radar lags). The lever arm must be non-zero for (a) to see anything: with the radar on the yaw axis the
turn leaves no lateral Doppler signature at all.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
if not os.path.exists("configs.json"):
    os.chdir(_root)                                      # iwr1642_live reads configs.json from the working directory
import fusion_offline as fo                              # noqa: E402  (log reading, $EGOVEL parsing, the fits)
import iwr1642_live as radar                             # noqa: E402

OFFSET_RANGE_S = (-0.40, 0.10)      # search window for EGO_TIME_OFFSET_S — EGO_MOTION.md §6
OFFSET_STEP_S = 0.010               # grid step — §6
TURN_MIN_DPS = 5.0                  # a frame counts as "turning" above this gyro yaw rate — §6
XCORR_MIN_V_MPS = 0.3               # radar yaw rate from the 2-parameter fit is meaningless when nearly standing


def calib_keys(args):
    """Lever arm / mount yaw / gyro bias: CLI overrides, else configs.json (EGO_MOTION.md §3), else 0."""
    try:
        cfg = json.load(open(os.path.join(_root, "configs.json"), encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    pick = lambda cli, key: float(cli) if cli is not None else float(cfg.get(key) or 0.0)
    return {"lx": pick(args.lever_arm_x, "LEVER_ARM_X_M"), "ly": pick(args.lever_arm_y, "LEVER_ARM_Y_M"),
            "mount_yaw_deg": pick(args.mount_yaw, "MOUNT_YAW_DEG"), "gyro_bias_dps": pick(args.gyro_bias, "EGO_GYRO_BIAS_DPS")}


def prepare_frames(log, cfg, mount_yaw_deg=0.0):
    """Per frame: t, azimuth (vehicle frame — rotated by MOUNT_YAW_DEG), Doppler, and the offset-independent
    2-parameter fit (seed for the lever-arm fit + the radar yaw rate for the cross-correlation)."""
    tol, period = fo.inlier_tol(cfg), cfg.get("doppler_period_mps")
    out, v_prev = [], 0.0
    for fr in log["radar"]:
        az, dop = fo.frame_points(fr, cfg)
        az = az - math.radians(mount_yaw_deg)              # boresight left of the vehicle axis -> a point sits further right
        fit = fo.fit_ego_2p(az, dop, tol, period, v_ref=v_prev) if len(az) else None
        if fit is not None:
            v_prev = fit["vy"]
        out.append({"t": fr["t"], "az": az, "dop": dop, "fit": fit})
    return out


def refine_parabola(x, y, i):
    """Sub-step minimum of a sampled curve through the three points around index i (falls back to x[i] at the edges)."""
    if i <= 0 or i >= len(x) - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    if den <= 0:
        return float(x[i])
    return float(x[i] + 0.5 * (y0 - y2) / den * (x[i + 1] - x[i]))


def grid_search(frames, gyro_t, gyro_dps, lx, ly, tol, period, offsets):
    """(a): summed lever-arm-compensated fit cost over the turning frames for every candidate offset.
    Turning frames are chosen once, at offset 0 with a widened window, so the set does not change with the candidate."""
    turning = []
    for f in frames:
        if f["fit"] is None or len(f["az"]) < radar.EGO_MIN_POINTS:
            continue
        w0 = np.interp(f["t"] + np.array([offsets[0], 0.0, offsets[-1]]), gyro_t, gyro_dps)
        if np.max(np.abs(w0)) > TURN_MIN_DPS:
            turning.append(f)
    if not turning:
        return None, None, 0
    costs = np.zeros(len(offsets))
    for k, off in enumerate(offsets):
        tot = 0.0
        for f in turning:
            w = math.radians(float(np.interp(f["t"] + off, gyro_t, gyro_dps)))
            _, cost, _ = fo.fit_ego_1p_lever(f["az"], f["dop"], w, lx, ly, tol, period, v0=f["fit"]["vy"])
            tot += cost if np.isfinite(cost) else tol * tol
        costs[k] = tot / len(turning)
    i = int(np.argmin(costs))
    return refine_parabola(offsets, costs, i), costs, len(turning)


def cross_correlation(frames, gyro_t, gyro_dps, lx, offsets):
    """(b): normalised cross-correlation of gyro yaw rate (interpolated to t_frame + offset) against the radar yaw
    rate from the 2-parameter fit, yaw_radar = v_lat_left / LEVER_ARM_X_M = -vx_right / lx (EGO_MOTION.md §4.6)."""
    sel = [f for f in frames if f["fit"] is not None and abs(f["fit"]["vy"]) >= XCORR_MIN_V_MPS]
    if len(sel) < 10 or abs(lx) < 1e-6:
        return None, None, len(sel)
    t = np.array([f["t"] for f in sel])
    yr = np.degrees(np.array([-f["fit"]["vx"] for f in sel]) / lx)
    yr = yr - np.median(yr)
    if np.std(yr) < 1e-6:
        return None, None, len(sel)
    corr = np.zeros(len(offsets))
    for k, off in enumerate(offsets):
        g = np.interp(t + off, gyro_t, gyro_dps)
        g = g - np.mean(g)
        den = np.sqrt(np.sum(g * g) * np.sum(yr * yr))
        corr[k] = float(np.sum(g * yr) / den) if den > 0 else 0.0
    i = int(np.argmax(corr))
    return refine_parabola(offsets, -corr, i), corr, len(sel)


def estimate(log_path, cfg, lx, ly, mount_yaw_deg=0.0, gyro_bias_dps=0.0, out=print):
    """Full procedure on one run.jsonl.gz (path, or a dict from fo.read_log) -> dict(offset_grid, offset_xcorr,
    frame_period_s, passed, ...)."""
    log = fo.read_log(log_path) if isinstance(log_path, str) else log_path
    name = log_path if isinstance(log_path, str) else (log.get("meta", {}).get("tag") or "log")
    imu = fo.LoggedImu(0.0, gyro_bias_dps)
    for t_rx, line in log["ego"]:
        imu.feed_line(line, t_rx)
    if not log["radar"] or len(imu.t) < 10:
        raise SystemExit(f"{name}: radar {len(log['radar'])} frames, usable $EGOVEL {len(imu.t)} lines — nothing to calibrate")
    gyro_t, gyro_dps = imu.arrays()
    gyro_dps = gyro_dps - gyro_bias_dps
    frames = prepare_frames(log, cfg, mount_yaw_deg)
    ft = np.array([f["t"] for f in frames])
    period_cfg = cfg.get("frame_period_s")
    period_meas = float(np.median(np.diff(ft))) if len(ft) > 1 else float("nan")
    frame_period = period_cfg or period_meas
    tol, dperiod = fo.inlier_tol(cfg), cfg.get("doppler_period_mps")
    offsets = np.round(np.arange(OFFSET_RANGE_S[0], OFFSET_RANGE_S[1] + 1e-9, OFFSET_STEP_S), 3)

    a, costs, n_turn = grid_search(frames, gyro_t, gyro_dps, lx, ly, tol, dperiod, offsets)
    b, corr, n_x = cross_correlation(frames, gyro_t, gyro_dps, lx, offsets)

    out(f"log {name}: {len(frames)} radar frames ({ft[0]:.2f}–{ft[-1]:.2f} s), {len(imu.t)} $EGOVEL samples "
        f"({imu.n_bad} rejected), gyro |w| max {np.max(np.abs(gyro_dps)):.1f} dps")
    out(f"frame period: {frame_period * 1000:.1f} ms (cfg {period_cfg}, measured median {period_meas * 1000:.1f} ms) · "
        f"lever arm x {lx:+.2f} y {ly:+.2f} m · mount yaw {mount_yaw_deg:+.1f}° · gyro bias {gyro_bias_dps:+.2f} dps")
    if abs(lx) < 1e-6 and abs(ly) < 1e-6:
        out("⚠️ lever arm is 0 — the grid search cannot see the turn; set LEVER_ARM_X_M (tape measure, EGO_MOTION.md §3)")
    if a is None:
        out(f"(a) grid search: no turning frames (|w| > {TURN_MIN_DPS} dps) with a usable fit — is this the step-turn drill?")
    else:
        i0 = int(np.argmin(np.abs(offsets)))
        out(f"(a) grid search: offset = {a:+.3f} s over {n_turn} turning frames · cost min {costs.min():.4f} vs at 0: {costs[i0]:.4f}")
    if b is None:
        out(f"(b) cross-correlation: not enough moving frames with a 2-parameter fit ({n_x}) or lever arm x = 0")
    else:
        out(f"(b) cross-correlation: offset = {b:+.3f} s over {n_x} frames · peak corr {corr.max():+.3f}")
    passed = a is not None and b is not None and abs(a - b) < frame_period
    if a is not None and b is not None:
        out(f"|a - b| = {abs(a - b) * 1000:.0f} ms vs 1 frame = {frame_period * 1000:.0f} ms -> {'PASS' if passed else 'FAIL'}")
    else:
        out("-> FAIL (one of the estimates is missing)")
    if a is not None:
        out(f'\nconfigs.json:  "EGO_TIME_OFFSET_S": {a:.3f}')
    return {"offset_grid": a, "offset_xcorr": b, "frame_period_s": frame_period, "passed": passed,
            "n_turning": n_turn, "n_xcorr": n_x, "offsets": offsets, "costs": costs, "corr": corr}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="record_sync run.jsonl.gz of the step-turn drill")
    ap.add_argument("--cfg", default=None, help="radar profile .cfg (default: the one stored in the log, else configs.json)")
    ap.add_argument("--lever-arm-x", type=float, default=None, help="LEVER_ARM_X_M (default configs.json)")
    ap.add_argument("--lever-arm-y", type=float, default=None, help="LEVER_ARM_Y_M (default configs.json)")
    ap.add_argument("--mount-yaw", type=float, default=None, help="MOUNT_YAW_DEG (default configs.json)")
    ap.add_argument("--gyro-bias", type=float, default=None, help="EGO_GYRO_BIAS_DPS (default configs.json)")
    a = ap.parse_args()
    keys = calib_keys(a)
    log = fo.read_log(a.log)
    cfg_path = a.cfg or log["meta"].get("cfg_file") or radar.CFG_FILE
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(_root, cfg_path)
    cfg = radar.parse_cfg(cfg_path)
    if not cfg.get("doppler_res_mps"):
        raise SystemExit(f"profile {cfg_path} not parsed — pass --cfg radar_configs/<profile>.cfg")
    print(f"profile {cfg_path}: bin {cfg['doppler_res_mps']:.3f} m/s, Doppler period {cfg['doppler_period_mps']:.2f} m/s")
    r = estimate(log, cfg, keys["lx"], keys["ly"], keys["mount_yaw_deg"], keys["gyro_bias_dps"])
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
