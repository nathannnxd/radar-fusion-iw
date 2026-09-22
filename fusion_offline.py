"""fusion_offline.py — run fusion over a record_sync folder (rec_*/): video + radar on one shared timeline,
or replay a record_sync run.jsonl.gz (radar + $EGOVEL + camera records) through the real ego-motion code path.

    python fusion_offline.py rec_20260915_180817 [--out fused.mp4] [--meters 8] [--detector yolo-world]
                              [--blind a,b | --miss a,b | --occlude a,b]
    python fusion_offline.py --log rec_*/run.jsonl.gz [--cfg radar_configs/hangar_v9.cfg] [--time-offset -0.12]
                              [--gyro-bias 0.3] [--segment 10,70,straight_1mps] [--static-mask -3,2,3,12]

Simulated camera loss (seconds a..b):
  --blind    the whole frame is blurred, no detections           → expect reason "haze / smoke / defocus"
  --miss     the frame is clean, no detections                   → "detector miss"
  --occlude  a "sheet" is drawn over the REAL person (YOLO on the clean frame); the detector runs on the sheet frame
             → "occluded"; also measures IoU of the virtual box against ground truth
Camera time — camera_times.csv; radar time — radar_times.csv (the byte chunk containing the last byte of the frame).

--log mode (EGO_MOTION.md §6): every radar frame goes through iwr1642_live.Pipeline.process(frame, imu=...) with the
IMU sample interpolated from the logged $EGOVEL lines — fed through ego_velocity.EgoVelocityReader.feed_line /
sample_at when the reader has them (the live code path), else through the local LoggedImu with the same dict keys.
--time-offset is EGO_TIME_OFFSET_S (the IMU time for a radar frame is t_frame + offset; negative: the radar's
timestamp is later than the IMU's — radar lags), --gyro-bias is EGO_GYRO_BIAS_DPS (subtracted from the raw yaw rate).
Prints one compact ego_info line per frame and a summary: state histogram, mean/median/p95 of v per --segment,
and the false-moving rate on points inside the --static-mask rectangle (x1,y1,x2,y2 metres, x right / y forward).
The helpers below (read_log, parse_egovel, LoggedImu, fit_ego_2p, fit_ego_1p_lever) are shared with
tools/calib_time_offset.py and tools/calib_mount.py.
"""
import argparse
import base64
import bisect
import csv
import gzip
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# portable — works regardless of the directory this script is launched from (Windows/Linux/macOS/Pi alike),
# unlike the previous hardcoded Windows path this replaced
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import iwr1642_live as radar
import radar_filter
import ego_velocity
import alerts                                            # state naming only — the replay must label frames exactly as the live alert layer does

F = None                                                 # the fusion module, loaded on demand (see load_fusion)


def load_fusion():
    """fusion-python-3.10.py isn't a valid module name (hyphen + dot), so it can't be `import`ed directly — load it by
    file path instead. Lazy: --log replay and the calibration tools don't need YOLO/torch/the camera model."""
    global F
    if F is None:
        spec = importlib.util.spec_from_file_location("fusion_live", os.path.join(_here, "fusion-python-3.10.py"))
        F = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(F)
    return F


def radar_frames_with_time(rec):
    data = open(os.path.join(rec, "radar.bin"), "rb").read()
    offs, ts = [], []
    with open(os.path.join(rec, "radar_times.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            offs.append(int(row["byte_offset"])); ts.append(float(row["t_s"]))
    offs, ts = np.array(offs), np.array(ts)
    frames, pos = [], 0
    while True:
        i = data.find(radar.MAGIC, pos)
        if i < 0 or len(data) < i + radar.HEADER_LEN:
            break
        total = int.from_bytes(data[i + 12:i + 16], "little")
        if total < radar.HEADER_LEN or total > 65536 or len(data) < i + total:
            pos = i + 8; continue
        fr = radar.parse_frame(data[i:i + total])
        if fr:
            last_byte = i + total - 1
            k = np.searchsorted(offs, last_byte, side="right") - 1        # the chunk containing the last byte of the frame
            fr["t"] = float(ts[min(max(k, 0), len(ts) - 1)])
            frames.append(fr)
        pos = i + total
    return frames


def camera_times(rec):
    with open(os.path.join(rec, "camera_times.csv"), encoding="utf-8") as f:
        return {int(r["frame"]): float(r["t_s"]) for r in csv.DictReader(f)}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------- run.jsonl.gz (record_sync.py) — reading
def read_log(path):
    """record_sync run.jsonl.gz -> {"meta": {...}, "radar": [parsed frames with "t"], "ego": [(t_rx, line)], "cam": [...]}.
    Radar packets are decoded with iwr1642_live.parse_frame — the same parser the live pipeline uses."""
    out = {"meta": {}, "radar": [], "ego": [], "cam": []}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            k = rec.get("kind")
            if k == "radar":
                fr = radar.parse_frame(base64.b64decode(rec["b64"]))
                if fr:
                    fr["t"] = float(rec["t"])
                    out["radar"].append(fr)
            elif k == "ego":
                out["ego"].append((float(rec["t"]), rec["line"]))
            elif k == "cam":
                out["cam"].append(rec)
            elif k == "meta":
                out["meta"] = rec
    out["radar"].sort(key=lambda fr: fr["t"])
    out["ego"].sort(key=lambda e: e[0])
    return out


def parse_egovel(line):
    """$EGOVEL v1 (5 fields) / v2 (10 fields) sentence -> dict, or None if not EGOVEL / bad checksum / wrong field
    count (EGO_MOTION.md §2). Empty v2 fields mean unknown (None). Local fallback for ego_velocity's parser — the
    calibration tools deliberately use this one so a reader change can't silently move a calibration."""
    line = line.strip()
    if not line.startswith("$") or "*" not in line:
        return None
    body, _, cs_hex = line[1:].partition("*")
    try:
        want = int(cs_hex[:2], 16)
    except ValueError:
        return None
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    if cs != want:
        return None
    f = body.split(",")
    if f[0] != "EGOVEL" or len(f) not in (6, 11):
        return None
    num = lambda s: float(s) if s != "" else None
    try:
        d = {"vx": float(f[1]), "vy": float(f[2]), "yaw_rate_dps": float(f[3]), "seq": int(f[4]), "flags": int(f[5]),
             "esp_ms": None, "pitch": None, "roll": None, "acc_fwd": None, "gyro_cal": None}
        if len(f) == 11:
            d.update(esp_ms=num(f[6]), pitch=num(f[7]), roll=num(f[8]), acc_fwd=num(f[9]),
                     gyro_cal=int(f[10]) if f[10] != "" else None)
    except ValueError:
        return None
    d["zupt"] = bool(d["flags"] & 1)
    return d


class LoggedImu:
    """Offline stand-in for EgoVelocityReader.feed_line / sample_at (EGO_MOTION.md §5): keeps every logged sample
    (no 2-s ring buffer — replay may query in any order) and returns the same dict keys the estimator expects:
    yaw_rate (rad/s, + left, bias removed), pitch, roll, acc_fwd, vy_imu, zupt, gyro_ok, age_s."""
    STALE_S = 0.2                                        # gyro_ok requires age <= 0.2 s (EGO_MOTION.md §5)

    def __init__(self, time_offset_s=0.0, gyro_bias_dps=0.0):
        self.time_offset_s, self.gyro_bias_dps = float(time_offset_s), float(gyro_bias_dps)
        self.t, self.s = [], []                            # receive times (sorted) and parsed samples
        self.n_bad = 0

    def feed_line(self, line, t_rx):
        d = parse_egovel(line)
        if d is None:
            self.n_bad += 1
            return False
        i = bisect.bisect_right(self.t, t_rx)
        self.t.insert(i, float(t_rx)); self.s.insert(i, d)
        return True

    def arrays(self):
        """(t_rx, raw yaw_rate_dps) as numpy arrays — for np.interp in the calibration tools."""
        return np.array(self.t, dtype=float), np.array([d["yaw_rate_dps"] for d in self.s], dtype=float)

    def sample_at(self, t):
        tq = t + self.time_offset_s
        empty = {"yaw_rate": 0.0, "pitch": None, "roll": None, "acc_fwd": None, "vy_imu": 0.0, "zupt": False,
                 "gyro_ok": False, "age_s": float("inf")}
        if not self.t:
            return empty
        i = bisect.bisect_right(self.t, tq)                # samples [i-1] <= tq < [i]
        if i == 0:
            a = b = 0; frac = 0.0
        elif i >= len(self.t):
            a = b = len(self.t) - 1; frac = 0.0
        else:
            a, b = i - 1, i
            span = self.t[b] - self.t[a]
            frac = (tq - self.t[a]) / span if span > 1e-6 else 0.0
        sa, sb = self.s[a], self.s[b]
        lerp = lambda k: None if sa[k] is None or sb[k] is None else sa[k] + frac * (sb[k] - sa[k])
        near = sb if frac > 0.5 else sa
        age = abs(tq - self.t[a]) if i > 0 else abs(self.t[0] - tq)   # distance to the last sample at/before tq
        cal = near["gyro_cal"]
        return {"yaw_rate": math.radians(lerp("yaw_rate_dps") - self.gyro_bias_dps),
                "pitch": lerp("pitch"), "roll": lerp("roll"), "acc_fwd": lerp("acc_fwd"), "vy_imu": lerp("vy"),
                "zupt": bool(near["zupt"]), "gyro_ok": (cal is None or cal >= 2) and age <= self.STALE_S,
                "age_s": age}


class ReaderFeed:
    """Adapter over ego_velocity.EgoVelocityReader(port=None) — the live code path (feed_line / sample_at, ring buffer,
    esp_ms clock mapping). The CLI --time-offset / --gyro-bias are passed to the constructor so they REPLACE the
    configs.json values; an older reader without those kwargs gets them applied on top of its configured ones."""

    def __init__(self, time_offset_s=0.0, gyro_bias_dps=0.0):
        self.rd = ego_velocity.EgoVelocityReader(port=None, time_offset_s=time_offset_s, gyro_bias_dps=gyro_bias_dps)
        self.n_bad = 0

    def feed_line(self, line, t_rx):
        ok = self.rd.feed_line(line, t_rx)                  # the reader returns the parsed sample (or None when dropped)
        if not ok:
            self.n_bad += 1
        return bool(ok)

    def sample_at(self, t):
        return dict(self.rd.sample_at(t))        # the reader already applies time_offset_s / gyro_bias_dps


def make_imu_feed(engine="auto", time_offset_s=0.0, gyro_bias_dps=0.0):
    """'reader'/'auto' — the live EgoVelocityReader code path (feed_line/sample_at, ring buffer, esp_ms clock mapping),
    'local' — the log-only LoggedImu interpolator. Returns (feed, engine_name)."""
    if engine in ("auto", "reader"):
        return ReaderFeed(time_offset_s, gyro_bias_dps), "reader"
    return LoggedImu(time_offset_s, gyro_bias_dps), "local"


# ---------------------------------------------------------------- stationary-world fits (stateless, shared by the tools)
def frame_points(frame, cfg):
    """One frame -> (azimuth rad, right +; Doppler m/s, + receding) with the estimator's gating (EGO_MIN_RANGE_M,
    finite SNR), via iwr1642_live.points_to_detections so the geometry is exactly the pipeline's."""
    dets = radar.points_to_detections(frame, cfg, 0.0)
    pts = [(d["azimuth_rad"], d["doppler_mps"]) for d in dets
           if d["range_m"] >= radar.EGO_MIN_RANGE_M and d["snr_db"] == d["snr_db"]]
    if not pts:
        return np.zeros(0), np.zeros(0)
    a = np.array(pts, dtype=float)
    return a[:, 0], a[:, 1]


def inlier_tol(cfg):
    """0.75·bin + EGO_INLIER_TOL_MPS — the estimator's inlier tolerance (EGO_MOTION.md §4.2)."""
    return 0.75 * (cfg.get("doppler_res_mps") or 0.13) + radar.EGO_INLIER_TOL_MPS


def fit_ego_2p(az, dop, tol, period=None, v_ref=0.0):
    """Stateless 2-parameter stationary-world fit — EgoEstimator's model (Kellner et al. 2013) without its median /
    rate-limit / hold: grid search on the period-wrapped inlier count, then Tukey IRLS on the inliers of
    dop = -(vy·cos az + vx·sin az). vx is lateral RIGHT + (sensor frame). Returns a dict or None (no usable fit).
    Kept local on purpose: the calibration tools must measure the raw geometry, not the estimator's filtered output."""
    n = len(az)
    if n < radar.EGO_MIN_POINTS:
        return None
    c, s = np.cos(az), np.sin(az)
    lo, hi = radar.EGO_SEARCH_MPS
    grid = np.arange(lo, hi + 1e-9, 0.05)
    res = dop[None, :] + grid[:, None] * c[None, :]
    if period:
        res -= np.round(res / period) * period
    inl = np.abs(res) < tol
    counts = inl.sum(axis=1)
    best = int(counts.max())
    if best < radar.EGO_MIN_POINTS or best < radar.EGO_MIN_INLIER_FRAC * n:
        return None
    cands = grid[counts >= best - 1]
    v0 = float(cands[np.argmin(np.abs(cands - v_ref))])
    sel = inl[int(np.argmin(np.abs(grid - v0)))]
    vv = dop[sel]
    if period:
        vv = vv - np.round((vv + v0 * c[sel]) / period) * period
    A = np.column_stack([c[sel], s[sel]])
    w = np.ones(len(vv))
    sol, r = np.array([v0, 0.0]), vv + A @ np.array([v0, 0.0])
    for _ in range(3):
        sol, *_ = np.linalg.lstsq(A * w[:, None], -vv * w, rcond=None)
        r = vv + A @ sol
        w = np.clip(1 - (r / tol) ** 2, 0.0, None)
    if not np.all(np.isfinite(sol)):
        return None
    return {"vy": float(sol[0]), "vx": float(sol[1]), "n_inl": int(sel.sum()), "n": n,
            "az_spread_deg": math.degrees(float(az[sel].max() - az[sel].min())),
            "rms": float(np.sqrt(np.mean(r[w > 0] ** 2))) if np.any(w > 0) else float("nan"),
            "ambiguous": bool(period) and float(cands.max() - cands.min()) > 0.6 * period}


def fit_ego_1p_lever(az, dop, omega, lx, ly, tol, period=None, v0=0.0):
    """Lever-arm-compensated single-unknown fit (EGO_MOTION.md §4.4). Sensor at (lx forward, ly left) of the yaw
    centre, turning at omega (rad/s, + left): a stationary point at azimuth θ (right +) shows
        dop = -((v - ω·ly)·cos θ - ω·lx·sin θ)   =>   dop - ω·(lx·sin θ + ly·cos θ) = -v·cos θ
    Cauchy IRLS (w = 1/(1+(r/c)²), c = tol) for v; Doppler unwrapped around the seed v0.
    Returns (v, cost, n_inl) — cost is the mean Cauchy loss c²·ln(1+(r/c)²) over all points, the quantity the time-
    offset grid search minimises (quadratic near the optimum, saturating for movers)."""
    c, s = np.cos(az), np.sin(az)
    dd = dop - omega * (lx * s + ly * c)
    if period:
        dd = dd - np.round((dd + v0 * c) / period) * period
    w = np.ones(len(dd)); v = float(v0); r = dd + v * c
    for _ in range(5):
        den = float(np.sum(w * c * c))
        if den <= 1e-9:
            return float("nan"), float("nan"), 0
        v = -float(np.sum(w * dd * c)) / den
        r = dd + v * c
        w = 1.0 / (1.0 + (r / tol) ** 2)
    cost = float(np.mean(tol * tol * np.log1p((r / tol) ** 2)))
    return v, cost, int(np.sum(np.abs(r) < tol))


# ---------------------------------------------------------------- --log replay
IMU_LEAD_S = 0.5                                         # feed $EGOVEL lines this far ahead of the frame (covers +0.1 s offset + interpolation)


def ego_state(info):
    """State label (EGO_MOTION.md §4.3) — alerts.ego_state, so a replayed frame reads the same as a live one."""
    return alerts.ego_state(info)


def format_ego_line(t, fr, info, imu):
    g = lambda k, d="-": info.get(k, d)
    w = info.get("yaw_rate", imu.get("yaw_rate", 0.0)) or 0.0
    return (f"t={t:8.3f} f={fr['frame']:6d} {ego_state(info):8s} v={float(g('v', 0.0)):+6.2f} raw={float(g('raw', 0.0)):+6.2f} "
            f"w={math.degrees(w):+6.1f}dps n={g('n_inliers', 0)}/{g('n_points', 0)} spr={float(g('az_spread_deg', 0.0)):4.0f} "
            f"amb={int(bool(g('ambiguous', False)))} valid={int(bool(g('valid', False)))} gyro_ok={int(bool(imu.get('gyro_ok', False)))} "
            f"age={imu.get('age_s', float('inf')):.2f} "
            # §4.1: pitch/roll are logged, never a rejection — so a slope-induced bias is distinguishable from a
            # calibration error when the drill numbers are read back
            f"pr={float(g('pitch_deg', 0.0)):+3.0f}/{float(g('roll_deg', 0.0)):+3.0f}{' SLOPE' if g('slope', False) else ''}")


def replay_log(path, cfg=None, cfg_path=None, time_offset_s=0.0, gyro_bias_dps=0.0, imu_engine="auto",
               segments=None, static_mask=None, use_background=True, print_every=1, out=print):
    """Replay a run.jsonl.gz through Pipeline.process with the logged IMU. Returns a summary dict (also printed)."""
    log = read_log(path)
    if cfg is None:
        cfg_path = cfg_path or log["meta"].get("cfg_file") or radar.CFG_FILE
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(_here, cfg_path)
        cfg = radar.parse_cfg(cfg_path)
    frames, ego_lines = log["radar"], log["ego"]
    if not frames:
        raise SystemExit(f"{path}: no radar records")
    feed, engine = make_imu_feed(imu_engine, time_offset_s, gyro_bias_dps)
    n_cam = len(log["cam"]); n_cam_det = sum(1 for c in log["cam"] if c.get("dets"))
    out(f"log {path}: radar {len(frames)} frames {frames[0]['t']:.2f}–{frames[-1]['t']:.2f} s · $EGOVEL {len(ego_lines)} lines "
        f"· camera {n_cam} frames ({n_cam_det} with detections) · tag {log['meta'].get('tag') or '-'}")
    out(f"profile: bin {cfg.get('doppler_res_mps')} m/s, period {cfg.get('doppler_period_mps')} m/s, frame {cfg.get('frame_period_s')} s "
        f"· imu engine {engine} · time offset {time_offset_s:+.3f} s · gyro bias {gyro_bias_dps:+.2f} dps")

    radar.Track._next_id = 1
    pipe = radar.Pipeline(cfg, None, 0.0, use_background=use_background)
    hist = Counter(); rows = []; mask_n = mask_false = 0; per_seg_mask = defaultdict(lambda: [0, 0])
    segs = segments or [(frames[0]["t"], frames[-1]["t"] + 1e-6, "all")]
    ei = 0
    for k, fr in enumerate(frames):
        t = fr["t"]
        while ei < len(ego_lines) and ego_lines[ei][0] <= t + IMU_LEAD_S:     # feed the reader as the live loop would
            feed.feed_line(ego_lines[ei][1], ego_lines[ei][0]); ei += 1
        imu = feed.sample_at(t)
        pipe.ego_yaw_rate_dps = math.degrees(imu.get("yaw_rate") or 0.0)
        res = pipe.process(fr, imu=imu)                  # same entry point as the live loop (EGO_MOTION.md §6)
        info = res["ego"]
        st = ego_state(info)
        hist[st] += 1
        rows.append((t, float(info.get("v", 0.0)), st, bool(imu.get("gyro_ok", False))))
        if static_mask is not None:
            x1, y1, x2, y2 = static_mask
            for d in res["dets"]:
                if x1 <= d["x_m"] <= x2 and y1 <= d["y_m"] <= y2:
                    mask_n += 1; mask_false += (not d["is_static"])
                    for a, b, name in segs:
                        if a <= t < b:
                            per_seg_mask[name][0] += 1; per_seg_mask[name][1] += (not d["is_static"])
        if print_every and k % print_every == 0:
            out(format_ego_line(t, fr, info, imu))

    summary = {"n_frames": len(frames), "n_ego_lines": len(ego_lines), "n_bad_ego_lines": getattr(feed, "n_bad", 0),
               "imu_engine": engine, "state_hist": dict(hist), "segments": {}, "static_mask": None}
    out("\nsummary:")
    out(f"  frames {len(frames)} · states: " + ", ".join(f"{s} {c} ({100 * c / len(frames):.0f} %)" for s, c in hist.most_common()))
    out(f"  $EGOVEL lines {len(ego_lines)} (rejected {summary['n_bad_ego_lines']}) · gyro_ok on {sum(r[3] for r in rows)} frames")
    for a, b, name in segs:
        v = np.array([r[1] for r in rows if a <= r[0] < b])
        if len(v) == 0:
            out(f"  segment {name} [{a:.1f}, {b:.1f}) s: no frames"); continue
        d = {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
             "p95_abs": float(np.percentile(np.abs(v), 95))}
        summary["segments"][name] = d
        line = (f"  segment {name} [{a:.1f}, {b:.1f}) s: {d['n']} frames · v mean {d['mean']:+.3f} median {d['median']:+.3f} "
                f"p95 {d['p95']:+.3f} (|v| p95 {d['p95_abs']:.3f}) m/s")
        if name in per_seg_mask and per_seg_mask[name][0]:
            n0, f0 = per_seg_mask[name]
            line += f" · false-moving in mask {100 * f0 / n0:.1f} % ({f0}/{n0} points)"
        out(line)
    if static_mask is not None:
        rate = mask_false / mask_n if mask_n else float("nan")
        summary["static_mask"] = {"n_points": mask_n, "n_false_moving": mask_false, "rate": rate}
        out(f"  static mask {static_mask}: {mask_n} points, false-moving {100 * rate:.1f} %" if mask_n else
            f"  static mask {static_mask}: no points inside")
    return summary


def parse_segment(s):
    """'a,b' or 'a,b,label' -> (a, b, label)."""
    p = s.split(",")
    if len(p) not in (2, 3):
        raise argparse.ArgumentTypeError("--segment wants a,b[,label]")
    a, b = float(p[0]), float(p[1])
    return a, b, (p[2] if len(p) == 3 else f"{a:g}-{b:g}s")


def main_log(a):
    cfg = radar.parse_cfg(a.cfg) if a.cfg else None
    mask = tuple(float(x) for x in a.static_mask.split(",")) if a.static_mask else None
    if mask is not None and len(mask) != 4:
        raise SystemExit("--static-mask wants x1,y1,x2,y2 (metres; x right, y forward)")
    replay_log(a.log, cfg=cfg, time_offset_s=a.time_offset, gyro_bias_dps=a.gyro_bias, imu_engine=a.imu_engine,
               segments=a.segment or None, static_mask=mask, use_background=not a.no_background,
               print_every=0 if a.quiet else a.print_every)


def main_rec(a):
    F = load_fusion()
    import cv2
    rec = a.rec.rstrip("/\\")
    win = lambda s: tuple(float(x) for x in s.split(",")) if s else None
    blind, miss, occl = win(a.blind), win(a.miss), win(a.occlude)

    meta = json.load(open(os.path.join(rec, "meta.json"), encoding="utf-8"))
    cfg_path = a.cfg or meta.get("cfg_file", radar.CFG_FILE)
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(cfg_path))
    cfg = radar.parse_cfg(cfg_path)
    rc_cm = meta.get("radar_to_camera_cm", {}) or {}
    radar_to_cam = {k: rc_cm.get(k, 0.0) / 100.0 for k in ("right", "up", "forward")}
    rframes = radar_frames_with_time(rec)
    ctimes = camera_times(rec)
    cap = cv2.VideoCapture(os.path.join(rec, "camera.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"radar: {len(rframes)} frames, {rframes[0]['t']:.2f}–{rframes[-1]['t']:.2f} s · camera: {len(ctimes)} frames {w}x{h}"
          f" · radar relative to camera {radar_to_cam} · Doppler period {cfg.get('doppler_period_mps')}")

    radar.Track._next_id = 1
    pipe = radar.Pipeline(cfg, None, 0.0, use_background=True)
    cam = F.CameraModel(w, h, hfov_deg=F.CAM_HFOV_DEG, radar_to_cam=radar_to_cam)
    fus = F.Fusion(cam, csv_path=os.path.join(rec, "fusion.csv"))
    yolo, keep = F.load_detector(a.detector or F.DETECTOR)

    out_path = a.out or os.path.join(rec, "fused.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w + 500, max(h, 540)))
    latest = {"fused": [], "dets": [], "matched": {}, "tracks": [], "rdets": [], "rkinds": []}
    ri, n = 0, 0
    stats = defaultdict(int); reasons = {}; ious = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = ctimes.get(n, n / 30.0); n += 1
        while ri < len(rframes) and rframes[ri]["t"] <= t:                   # radar up to this moment
            fr = rframes[ri]; ri += 1
            out = pipe.process(fr)
            filtered = radar_filter.sync_and_filter(out["tracks"], t - fr["t"])
            fused, dets, matched = fus.on_radar(fr["t"], fr["frame"], filtered)
            latest.update(fused=fused, dets=dets, matched=matched, tracks=filtered, rdets=out["dets"], rkinds=out["kinds"])
            stats["radar_frames"] += 1
            stats["matched_frames"] += any(f["matched_now"] for f in fused)
            stats["confirmed_frames"] += any(f["cam_id"] is not None for f in fused)
        in_blind = bool(blind and blind[0] <= t <= blind[1])
        in_miss = bool(miss and miss[0] <= t <= miss[1])
        in_occl = bool(occl and occl[0] <= t <= occl[1])
        truth_box = None
        if in_occl:
            truth = [d for d in F.yolo_detections(yolo, frame, t, keep) if d.cls == "person"]   # ground truth — before the sheet
            if truth:
                held = [f for f in latest["fused"] if f.get("state") in ("both", "hold") and f.get("bbox")]
                if held:                                       # ground truth — the person closest to the held box
                    hb = held[0]["bbox"]; hcx = (hb[0] + hb[2]) / 2
                    d0 = min(truth, key=lambda d: abs(d.cx - hcx))
                else:
                    d0 = max(truth, key=lambda d: d.bh)
                truth_box = (d0.x1, d0.y1, d0.x2, d0.y2)
                x1, y1, x2, y2 = max(0, d0.x1 - 15), max(0, d0.y1 - 15), min(w, d0.x2 + 15), min(h, d0.y2 + 15)
                sheet = np.full((y2 - y1, x2 - x1, 3), 190, np.uint8)
                frame[y1:y2, x1:x2] = cv2.add(sheet, np.random.default_rng(n).integers(0, 6, sheet.shape, dtype=np.uint8))
        if in_blind:
            frame = cv2.GaussianBlur(frame, (61, 61), 0)
        dets = [] if (in_blind or in_miss) else F.yolo_detections(yolo, frame, t, keep)
        fus.push_camera(t, dets, frame)
        stats["cam_frames"] += 1; stats["cam_person"] += any(d.cls == "person" for d in dets)
        if n % 50 == 0:
            yaw = fus.apply_calibration()
            if yaw is not None and n % 300 == 0:
                print(f"  t={t:5.1f}s  yaw radar↔camera = {yaw:+.1f}° over {len(fus.calib.pairs)} pairs")
        label = "CAMERA BLIND (simulated haze)" if in_blind else "DETECTOR OFF (simulated miss)" if in_miss else \
            "SHEET OVER PERSON (simulated occlusion)" if in_occl else None
        if label:
            cv2.putText(frame, label, (w // 2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        left = F.draw_overlay(frame, latest["fused"], dets, latest["matched"], cam, latest["tracks"])
        holds = [f for f in latest["fused"] if f.get("state") == "hold"]
        stats["hold_frames"] += bool(holds)
        stats["out_of_frame"] += sum(1 for f in latest["fused"] if f.get("state") == "out-of-frame")
        stats["absorbed"] += sum(1 for f in latest["fused"] if f.get("absorbed_by") is not None)
        mode = "blind" if in_blind else "miss" if in_miss else "occlude" if in_occl else "real"
        for f in holds:
            reasons.setdefault(mode, defaultdict(int))[f.get("lost_reason") or "?"] += 1
            if truth_box is not None and f.get("bbox"):
                ious.append(iou(f["bbox"], truth_box))
                cv2.rectangle(left, truth_box[:2], truth_box[2:], (255, 255, 255), 1)   # ground truth — thin white
        cv2.putText(left, f"t={t:5.1f}s  yaw={cam.yaw:+.1f}", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        right = radar.render(latest["rdets"], latest["rkinds"], latest["tracks"], pipe.bg, meters=a.meters,
                             title=f"radar t={t:.1f}s", only_ids=F.interesting_ids(latest["fused"]))
        canvas = np.zeros((max(h, right.shape[0]), w + right.shape[1], 3), dtype=np.uint8)
        canvas[:h, :w] = left; canvas[:right.shape[0], w:] = right
        vw.write(canvas)
    vw.release(); fus.close()
    yaw = fus.calib.yaw()
    print("\nsummary:")
    print(f"  camera frames {stats['cam_frames']}, with a person per YOLO {stats['cam_person']} ({100*stats['cam_person']/max(stats['cam_frames'],1):.0f} %)")
    print(f"  radar frames {stats['radar_frames']}: currently matched {stats['matched_frames']}, with a confirmed object {stats['confirmed_frames']}")
    print(f"  frames where the box was held by radar: {stats['hold_frames']}; duplicates absorbed: {stats['absorbed']}; left the frame (frame-tracks): {stats['out_of_frame']}")
    for mode, cnt in reasons.items():
        print(f"  camera-loss reasons [{mode}]: {dict(cnt)}")
    if ious:
        ious = np.array(ious)
        print(f"  virtual box vs ground truth (occlude): IoU mean {ious.mean():.2f}, median {np.median(ious):.2f}, share IoU<0.5: {(ious<0.5).mean():.0%} (n={len(ious)})")
    print(f"  radar↔camera rotation (calibration): {yaw:+.1f}° over {len(fus.calib.pairs)} pairs" if yaw is not None else "  calibration: too few pairs")
    print(f"  objects in memory at the end: {len(fus.mem)}; {[(m.radar_id, m.cam_id, m.cls, m.hits) for m in fus.mem.values() if m.hits >= F.PAIR_CONFIRM]}")
    print(f"  video: {out_path}; CSV: {os.path.join(rec, 'fusion.csv')}")


def main():
    ap = argparse.ArgumentParser()                       # the module docstring has the full usage (non-cp1251 chars: keep it off --help)
    ap.add_argument("rec", nargs="?", help="record_sync folder (video + radar.bin replay)")
    ap.add_argument("--out", default=None); ap.add_argument("--meters", type=int, default=10)
    ap.add_argument("--blind"); ap.add_argument("--miss"); ap.add_argument("--occlude")
    ap.add_argument("--detector", default=None)
    ap.add_argument("--cfg", default=None, help="radar profile .cfg (default: the one stored in the recording, else configs.json)")
    ap.add_argument("--log", default=None, help="record_sync run.jsonl.gz — ego-motion replay instead of the video fusion")
    ap.add_argument("--time-offset", type=float, default=0.0, help="EGO_TIME_OFFSET_S to apply, s (EGO_MOTION.md §3)")
    ap.add_argument("--gyro-bias", type=float, default=0.0, help="EGO_GYRO_BIAS_DPS to subtract, deg/s")
    ap.add_argument("--imu-engine", choices=("auto", "reader", "local"), default="auto")
    ap.add_argument("--segment", type=parse_segment, action="append", help="a,b[,label] seconds; repeatable")
    ap.add_argument("--static-mask", default=None, help="x1,y1,x2,y2 metres: points inside are known static (false-moving rate)")
    ap.add_argument("--no-background", action="store_true"); ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--quiet", action="store_true", help="summary only")
    a = ap.parse_args()
    if a.log:
        main_log(a)
    elif a.rec:
        main_rec(a)
    else:
        ap.error("give a rec_* folder or --log run.jsonl.gz")


if __name__ == "__main__":
    main()
