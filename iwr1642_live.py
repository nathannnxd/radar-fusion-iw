# =====================================================================
#  IWR1642 → points → (model or rule) → clusters → tracks → /radar/tracks contract
#  Standalone script for the laptop next to the radar. Not for Colab (no COM there).
#  Without a radar: DUMP_FILE = "radar_dump.bin" — replays a recording.
#
#  Single-frame pipeline:
#    bytes → parse_frame → points_to_detections (r, az, v, snr; leakage cutoff; ego compensation)
#          → background.mark (background map, only while the radar is stationary)
#          → predict_points (LightGBM over points, or a rule)
#          → cluster_objects (DBSCAN over x, y, v) → Tracker.step (EKF: x, y, vr) → contract
# =====================================================================
import json
import math
import struct
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import serial
except ImportError:
    serial = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import joblib
except ImportError:
    joblib = None
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

import ego_velocity

with open("configs.json", "r") as file:
    code_config = json.load(file)            # Read/edit configs.json
# ---------------------------------------------------------------- SETTINGS
CLI_PORT = code_config["CLI_PORT"]           # XDS110 Class Application/User UART
DATA_PORT = code_config["DATA_PORT"]         # XDS110 Class Auxiliary Data Port
CFG_FILE = code_config["RADAR_CONFIG"]
MODEL_PATH = "radar_lightgbm_model.pkl"      # our own .pkl; the /content/drive/... path doesn't exist on the laptop
DUMP_FILE = None                             # "radar_dump.bin" — replay a recording without ports
SEND_CFG = True                              # False if the radar is already streaming (e.g. started from Visualizer)
EGO_SPEED_MPS = 0.0                          # carrier speed, m/s; 0 on the bench. Overridden live in main() by
                                             # EGO_SERIAL_PORT below if set — this is only the value used before
                                             # the first reading arrives, or if that port is left empty (disabled)
EGO_SERIAL_PORT = code_config.get("EGO_SERIAL_PORT", "")   # ESP32+IMU port (see ego_velocity.py / EGO_VELOCITY.md); "" — disabled
EGO_BAUD = code_config.get("EGO_BAUD", 115200)
SHOW_WINDOW = True                           # OpenCV window
SHOW_FPS = code_config["SHOW_FPS"]           # 1 — draw the fps counter on the display; 0 — off
DRAW_METERS = 15                             # image radius, m: 10 for a room, 30–50 for a field
LOG_JSONL = True                             # write frames and tracks to frames_<time>.jsonl

MIN_RANGE_M = 0.5                            # closer than this — antenna leakage / housing (in both dumps a point at 0–0.5 m with SNR 27 dB)
STATIC_DOPPLER_MPS = 0.12                    # |Doppler after ego compensation| below this — the point is stationary
USE_BACKGROUND = True                        # background map: learns for the first BACKGROUND_LEARN_S seconds, only while ego ≈ 0
BACKGROUND_LEARN_S = 3.0
BACKGROUND_CELL_M = 0.25
BACKGROUND_MIN_OCCUPANCY = 0.4               # a cell is background if occupied by a static point in ≥40 % of the learning frames
CLUSTER_EPS_M = 0.7                          # cluster radius in (x, y, v·CLUSTER_V_WEIGHT)
CLUSTER_V_WEIGHT = 0.7                       # 1 m/s of speed difference ≈ 0.7 m of distance → different objects
TRACK_CONFIRM_HITS = 3                       # hits needed for a candidate to become a track
TRACK_CONFIRM_WINDOW = 5                     # ...within this many first frames
TRACK_MAX_MISSES = 10                        # frames without a measurement a track survives on prediction (1 s at 10 Hz)

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_LEN = 40
TLV_DETECTED_POINTS, TLV_RANGE_PROFILE, TLV_NOISE_PROFILE, TLV_STATS, TLV_SIDE_INFO, TLV_TEMP = 1, 2, 3, 6, 7, 9

# Features in the exact training order (train DataFrame.drop(columns=['target_kind']))
EXPECTED_FEATURES = [
    "ego_speed_mps", "ego_moving", "range_m", "azimuth_rad", "azimuth_deg",
    "doppler_mps", "snr_db", "x_m", "y_m", "range_bin", "doppler_bin",
    "azimuth_bin", "doppler_aliased",
]
# Model classes are POINT TYPES, not objects. The radar doesn't distinguish person/vehicle.
POINT_CLASS_COLOR = {          # BGR
    "target": (0, 220, 0), "target_micro": (0, 160, 255), "clutter": (128, 128, 128),
    "false_alarm": (60, 60, 60), "ghost": (255, 0, 255), "background": (70, 70, 110),
}
TARGET_CLASSES = {"target", "target_micro"}


# ---------------------------------------------------------------- .cfg → parameters
def parse_cfg(path):
    """Range/speed resolution, max speed, frame period — from .cfg."""
    res = {"range_res_m": None, "doppler_res_mps": None, "num_doppler_bins": None, "num_range_bins": None,
           "max_doppler_mps": None, "doppler_period_mps": None, "frame_period_s": None, "max_range_m": None}
    profile = frame = None
    try:
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if parts and parts[0] == "profileCfg":
                profile = [float(p) for p in parts[1:]]
            elif parts and parts[0] == "frameCfg":
                frame = [float(p) for p in parts[1:]]
    except FileNotFoundError:
        return res
    if profile and frame:
        # profileCfg: id startFreq(GHz) idleTime(us) adcStartTime(us) rampEndTime(us) txOutPower txPhaseShifter
        #             freqSlope(MHz/us) txStartTime(us) numAdcSamples digOutSampleRate(ksps) ...
        start_freq_ghz, idle_us, ramp_end_us = profile[1], profile[2], profile[4]
        freq_slope, num_adc, sample_rate_ksps = profile[7], profile[9], profile[10]
        # frameCfg: chirpStartIdx chirpEndIdx numLoops numFrames framePeriodicity(ms) ...
        n_chirps = int(frame[1] - frame[0] + 1)
        num_loops = int(frame[2])
        c = 299_792_458.0
        bw = freq_slope * 1e12 * (num_adc / (sample_rate_ksps * 1e3))
        res["range_res_m"] = c / (2 * bw)
        res["num_range_bins"] = int(2 ** math.ceil(math.log2(num_adc)))
        res["max_range_m"] = res["range_res_m"] * res["num_range_bins"] * 0.8
        lam = c / (start_freq_ghz * 1e9 + bw / 2)                 # wavelength at the center frequency
        tc = (idle_us + ramp_end_us) * 1e-6 * n_chirps
        res["num_doppler_bins"] = num_loops
        res["doppler_res_mps"] = lam / (2 * num_loops * tc)
        res["max_doppler_mps"] = lam / (4 * tc)
        res["doppler_period_mps"] = 2 * res["max_doppler_mps"]     # speed ambiguity period
        res["frame_period_s"] = frame[4] / 1000.0
    return res


# ---------------------------------------------------------------- TI frame parsing
def parse_frame(packet):
    """One OOB-demo packet (SDK 2.x / 3.x) → dict: points, side (snr, noise), range_profile_db."""
    if len(packet) < HEADER_LEN or packet[:8] != MAGIC:
        return None
    version, total_len, platform, frame_num, cpu_cycles, num_obj, num_tlv, subframe = \
        struct.unpack_from("<8I", packet, 8)
    sdk_major = (version >> 24) & 0xFF
    frame = {"frame": frame_num, "num_obj": num_obj, "sdk_major": sdk_major,
             "points": [], "side": [], "range_profile_db": None}
    off = HEADER_LEN
    for _ in range(num_tlv):
        if off + 8 > len(packet):
            break
        tlv_type, tlv_len = struct.unpack_from("<2I", packet, off)
        body = packet[off + 8: off + 8 + tlv_len]
        off += 8 + tlv_len
        if tlv_type == TLV_DETECTED_POINTS:
            if sdk_major >= 3:                    # 4 floats: x, y, z (m), v (m/s)
                for i in range(len(body) // 16):
                    frame["points"].append(list(struct.unpack_from("<4f", body, i * 16)))
            else:                                 # numObj, xyzQFormat, then rangeIdx dopplerIdx peakVal x y z
                n, qfmt = struct.unpack_from("<2H", body, 0)
                sc = 1.0 / (1 << qfmt)
                for i in range(n):
                    r_idx, d_idx, peak, xi, yi, zi = struct.unpack_from("<HhHhhh", body, 4 + i * 12)
                    frame["points"].append([xi * sc, yi * sc, zi * sc, float(d_idx), float(peak), float(r_idx)])
        elif tlv_type == TLV_SIDE_INFO:
            for i in range(len(body) // 4):
                snr, noise = struct.unpack_from("<2h", body, i * 4)
                frame["side"].append((snr * 0.1, noise * 0.1))       # units of 0.1 dB
        elif tlv_type == TLV_RANGE_PROFILE:
            # uint16 per bin, log2 magnitude in Q9 → dB = val / 512 · 20·log10(2)
            prof = np.frombuffer(body[: (len(body) // 2) * 2], dtype="<u2").astype(np.float32)
            frame["range_profile_db"] = prof * (20 * math.log10(2) / 512)
    return frame


def frames_from_bytes(buffer):
    """Generator of frames from a byte buffer; the return value is the unprocessed remainder."""
    while True:
        idx = buffer.find(MAGIC)
        if idx < 0:
            return buffer[-7:] if len(buffer) > 7 else buffer
        buffer = buffer[idx:]
        if len(buffer) < HEADER_LEN:
            return buffer
        total_len = struct.unpack_from("<I", buffer, 12)[0]
        if total_len < HEADER_LEN or total_len > 65536:
            buffer = buffer[8:]
            continue
        if len(buffer) < total_len:
            return buffer
        yield parse_frame(buffer[:total_len])
        buffer = buffer[total_len:]


def noise_floor(frame, cfg):
    """Noise floor from the Range Profile: near zone (0.5–3 m) and far (median). Dust proxy:
    a plume behind the implement raises the near floor before points appear."""
    prof = frame.get("range_profile_db")
    if prof is None or len(prof) < 16:
        return None
    rr = cfg.get("range_res_m") or 0.044
    lo, hi = int(0.5 / rr), max(int(3.0 / rr), int(0.5 / rr) + 4)
    near = float(np.median(prof[lo:hi])) if hi <= len(prof) else float(np.median(prof[lo:]))
    far = float(np.median(prof[len(prof) // 2:]))
    return {"near_db": near, "far_db": far, "near_excess_db": near - far}


# ---------------------------------------------------------------- points → features
def points_to_detections(frame, cfg, ego_speed):
    """Frame points → list of dicts (train-JSON format + internal fields).

    doppler_mps      — as measured by the radar (in the radar frame; + = receding);
    doppler_rel_mps  — after compensating for carrier motion: ≈ 0 for a stationary object.
                       A stationary object seen from a radar moving forward at speed ego appears
                       with Doppler −ego·cos(azimuth); we subtract it.
    is_static        — |doppler_rel| < STATIC_DOPPLER_MPS.
    """
    dets = []
    rr = cfg.get("range_res_m") or 0.044
    dr = cfg.get("doppler_res_mps") or 0.13
    nd = cfg.get("num_doppler_bins") or 16
    for i, p in enumerate(frame["points"]):
        x, y, z = p[0], p[1], p[2]
        rng = math.sqrt(x * x + y * y + z * z)
        if rng < MIN_RANGE_M:
            continue
        az = math.atan2(x, y)                         # TI: y — forward, x — right
        if frame["sdk_major"] >= 3:
            v = p[3]
            snr = frame["side"][i][0] if i < len(frame["side"]) else float("nan")
            d_bin, r_bin = int(round(v / dr)) + nd // 2, int(round(rng / rr))
        else:
            d_idx, peak, r_idx = p[3], p[4], p[5]
            v = d_idx * dr
            snr = 10 * math.log10(max(peak, 1.0))
            d_bin, r_bin = int(d_idx) + nd // 2, int(r_idx)
        v_rel = v + ego_speed * math.cos(az)
        dets.append({
            "range_m": rng, "azimuth_rad": az, "azimuth_deg": math.degrees(az),
            "doppler_mps": v, "snr_db": snr, "x_m": x, "y_m": y,
            "range_bin": r_bin, "doppler_bin": d_bin,
            "azimuth_bin": int(round((math.degrees(az) + 90) / 180 * 63)),
            "doppler_aliased": abs(v) > (nd / 2) * dr * 0.98,
            "doppler_rel_mps": v_rel, "is_static": abs(v_rel) < STATIC_DOPPLER_MPS,
            "background": False,
        })
    return dets


# ---------------------------------------------------------------- background map
class BackgroundMap:
    """Occupancy grid for a stationary radar. Learns for the first learn_s seconds: a cell where a static
    point was present in ≥ min_occ of frames is background. After that, static points in such cells are
    marked background=True and excluded from clustering. Disabled when ego ≠ 0 (the background moves with the view)."""

    def __init__(self, cell=0.25, learn_s=3.0, min_occ=0.6, x_lim=20.0, y_lim=50.0):
        self.cell, self.learn_s, self.min_occ = cell, learn_s, min_occ
        self.nx, self.ny = int(2 * x_lim / cell), int(y_lim / cell)
        self.counts = np.zeros((self.nx, self.ny), dtype=np.int32)
        self.frames_seen, self.t_start = 0, None
        self.mask = None                                     # None = still learning

    def _cell(self, d):
        ix = int((d["x_m"] + self.nx * self.cell / 2) / self.cell)
        iy = int(d["y_m"] / self.cell)
        return (ix, iy) if 0 <= ix < self.nx and 0 <= iy < self.ny else None

    @property
    def learning(self):
        return self.mask is None

    def mark(self, dets, t, ego_speed):
        if abs(ego_speed) > 0.1:                           # carrier is moving — the map is meaningless
            return dets
        if self.t_start is None:
            self.t_start = t
        if self.learning:
            self.frames_seen += 1
            for d in dets:
                c = self._cell(d)
                if c and d["is_static"]:
                    self.counts[c] += 1
            if t - self.t_start >= self.learn_s and self.frames_seen >= 10:
                core = self.counts >= self.min_occ * self.frames_seen
                # expand to neighboring cells: the reflection jitters by ±1 cell from frame to frame
                mask = core.copy()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        mask |= np.roll(np.roll(core, dx, axis=0), dy, axis=1)
                self.mask = mask
            return dets
        for d in dets:
            c = self._cell(d)
            d["background"] = bool(c and d["is_static"] and self.mask[c])
        return dets

    @property
    def n_cells(self):
        return int(self.mask.sum()) if self.mask is not None else 0


# ---------------------------------------------------------------- point classification
def predict_points(model, dets, ego_speed, ego_moving):
    """Point type from the model (or a rule) + confidence. Background → 'background' overrides any answer."""
    if not dets:
        return [], []
    if model is None:
        kinds = ["target" if d["snr_db"] >= 15 else "false_alarm" for d in dets]
        conf = [50.0] * len(dets)
    else:
        rows = [{"ego_speed_mps": float(ego_speed), "ego_moving": int(ego_moving),
                 **{k: d[k] for k in EXPECTED_FEATURES if k in d},
                 "doppler_aliased": int(d["doppler_aliased"])} for d in dets]
        df = pd.DataFrame(rows)[EXPECTED_FEATURES]
        kinds = [str(k) for k in model.predict(df)]
        conf = list(model.predict_proba(df).max(axis=1) * 100)
    kinds = ["background" if d["background"] else k for d, k in zip(dets, kinds)]
    return kinds, conf


# ---------------------------------------------------------------- Doppler ambiguity
DOPPLER_PERIOD = None      # m/s; set by the Pipeline from .cfg. None — wraparound not applied.


def doppler_diff(a, b):
    """Speed difference accounting for wraparound: +0.85 and −0.97 m/s at a period of 1.95 is 0.13 m/s, not 1.82."""
    d = a - b
    if DOPPLER_PERIOD:
        d -= round(d / DOPPLER_PERIOD) * DOPPLER_PERIOD
    return d


# ---------------------------------------------------------------- clustering
def cluster_objects(dets, kinds, confs, eps_m=CLUSTER_EPS_M, v_weight=CLUSTER_V_WEIGHT, min_pts=1):
    """DBSCAN over (x, y, speed): a person a meter from a pole won't merge with it if the speeds differ.
    Speed is encoded as a point on a circle of the ambiguity period, so an object at the ±v_max boundary
    doesn't split into "approaching" and "receding" clusters."""
    idx = [i for i, k in enumerate(kinds) if k in TARGET_CLASSES]
    if not idx:
        return []
    if DOPPLER_PERIOD:
        # circle diameter = w·(period/2): opposite speeds (max difference in magnitude) are as far apart
        # as they were on a line before, while +v_max and −v_max (neighbors across the wraparound) are close
        R = v_weight * DOPPLER_PERIOD / 4
        feats = np.array([[dets[i]["x_m"], dets[i]["y_m"],
                           R * math.cos(2 * math.pi * dets[i]["doppler_mps"] / DOPPLER_PERIOD),
                           R * math.sin(2 * math.pi * dets[i]["doppler_mps"] / DOPPLER_PERIOD)] for i in idx])
    else:
        feats = np.array([[dets[i]["x_m"], dets[i]["y_m"], dets[i]["doppler_mps"] * v_weight] for i in idx])
    labels = DBSCAN(eps=eps_m, min_samples=min_pts).fit_predict(feats) if DBSCAN is not None \
        else np.zeros(len(idx), dtype=int)
    objs = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        mem = [idx[j] for j in range(len(idx)) if labels[j] == lab]
        # weights by SNR; a point whose side-info TLV entry is missing (snr_db = NaN, see
        # points_to_detections) gets a neutral weight instead of poisoning the weighted centroid
        w = np.array([max(dets[i]["snr_db"], 1.0) if math.isfinite(dets[i]["snr_db"]) else 1.0 for i in mem])
        cx = float(np.average([dets[i]["x_m"] for i in mem], weights=w))
        cy = float(np.average([dets[i]["y_m"] for i in mem], weights=w))
        votes = defaultdict(float)
        for i in mem:
            votes[kinds[i]] += confs[i]
        objs.append({
            "x_m": cx, "y_m": cy, "range_m": math.hypot(cx, cy),
            "azimuth_deg": math.degrees(math.atan2(cx, cy)),
            "doppler_mps": float(np.average([dets[i]["doppler_mps"] for i in mem], weights=w)),
            "snr_db": float(np.nanmax([dets[i]["snr_db"] for i in mem])),
            "range_min_m": float(min(dets[i]["range_m"] for i in mem)),   # nearest point of the object — for stopping
            "n_points": len(mem), "votes": dict(votes),
        })
    return sorted(objs, key=lambda o: o["range_m"])


# ---------------------------------------------------------------- tracks (EKF: measurement x, y, vr)
class Track:
    """State [x, y, vx, vy] in the radar frame. Measurement — (x, y, radial speed).
    Radial speed comes straight from the radar with ~0.1 m/s accuracy, so the track's speed
    is known from the first or second frame, rather than "computed from the position difference"."""
    _next_id = 1
    R = np.diag([0.2, 0.2, 0.15]) ** 2                     # measurement noise: m, m, m/s

    def __init__(self, obj, t):
        self.id = Track._next_id; Track._next_id += 1
        r = max(obj["range_m"], 1e-3)
        vr = obj["doppler_mps"]
        # initial speed — along the beam, from Doppler
        self.x = np.array([obj["x_m"], obj["y_m"], vr * obj["x_m"] / r, vr * obj["y_m"] / r])
        self.P = np.diag([0.3, 0.3, 0.8, 0.8]) ** 2
        self.hits, self.misses, self.age = 1, 0, 1
        self.confirmed = False
        self.history = [(self.x[0], self.x[1])]
        self.doppler, self.snr, self.n_points = vr, obj["snr_db"], obj["n_points"]
        self.near_offset = max(0.0, obj["range_m"] - obj.get("range_min_m", obj["range_m"]))  # center − nearest point
        self.votes = defaultdict(float, obj.get("votes", {}))
        self.t_created, self.t_updated = t, t
        self.range_hist = [(t, obj["range_m"])]              # for resolving Doppler ambiguity from positions

    @staticmethod
    def step_state(x, P, dt, a=2.5):
        """Constant-acceleration EKF predict, out-of-place. a: process noise, acceleration ~2.5 m/s²
        (maneuvers, turns). Reused by predict() (mutating) and by radar_filter.py (non-mutating,
        to extrapolate a track to an arbitrary sync time without disturbing the live tracker)."""
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        return F @ x, F @ P @ F.T + G @ G.T * a * a

    def predict(self, dt):
        self.x, self.P = self.step_state(self.x, self.P, dt)
        self.age += 1

    def _h(self, x):
        r = max(math.hypot(x[0], x[1]), 1e-3)
        vr = (x[0] * x[2] + x[1] * x[3]) / r
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [x[2] / r - x[0] * vr / r ** 2, x[3] / r - x[1] * vr / r ** 2, x[0] / r, x[1] / r]])
        return np.array([x[0], x[1], vr]), H

    def innovation(self, obj, R=None):
        z = np.array([obj["x_m"], obj["y_m"], obj["doppler_mps"]])
        hx, H = self._h(self.x)
        S = H @ self.P @ H.T + (self.R if R is None else R)
        return z - hx, S, H

    def gate_distance(self, obj):
        """Matching — by position (2D Mahalanobis). Doppler is excluded from the gate: otherwise an object
        that abruptly changes direction (a person turning around) would fall out of the track."""
        d, S, _ = self.innovation(obj)
        d2, S2 = d[:2], S[:2, :2]
        return float(math.sqrt(d2 @ np.linalg.solve(S2, d2)))

    def range_rate_from_positions(self):
        """Rate of change of range from position history over ~0.6 s (least squares). Unambiguous — unlike Doppler."""
        pts = [(t_, r_) for t_, r_ in self.range_hist if self.range_hist[-1][0] - t_ <= 0.6]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.25:
            return None
        T = np.array([p[0] for p in pts]); Rg = np.array([p[1] for p in pts])
        return float(np.polyfit(T - T[0], Rg, 1)[0])

    def unwrap_doppler(self, z_v):
        """Ambiguity resolution: the hypothesis z_v + k·period closest to the position-derived speed (or to
        the prediction if there's too little history). Returns (speed, "uncertain" flag)."""
        if not DOPPLER_PERIOD:
            return z_v, False
        ref = self.range_rate_from_positions()
        if ref is None:
            ref = self.radial_mps
        cands = sorted(((abs(z_v + k * DOPPLER_PERIOD - ref), z_v + k * DOPPLER_PERIOD) for k in (-1, 0, 1)))
        ambiguous = (cands[1][0] - cands[0][0]) < 0.4     # two hypotheses are nearly equally likely
        return cands[0][1], ambiguous

    def update(self, obj, t):
        z_v, ambiguous = self.unwrap_doppler(obj["doppler_mps"])
        obj = {**obj, "doppler_mps": z_v}
        R = self.R.copy()
        if ambiguous:
            R[2, 2] *= 16                                  # an uncertain Doppler barely moves the speed estimate
        d, S, H = self.innovation(obj, R)
        # Doppler robustness: a radial-speed residual over 3σ → a turn/maneuver or a foreign point,
        # so we update with inflated speed noise instead of discarding the measurement
        if abs(d[2]) > 3 * math.sqrt(S[2, 2]):
            R[2, 2] = (abs(d[2]) / 2) ** 2 + R[2, 2]
            d, S, H = self.innovation(obj, R)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ d
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1; self.misses = 0
        self.range_hist.append((t, obj["range_m"])); self.range_hist = self.range_hist[-12:]
        self.doppler, self.snr, self.n_points = obj["doppler_mps"], obj["snr_db"], obj["n_points"]
        off = max(0.0, obj["range_m"] - obj.get("range_min_m", obj["range_m"]))
        self.near_offset = 0.7 * self.near_offset + 0.3 * off                # smoothed: the cluster "breathes"
        for k, v in obj.get("votes", {}).items():
            self.votes[k] += v
        self.history.append((self.x[0], self.x[1])); self.history = self.history[-30:]
        self.t_updated = t

    def miss(self):
        self.misses += 1

    range_m = property(lambda s: math.hypot(s.x[0], s.x[1]))
    azimuth_deg = property(lambda s: math.degrees(math.atan2(s.x[0], s.x[1])))
    speed_mps = property(lambda s: math.hypot(s.x[2], s.x[3]))
    radial_mps = property(lambda s: (s.x[0] * s.x[2] + s.x[1] * s.x[3]) / max(s.range_m, 1e-3))
    range_near_m = property(lambda s: max(0.0, s.range_m - s.near_offset))   # to the nearest point of the object
    sigma_xy_m = property(lambda s: float(math.sqrt(max(s.P[0, 0] + s.P[1, 1], 0.0))))
    coasting = property(lambda s: s.misses > 0)

    def ground_velocity(self, ego_speed_mps):
        """(vx, vy) in a stationary ground frame instead of the platform's own moving frame — i.e. what
        this object's velocity would read on a non-moving radar. x/vx, y/vy, radial_mps and speed_mps are
        deliberately left as platform-relative everywhere else (that's the correct frame for collision/
        closing-speed judgments — see EGO_VELOCITY.md); this is an additional, separate view for telling
        "this object is actually moving" from "it only looks like it's moving because the platform is".

        ego_speed_mps is assumed purely forward (along +y, the platform's own heading) — no lateral
        term, matching the same assumption points_to_detections() already makes for Doppler ego
        compensation. A track's own vx is therefore unaffected; only vy shifts."""
        return float(self.x[2]), float(self.x[3]) + ego_speed_mps

    def ground_speed_mps(self, ego_speed_mps):
        gvx, gvy = self.ground_velocity(ego_speed_mps)
        return math.hypot(gvx, gvy)

    @property
    def kind(self):
        """Track class — votes from points accumulated over its lifetime, not the last frame's answer."""
        return max(self.votes, key=self.votes.get) if self.votes else "unknown"

    @property
    def kind_conf(self):
        tot = sum(self.votes.values())
        return 100 * self.votes[self.kind] / tot if tot else 0.0

    def contract(self):
        """The /radar/tracks contract record — the same thing the simulator and the tractor will publish."""
        return {"id": self.id, "range_m": round(self.range_m, 3), "range_near_m": round(self.range_near_m, 3),
                "azimuth_deg": round(self.azimuth_deg, 2),
                "x_m": round(float(self.x[0]), 3), "y_m": round(float(self.x[1]), 3),
                "vx_mps": round(float(self.x[2]), 3), "vy_mps": round(float(self.x[3]), 3),
                "radial_mps": round(self.radial_mps, 3), "doppler_mps": round(self.doppler, 3),
                "snr_db": round(self.snr, 1), "n_points": self.n_points,
                "kind": self.kind, "kind_conf": round(self.kind_conf, 1),
                "age_frames": self.age, "hits": self.hits, "misses": self.misses,
                "coasting": self.coasting, "sigma_xy_m": round(float(math.sqrt(self.P[0, 0] + self.P[1, 1])), 3)}


class Tracker:
    def __init__(self, confirm_hits=TRACK_CONFIRM_HITS, confirm_window=TRACK_CONFIRM_WINDOW,
                 max_misses=TRACK_MAX_MISSES, gate=3.5, merge_m=0.5):
        self.confirm_hits, self.confirm_window, self.max_misses = confirm_hits, confirm_window, max_misses
        self.gate, self.merge_m = gate, merge_m
        self.tracks = []

    def step(self, objs, t, dt):
        for tr in self.tracks:
            tr.predict(dt)
        cand = sorted((tr.gate_distance(o), i, j) for i, tr in enumerate(self.tracks)
                      for j, o in enumerate(objs) if tr.gate_distance(o) < self.gate)
        used_t, used_o = set(), set()
        for _, i, j in cand:
            if i in used_t or j in used_o:
                continue
            self.tracks[i].update(objs[j], t); used_t.add(i); used_o.add(j)
        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                tr.miss()
        for j, o in enumerate(objs):
            if j not in used_o:
                self.tracks.append(Track(o, t))
        alive = []
        for tr in self.tracks:
            if not tr.confirmed and tr.hits >= self.confirm_hits:
                tr.confirmed = True
            dead = tr.misses > self.max_misses or \
                (not tr.confirmed and tr.age > self.confirm_window and tr.hits < self.confirm_hits)
            if not dead:
                alive.append(tr)
        alive.sort(key=lambda tr: -tr.hits)                 # duplicates: keep the track with more history
        kept = []
        for tr in alive:
            if any(math.hypot(tr.x[0] - k.x[0], tr.x[1] - k.x[1]) < self.merge_m
                   and abs(doppler_diff(tr.radial_mps, k.radial_mps)) < 0.5 for k in kept):
                continue
            kept.append(tr)
        self.tracks = kept
        return [tr for tr in self.tracks if tr.confirmed]


# ---------------------------------------------------------------- input-output
def send_config(cli_port, cfg_path):
    # NOTE: keep the returned CLI port OPEN while reading the data port. On Linux the XDS110 resets the
    # data port to 115200 the moment the CLI port is closed (Windows doesn't care) -> garbage instead of frames.
    ser = serial.Serial(cli_port, 115200, timeout=1)
    with open(cfg_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("%"):
                ser.write((line + "\n").encode())
                time.sleep(0.05)
    print("✓ Config sent to", cli_port)
    return ser


def byte_source():
    if DUMP_FILE:
        data = open(DUMP_FILE, "rb").read()
        print(f"Reading dump {DUMP_FILE}: {len(data)} bytes, {data.count(MAGIC)} frames")
        for i in range(0, len(data), 4096):
            yield data[i:i + 4096]
            time.sleep(0.02)
        return
    if serial is None:
        raise SystemExit("pip install pyserial")
    cli_ser = None
    try:
        if SEND_CFG:
            cli_ser = send_config(CLI_PORT, CFG_FILE)
        else:
            cli_ser = serial.Serial(CLI_PORT, 115200, timeout=1)   # hold the CLI port open anyway (see send_config)
    except Exception as e:
        print(f"⚠️ Config not sent ({e}). Fine if the radar is already streaming.")
    try:
        with serial.Serial(DATA_PORT, 921600, timeout=0.01) as ser:
            print("=== Listening on", DATA_PORT, "· Ctrl+C or q to exit ===")
            while True:
                chunk = ser.read(max(1, ser.in_waiting))     # yield whatever arrived, don't wait for 4096 bytes: better timing
                if chunk:
                    yield chunk
    finally:
        if cli_ser is not None:
            cli_ser.close()


class FpsMeter:
    """EMA of inter-call timing, ticked once per frame at each display's own natural rate."""
    def __init__(self, alpha=0.2):
        self.alpha, self.fps, self._last = alpha, 0.0, None

    def tick(self, now=None):
        now = time.time() if now is None else now
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else self.fps + self.alpha * (inst - self.fps)
        self._last = now
        return self.fps


def render(dets, kinds, tracks, bg, meters=DRAW_METERS, title="", only_ids=None, fps=None):
    """Top-down view frame (BGR 500×540): points by type, tracks with ID, trail, and speed arrow.
    Used by both the live window and video recording (RECORD_VIDEO)."""
    meters = max(1, int(round(meters)))
    W, H0 = 500, 540
    img = np.zeros((H0, W, 3), dtype=np.uint8)
    ox, oy = W // 2, H0 - 20                                   # radar — bottom center
    s = (H0 - 60) / meters
    step = 5 if meters > 12 else 2 if meters > 6 else 1
    for r in range(step, meters + 1, step):
        cv2.circle(img, (ox, oy), int(r * s), (45, 45, 45), 1)
        cv2.putText(img, f"{r}m", (ox + 5, oy - int(r * s) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (95, 95, 95), 1)
    for ang in (-60, -30, 30, 60):                             # azimuth rays
        ex, ey = ox + int(meters * s * math.sin(math.radians(ang))), oy - int(meters * s * math.cos(math.radians(ang)))
        cv2.line(img, (ox, oy), (ex, ey), (35, 35, 35), 1)
    cv2.line(img, (ox, oy), (ox, oy - int(meters * s)), (55, 55, 55), 1)
    for d, k in zip(dets, kinds):
        if only_ids is not None and (k not in TARGET_CLASSES or d.get("is_static", False)):
            continue                                        # "interesting only" mode: skip background and static points
        px, py = int(ox + d["x_m"] * s), int(oy - d["y_m"] * s)
        if 0 <= px < W and 0 <= py < H0:
            cv2.circle(img, (px, py), 3, POINT_CLASS_COLOR.get(k, (255, 255, 255)), -1)
    for tr in tracks:
        if only_ids is not None and tr.id not in only_ids:
            continue
        color = (0, 255, 255) if not tr.coasting else (0, 140, 200)
        pts = [(int(ox + x * s), int(oy - y * s)) for x, y in tr.history]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(img, p0, p1, color, 1)
        px, py = pts[-1]
        rad = int(max(10, 6 + 2 * tr.n_points))                 # ring size — by point count
        cv2.circle(img, (px, py), rad, color, 2)
        vx, vy = tr.x[2], tr.x[3]
        if math.hypot(vx, vy) > 0.15:
            cv2.arrowedLine(img, (px, py), (int(px + vx * s), int(py - vy * s)), color, 2, tipLength=0.3)
        tag = f"#{tr.id} {tr.range_m:.1f}m {tr.radial_mps:+.1f}m/s"
        if tr.kind != "unknown":
            tag = f"#{tr.id} {tr.kind[:7]} {tr.range_m:.1f}m {tr.radial_mps:+.1f}m/s"
        if tr.coasting:
            tag += f" ~{tr.misses}"
        cv2.putText(img, tag, (px + rad + 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    status = "background: learning" if (bg is not None and bg.learning) else \
        (f"background: {bg.n_cells} cells" if bg is not None else "background: off")
    cv2.putText(img, status, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    if fps is not None:
        cv2.putText(img, f"{fps:4.1f} fps", (W - 90, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    if title:
        cv2.putText(img, title, (8, H0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    # legend
    y = 30
    for k, c in (("target", POINT_CLASS_COLOR["target"]), ("target_micro", POINT_CLASS_COLOR["target_micro"]),
                 ("clutter", POINT_CLASS_COLOR["clutter"]), ("background", POINT_CLASS_COLOR["background"])):
        cv2.circle(img, (12, y), 3, c, -1); cv2.putText(img, k, (20, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1); y += 14
    cv2.circle(img, (12, y), 6, (0, 255, 255), 1); cv2.putText(img, "track (~N = coasting N frames)", (20, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1)
    return img


def draw(dets, kinds, tracks, bg, meters=DRAW_METERS, title="", fps=None):
    if cv2 is None or not SHOW_WINDOW:
        return False
    cv2.imshow("IWR1642 - points / tracks", render(dets, kinds, tracks, bg, meters, title, fps=fps))
    return (cv2.waitKey(1) & 0xFF) == ord("q")


# ---------------------------------------------------------------- one frame (also used in tests)
class Pipeline:
    def __init__(self, cfg, model=None, ego_speed=0.0, use_background=USE_BACKGROUND):
        self.cfg, self.model, self.ego = cfg, model, ego_speed
        self.period = cfg.get("frame_period_s") or 0.1
        global DOPPLER_PERIOD
        DOPPLER_PERIOD = cfg.get("doppler_period_mps")
        self.bg = BackgroundMap(BACKGROUND_CELL_M, BACKGROUND_LEARN_S, BACKGROUND_MIN_OCCUPANCY) \
            if use_background else None
        self.tracker = Tracker()
        self.last_frame_num, self.t = None, 0.0

    def process(self, frame):
        # time — from the radar frame number: dropped USB packets don't compress the tracker's time
        if self.last_frame_num is not None:
            gap = frame["frame"] - self.last_frame_num
            dt = self.period * (gap if 0 < gap < 100 else 1)
        else:
            dt = self.period
        self.last_frame_num = frame["frame"]
        self.t += dt

        dets = points_to_detections(frame, self.cfg, self.ego)
        if self.bg is not None:
            dets = self.bg.mark(dets, self.t, self.ego)
        kinds, confs = predict_points(self.model, dets, self.ego, abs(self.ego) > 0.1)
        objs = cluster_objects(dets, kinds, confs)
        tracks = self.tracker.step(objs, self.t, dt)
        noise = noise_floor(frame, self.cfg)
        return {"t": self.t, "dt": dt, "dets": dets, "kinds": kinds, "confs": confs,
                "objs": objs, "tracks": tracks, "noise": noise, "ego_speed_mps": self.ego}


def main():
    model = None
    if joblib is not None:
        try:
            model = joblib.load(MODEL_PATH)          # the team's own .pkl; never load someone else's pickle
            print("✓ LightGBM model loaded:", MODEL_PATH)
        except Exception as e:
            print(f"⚠️ Model not loaded ({e}) — falling back to the SNR ≥ 15 dB rule")
    cfg = parse_cfg(CFG_FILE)
    print("From .cfg:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cfg.items()})
    if cfg.get("max_doppler_mps") and cfg["max_doppler_mps"] < 3:
        print(f"⚠️ Max unambiguous speed {cfg['max_doppler_mps']:.2f} m/s — too low for the tractor, "
              f"need a profile with ≥ 5 m/s")

    pipe = Pipeline(cfg, model, EGO_SPEED_MPS)
    ego_reader = ego_velocity.EgoVelocityReader(EGO_SERIAL_PORT, EGO_BAUD)   # no-ops to speed_mps=0.0 if EGO_SERIAL_PORT is ""
    log = open(f"frames_{datetime.now():%Y%m%d_%H%M%S}.jsonl", "w", encoding="utf-8") if LOG_JSONL else None
    fps_meter = FpsMeter()
    buffer, n, t0 = b"", 0, time.time()
    try:
        for chunk in byte_source():
            buffer += chunk
            gen = frames_from_bytes(buffer)
            while True:
                try:
                    frame = next(gen)
                except StopIteration as stop:
                    buffer = stop.value if stop.value is not None else b""
                    break
                if frame is None:
                    continue
                n += 1
                pipe.ego = ego_reader.speed_mps            # live update — see ego_velocity.py
                out = pipe.process(frame)
                fps_meter.tick()
                if n % 5 == 0:
                    fps = n / max(time.time() - t0, 1e-6)
                    counts = pd.Series(out["kinds"]).value_counts().to_dict() if out["kinds"] else {}
                    nz = f" · near noise +{out['noise']['near_excess_db']:.1f} dB" if out["noise"] else ""
                    print(f"\nframe {frame['frame']} · {fps:.1f} fps · points {len(out['dets'])} · {counts}{nz}")
                    for tr in out["tracks"]:
                        st = "predicted" if tr.coasting else "measured"
                        print(f"  #{tr.id:<3} {tr.kind:12s} {tr.range_m:5.2f} m  {tr.azimuth_deg:+4.0f}°  "
                              f"{tr.radial_mps:+5.2f} m/s  |v|={tr.speed_mps:4.2f}  hits {tr.hits:3d}  {st}")
                if log:
                    log.write(json.dumps({
                        "t": out["t"], "frame": frame["frame"], "ego_speed_mps": out["ego_speed_mps"],
                        "ego_moving": abs(out["ego_speed_mps"]) > 0.1, "noise": out["noise"],
                        "detections": [{**{k: v for k, v in d.items()}, "kind_pred": k_, "conf": round(float(c), 1)}
                                       for d, k_, c in zip(out["dets"], out["kinds"], out["confs"])],
                        "radar_tracks": [tr.contract() for tr in out["tracks"]],
                    }, ensure_ascii=False, default=float) + "\n")
                if draw(out["dets"], out["kinds"], out["tracks"], pipe.bg,
                        title=f"frame {frame['frame']}  t={out['t']:.1f}s", fps=fps_meter.fps if SHOW_FPS else None):
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\n⏹ Stopped.")
    finally:
        ego_reader.close()
        if log:
            log.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
        print(f"Frames processed: {n}")


if __name__ == "__main__":
    main()
