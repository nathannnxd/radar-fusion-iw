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
from collections import defaultdict, deque
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
STATIC_DOPPLER_MPS = 0.12                    # |Doppler after ego compensation| below this — the point is stationary. Fallback
                                             # only: Pipeline derives max(0.2, 1.2·bin) from the profile (EGO_MOTION.md §5)
USE_BACKGROUND = True                        # short-memory ego buffer: persistence evidence for static returns (EGO_MOTION.md §4)
BACKGROUND_FORGET_S = 5.0                    # cells decay with this time constant while moving (EGO_MOTION.md §4, last paragraph)
BACKGROUND_CELL_M = 0.25
BACKGROUND_PERSIST_FRAMES = 3                # a static return seen this many frames in one cell is confirmed (EGO_MOTION.md §5)
BACKGROUND_COUNT_CAP = 3 * BACKGROUND_PERSIST_FRAMES   # a cell's evidence never banks more than this: nothing decays while
                                             # STANDING (§4), so a pre-run standstill would otherwise carry four-digit
                                             # counts into the drive and bypass the persistence gate for ~30 s (§5)
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
           "max_doppler_mps": None, "doppler_period_mps": None, "frame_period_s": None, "max_range_m": None,
           "num_tx": None}
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
        lam = c / (start_freq_ghz * 1e9)                          # wavelength at the START frequency — the firmware's
                                                                  # convention (EGO_MOTION.md §4), not the centre frequency
        tc = (idle_us + ramp_end_us) * 1e-6 * n_chirps            # TDM-MIMO: one Doppler sample per num_tx chirps
        res["num_tx"] = n_chirps
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
def points_to_detections(frame, cfg, ego_speed, static_tol=None):
    """Frame points → list of dicts (train-JSON format + internal fields).

    doppler_mps      — as measured by the radar (in the radar frame; + = receding);
    doppler_rel_mps  — after compensating for carrier motion: ≈ 0 for a stationary object.
                       A stationary object seen from a radar moving forward at speed ego appears
                       with Doppler −ego·cos(azimuth); we subtract it.
    is_static        — |doppler_rel| < static_tol (STATIC_DOPPLER_MPS when not given; the Pipeline passes
                       the profile-derived max(0.2, 1.2·bin), EGO_MOTION.md §5).
    persist          — frames a static return was seen in this cell (filled by BackgroundMap.mark, 0 here).
    """
    dets = []
    static_tol = STATIC_DOPPLER_MPS if static_tol is None else static_tol
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
            "doppler_rel_mps": v_rel, "is_static": abs(v_rel) < static_tol,
            "background": False, "persist": 0,
        })
    return dets


# ---------------------------------------------------------------- ego-motion from the point cloud
EGO_MODE = code_config.get("EGO_MODE", "radar")   # "radar": carrier speed from the frame's stationary targets, every frame
                                                  # "fixed": EGO_SPEED_MPS (bench)
                                                  # "external": only what Pipeline.process(frame, ego_speed=...) is given
EGO_MIN_POINTS = 6                            # fewer inliers — no estimate this frame (EGO_MOTION.md §4.5)
EGO_MIN_RANGE_M = 1.0                         # closer points are antenna leakage / the carrier itself (EGO_MOTION.md §4.1)
EGO_INLIER_TOL_MPS = 0.06                     # inlier tolerance = 0.75 x Doppler bin + this (EGO_MOTION.md §4.2): a stationary
                                              # target's Doppler is quantised to +-bin/2; anything looser lets slow walkers vote
EGO_MIN_INLIER_FRAC = 0.4                     # the stationary-world model must explain this share of the points (§4.5)
EGO_MEDIAN_FRAMES = 3                         # reported speed = median of the last N accepted fits (kills 1-frame flips)
EGO_MAX_ACCEL_MPS2 = 2.5                      # plausibility gate |v - v_prev| <= a_max*dt + 0.5*bin (EGO_MOTION.md §4.5)
EGO_HOLD_S = 1.0                              # keep the last valid estimate this long without evidence, then 0 (§4.3 UNKNOWN)
EGO_BRIDGE_S = 2.0                            # the IMU's integrated velocity bridges a radar dropout at most this long (§1)
EGO_IMU_MAX_AGE_S = 0.5                       # an IMU sample older than this bridges nothing and vouches for nothing:
                                              # sample_at() returns vy_imu 0.0 / age inf with no link (EGO_MOTION.md §5)
EGO_STATIONARY_MPS = 0.15                     # |v| below this — `moving` is False (legacy consumers of ego_info)
EGO_SEARCH_MPS = (-3.0, 15.0)                 # speeds considered (reverse .. fast forward), grid step EGO_GRID_STEP_MPS
EGO_GRID_STEP_MPS = 0.05
EGO_VX_SEARCH_MPS = (-2.0, 2.0)               # lateral speeds the 2-parameter (no-gyro) grid search covers: a 1 rad/s yaw
EGO_VX_STEP_MPS = 0.5                         # at a 2 m lever arm plus side-slip; IRLS refines from there (§4.4)
EGO_MIN_AZ_SPREAD_DEG = 25.0                  # inlier azimuth spread (max - min) below this — 'ambiguous', hold (§4.5)
EGO_CREEP_BINS = 5                            # CREEPING = |v| in 1..5 bins, MOVING above (EGO_MOTION.md §4.3)
EGO_STANDING_GYRO_DPS = 1.5                   # STANDING needs |gyro| below this when the gyro is trusted (§4.3)
EGO_STATE_HYST_FRAMES = 2                     # a new state must be seen this many consecutive frames (§4.3 hysteresis)
EGO_XCHECK_MIN_INLIERS = 40                   # radar-yaw cross-check only on rich, wide frames ... (EGO_MOTION.md §4.6)
EGO_XCHECK_MIN_SPREAD_DEG = 60.0              # ... with this inlier azimuth spread (§4.6)
EGO_XCHECK_TOL_DPS = 3.0                      # |radar yaw - gyro yaw| above this ... (§4.6)
EGO_XCHECK_FRAMES = 15                        # ... for this many consecutive frames -> gyro_ok False (§4.6); same to recover
LEVER_ARM_X_M = float(code_config.get("LEVER_ARM_X_M", 2.0))    # radar forward of the rear axle (yaw centre), m (EGO_MOTION.md §3)
LEVER_ARM_Y_M = float(code_config.get("LEVER_ARM_Y_M", 0.0))    # radar left (+) of the vehicle centre line, m (§3)
MOUNT_YAW_DEG = float(code_config.get("MOUNT_YAW_DEG", 0.0))    # radar boresight left (+) of the vehicle forward axis (§3)
EGO_GYRO_BIAS_DPS = float(code_config.get("EGO_GYRO_BIAS_DPS", 0.0))   # last standstill gyro bias; removal happens in ego_velocity (§3)
EGO_TIME_OFFSET_S = float(code_config.get("EGO_TIME_OFFSET_S", 0.0))   # IMU time - radar frame timestamp for the same instant
                                                                       # (negative: the radar's stamp is later); sample_at adds it (§3)
EGO_SLOPE_DEG = 8.0                           # |pitch| or |roll| above this — 'slope' flag in info(): logged, never a
                                              # rejection (EGO_MOTION.md §4.1), so a replay can tell a slope-induced
                                              # bias from a calibration error
EGO_STATES = ("STANDING", "CREEPING", "MOVING", "UNKNOWN")      # EGO_MOTION.md §4.3


class EgoEstimator:
    """Carrier speed from the radar alone (Kellner et al., "Instantaneous ego-motion estimation using Doppler radar",
    ITSC 2013), EGO_MOTION.md §4. Sensor frame: x right, y forward = boresight, az = atan2(x, y) (right +). A stationary
    point shows Doppler -(v_fwd*cos az + vx*sin az) (TI: + = receding) with (v_fwd, vx) the radar's own velocity. The radar
    sits LEVER_ARM_X_M ahead of the yaw centre and LEVER_ARM_Y_M left of it, so a yaw rate w (rad/s, + left) gives
    v_fwd = v - w*LEVER_ARM_Y_M and vx = -w*LEVER_ARM_X_M: with a trusted gyro only the carrier speed v is unknown
    (1-parameter fit after removing the lever-arm term); without one the 2-parameter (v, vx) fit absorbs it and side-slip.
    Azimuths are rotated by MOUNT_YAW_DEG first, so v is along the vehicle's forward axis.

    Per frame: grid search on the inlier count over period-wrapped residuals (aliased targets still vote as long as the
    points are spread in azimuth) -> IRLS with a Cauchy weight on the inliers -> plausibility gate against the last
    accepted fit -> median of the last few fits. A state machine (STANDING / CREEPING / MOVING / UNKNOWN, 2-frame
    hysteresis) forces v = 0 exactly while standing, widens the inlier band while creeping and, when the radar has no
    fit, holds the last value for EGO_HOLD_S or bridges with the IMU's integrated velocity for EGO_BRIDGE_S. The radar's
    2-parameter fit also cross-checks the gyro on rich frames (yaw_rate_radar = v_left / LEVER_ARM_X_M) — never used as
    the operational yaw, only to flag gyro_ok False."""

    def __init__(self):
        self.v, self.vx, self.valid, self.t_valid = 0.0, 0.0, False, None
        self.state, self.source = "UNKNOWN", "radar"
        self.n_inliers, self.n_points, self.az_spread_deg, self.ambiguous = 0, 0, 0.0, False
        self.raw, self.bin = 0.0, 0.13                                        # this frame's unfiltered fit; Doppler bin
        self.v_fit, self.t_fit = 0.0, None                                    # last accepted fit: the plausibility-gate reference
        self.t0 = None                                                        # first frame time (bridge age before any fit)
        self.yaw_rate, self.yaw_rate_radar = 0.0, None                        # gyro yaw used in the fit (rad/s); radar cross-check
        self.gyro_ok_imu, self.gyro_ok = False, True                          # IMU's own flag; radar cross-check verdict
        self.imu_fresh = False                                                # an IMU sample young enough to be believed
        self.imu_seen = False                                                 # the link delivered a real sample at least once
        self.pitch_deg, self.roll_deg = 0.0, 0.0                              # last IMU attitude (§4.1: log, do not reject)
        self.gyro_bias_dps, self.time_offset_s = EGO_GYRO_BIAS_DPS, EGO_TIME_OFFSET_S
        self.hist = deque(maxlen=EGO_MEDIAN_FRAMES)
        self._pending, self._pending_n = None, 0                              # state-machine hysteresis
        self._xchk_bad, self._xchk_good, self._radar_still = 0, 0, False
        lo, hi = EGO_SEARCH_MPS
        self.grid = np.arange(lo, hi + 1e-9, EGO_GRID_STEP_MPS)
        self.vx_grid = np.arange(EGO_VX_SEARCH_MPS[0], EGO_VX_SEARCH_MPS[1] + 1e-9, EGO_VX_STEP_MPS)

    @property
    def moving(self):
        return self.valid and abs(self.v) >= EGO_STATIONARY_MPS

    def info(self):
        """EGO_MOTION.md §4.7 (+ the legacy fields, 'ttc_bound' from §4.3 and 'source': radar | imu-bridge)."""
        return {"v": round(self.v, 3), "vx": round(self.vx, 3), "raw": round(self.raw, 3), "valid": self.valid,
                "moving": self.moving, "state": self.state, "bin": round(self.bin, 3),
                "ttc_bound": self.state == "CREEPING",
                "yaw_rate": round(self.yaw_rate, 4),
                "yaw_rate_radar": None if self.yaw_rate_radar is None else round(self.yaw_rate_radar, 4),
                "gyro_ok": bool(self.gyro_ok_imu and self.gyro_ok), "imu_seen": self.imu_seen,
                "pitch_deg": round(self.pitch_deg, 1), "roll_deg": round(self.roll_deg, 1),
                "slope": bool(max(abs(self.pitch_deg), abs(self.roll_deg)) > EGO_SLOPE_DEG),
                "gyro_bias_dps": round(self.gyro_bias_dps, 3), "time_offset_s": round(self.time_offset_s, 3),
                "n_inliers": self.n_inliers, "n_points": self.n_points,
                "az_spread_deg": round(self.az_spread_deg, 1), "ambiguous": self.ambiguous, "source": self.source}

    # ---- pieces
    @staticmethod
    def _irls(A, b, x0, tol, iters=3):
        """Weighted least squares A x = -b with a Cauchy weight w = 1/(1+(r/tol)^2) (EGO_MOTION.md §4.4): points near the
        tolerance stop pulling the fit instead of being cut off. Returns None when the solve degenerates."""
        wt, sol = np.ones(len(b)), np.asarray(x0, dtype=float)
        for _ in range(iters):
            sw = np.sqrt(wt)
            sol, *_ = np.linalg.lstsq(A * sw[:, None], -b * sw, rcond=None)
            wt = 1.0 / (1.0 + ((b + A @ sol) / tol) ** 2)
        return sol if np.all(np.isfinite(sol)) else None

    def _fit2(self, cos_az, sin_az, vm, sel, v0, tol, period):
        """2-parameter (v, vx) fit: IRLS on the grid's inliers, then re-select inliers under the 2-parameter model
        (the 1-parameter grid drops the wide-azimuth points that carry the lateral information) and refit.
        Returns (sol, sel) or (None, sel)."""
        A = np.column_stack([cos_az, sin_az])
        sol, sel2 = None, sel
        for _ in range(2):
            vv = vm[sel2]
            if period:
                vv = vv - np.round((vv + v0 * cos_az[sel2]) / period) * period    # Doppler unwrapped around v0
            sol = self._irls(A[sel2], vv, [v0, 0.0], tol)
            if sol is None:
                return None, sel2
            rr = vm + A @ sol
            if period:
                rr -= np.round(rr / period) * period
            new = np.abs(rr) < tol
            if new.sum() < EGO_MIN_POINTS or np.array_equal(new, sel2):
                break
            sel2 = new
        return sol, sel2

    def _fit(self, az, vm, t, dt, tol, period, w):
        """One frame's fit. az — vehicle-frame azimuths (rad, right +); vm — measured Doppler; w — gyro yaw rate (rad/s)
        when trusted (1-parameter fit), None for the 2-parameter fit. Returns (v, vx_right) or (None, 0.0) when nothing
        acceptable was found (too few inliers, gated, ambiguous) — the caller then holds / bridges."""
        n = len(az)
        cos_az, sin_az = np.cos(az), np.sin(az)
        vm_c = vm - w * (LEVER_ARM_X_M * sin_az + LEVER_ARM_Y_M * cos_az) if w is not None else vm   # lever-arm term out
        # residual of every point under every candidate (v, vx): v_meas + v*cos(az) + vx*sin(az) (= 0 for a stationary
        # target). With the gyro the lever arm is already out, so vx = 0 is the only candidate; without it the lateral
        # term is large (w*LEVER_ARM_X_M ~ 0.6 m/s at 0.3 rad/s) and a v-only search finds no inliers at wide azimuths.
        vxg = np.array([0.0]) if w is not None else self.vx_grid
        # candidates: the plausibility gate (§4.5) discards everything farther than a_max*dt + 0.5*bin from the last
        # accepted fit, so only that window is worth evaluating. Without a gyro the residual tensor is
        # (candidates x vx_grid x N) and np.round over it dominates the frame — the window keeps the no-gyro branch
        # (exactly the degraded case) inside the Pi's 15 fps budget instead of 9x the gyro-aided cost.
        ref = gate = None
        grid = self.grid
        if self.t_fit is not None and (t - self.t_fit) <= EGO_HOLD_S and dt > 0:   # plausibility gate (§4.5): a tractor
            ref = self.v_fit                                                        # doesn't jump; narrower on thin evidence
            widest = EGO_MAX_ACCEL_MPS2 * dt + 0.5 * self.bin + 0.5 * EGO_GRID_STEP_MPS
            grid = self.grid[np.abs(self.grid - ref) <= widest]
            if not len(grid):
                return None, 0.0
        res = vm_c[None, None, :] + grid[:, None, None] * cos_az + vxg[None, :, None] * sin_az
        if period:
            res -= np.round(res / period) * period                                  # aliased targets still count
        inl = np.abs(res) < tol
        counts2 = inl.sum(axis=2)                                                   # (v, vx) -> inliers
        counts = counts2.max(axis=1)                                                # best vx for each candidate v
        ok = counts >= max(EGO_MIN_POINTS, EGO_MIN_INLIER_FRAC * n)
        if not ok.any():
            return None, 0.0
        if ref is not None:                                                         # narrow the gate on thin evidence
            gate = EGO_MAX_ACCEL_MPS2 * dt + (0.5 * self.bin if counts.max() >= 0.5 * n else 0.0)
            ok &= np.abs(grid - ref) <= gate + 0.5 * EGO_GRID_STEP_MPS              # (counts is the window's best now)
            if not ok.any():
                return None, 0.0
        best = int(counts[ok].max())
        cands = grid[ok & (counts >= best - 1)]                                     # near-ties: nearest to the previous
        self.ambiguous = bool(period) and float(cands.max() - cands.min()) > 0.6 * period   # value (standing when none)
        v0 = float(cands[np.argmin(np.abs(cands - (ref if ref is not None else 0.0)))])
        iv = int(np.argmin(np.abs(grid - v0)))
        sel = inl[iv, int(np.argmax(counts2[iv]))]
        if w is not None:                                                           # gyro-aided: v is the only unknown
            vv = vm_c[sel]
            if period:
                vv = vv - np.round((vv + v0 * cos_az[sel]) / period) * period       # Doppler unwrapped around v0
            sol = self._irls(cos_az[sel][:, None], vv, [v0], tol)
            v, vx = (None, 0.0) if sol is None else (float(sol[0]), -w * LEVER_ARM_X_M)
        else:                                                                       # no gyro: (v, vx) absorb the lever arm
            sol, sel = self._fit2(cos_az, sin_az, vm, sel, v0, tol, period)         # and side-slip
            v, vx = (None, 0.0) if sol is None else (float(sol[0]), float(sol[1]))
        self.n_inliers = int(sel.sum())
        self.az_spread_deg = math.degrees(float(az[sel].max() - az[sel].min()))
        # radar yaw cross-check (§4.6): the 2-parameter fit's lateral term on a rich, wide frame, raw Doppler
        # (lever arm included) -> yaw_rate_radar = v_left / LEVER_ARM_X_M. Cleared per frame in estimate(): the two
        # early returns above must not leave the previous frame's value looking current (drill T4 reads it).
        if self.n_inliers >= EGO_XCHECK_MIN_INLIERS and self.az_spread_deg >= EGO_XCHECK_MIN_SPREAD_DEG and LEVER_ARM_X_M:
            sol2 = sol if w is None else self._fit2(cos_az, sin_az, vm, sel, v0, tol, period)[0]
            if sol2 is not None:
                self.yaw_rate_radar = -float(sol2[1]) / LEVER_ARM_X_M
        if v is None:
            return None, 0.0
        if self.az_spread_deg < EGO_MIN_AZ_SPREAD_DEG:                              # v and v +- period (or v and a
            self.ambiguous = True                                                   # lateral slip) are indistinguishable
            return None, 0.0
        if ref is not None and abs(v - ref) > gate + 0.5 * EGO_GRID_STEP_MPS:      # the refined value must pass too
            return None, 0.0
        self.v_fit, self.t_fit = v, t
        return v, vx

    def _candidate_state(self, n, vm_abs, imu, gyro_still, v_fit):
        """EGO_MOTION.md §4.3 evidence for this frame, before hysteresis."""
        zupt = bool(imu.get("zupt")) and self.imu_fresh
        if n >= EGO_MIN_POINTS:
            # standing: the scene itself reads zero — median |Doppler| under one bin, few points above it (more are
            # tolerated when the IMU's ZUPT vouches: people walking around a standing tractor)
            self._radar_still = float(np.median(vm_abs)) < self.bin and \
                float(np.mean(vm_abs > self.bin)) < (0.5 if zupt else 0.25)
        else:
            # too few returns to judge: the IMU's ZUPT may vouch, but only if the last radar fit agrees the carrier
            # was already stopped. A constant-velocity drive trips the firmware's ZUPT (|a_fwd|, |a_right| and |yaw|
            # all under threshold at 4 m/s), and §4.3 lists no rule that lets the flag alone force v := 0.
            self._radar_still = zupt and (self.v_fit is None or abs(self.v_fit) < self.bin)
        if self._radar_still and gyro_still:
            return "STANDING"
        if v_fit is None:
            return "UNKNOWN"
        a = abs(v_fit)
        if a < self.bin:
            return "STANDING" if gyro_still else "CREEPING"                        # turning in place is not standing
        return "CREEPING" if a <= EGO_CREEP_BINS * self.bin else "MOVING"

    def _advance_state(self, cand):
        """Hysteresis: the candidate must repeat EGO_STATE_HYST_FRAMES frames before the state follows."""
        if cand == self.state:
            self._pending, self._pending_n = None, 0
            return
        if cand == self._pending:
            self._pending_n += 1
        else:
            self._pending, self._pending_n = cand, 1
        if self._pending_n >= EGO_STATE_HYST_FRAMES:
            self.state, self._pending, self._pending_n = cand, None, 0

    def _hold(self, t, imu):
        """No radar value this frame (§4.3 UNKNOWN): bridge with the IMU's integrated velocity for EGO_BRIDGE_S when it
        has one and is not in ZUPT, else keep the last value for EGO_HOLD_S, then 'standing, unknown'."""
        age = t - (self.t_valid if self.t_valid is not None else self.t0)
        vy_imu = imu.get("vy_imu") if self.imu_fresh else None
        if vy_imu is not None and math.isfinite(vy_imu) and not imu.get("zupt") and age <= EGO_BRIDGE_S:
            self.v, self.vx, self.valid, self.source = float(vy_imu), 0.0, True, "imu-bridge"
        elif age <= EGO_HOLD_S:
            pass
        else:
            self.v, self.vx, self.valid, self.source = 0.0, 0.0, False, "radar"
            self.hist.clear()
        return self.v

    # ---- one frame
    def estimate(self, dets, t, dt, period=None, doppler_res=None, imu=None):
        """imu — the dict from ego_velocity.EgoVelocityReader.sample_at(t) (yaw_rate rad/s + left bias-removed, pitch,
        roll, acc_fwd, vy_imu, zupt, gyro_ok, age_s) or None. Returns the carrier speed v (m/s, forward +)."""
        imu = imu or {}
        if self.t0 is None:
            self.t0 = t
        self.bin = doppler_res or 0.13
        tol = 0.75 * self.bin + EGO_INLIER_TOL_MPS
        if self.state == "CREEPING":
            tol *= 2                                                                 # §4.3: quantisation dominates at 1..5 bins
        w_imu = imu.get("yaw_rate")
        age = float(imu.get("age_s", math.inf))
        self.imu_fresh = age <= EGO_IMU_MAX_AGE_S
        if math.isfinite(age):
            self.imu_seen = True                                                     # distinguishes 'never had a link'
        self.gyro_ok_imu = bool(imu.get("gyro_ok")) and w_imu is not None and math.isfinite(w_imu)
        # the yaw the fit actually uses — and the only one info() publishes (§4.6): after the radar/gyro cross-check
        # has rejected the gyro, the tracker's de-rotation and the corridor curvature fall back to 'straight' rather
        # than following a sensor the system has just declared untrustworthy
        gyro_trusted = self.gyro_ok_imu and self.gyro_ok
        self.yaw_rate = float(w_imu) if gyro_trusted else 0.0
        self.pitch_deg = float(imu.get("pitch", 0.0) or 0.0)                          # §4.1: carried, logged, not a gate
        self.roll_deg = float(imu.get("roll", 0.0) or 0.0)
        self.gyro_bias_dps = float(imu.get("gyro_bias_dps", EGO_GYRO_BIAS_DPS))
        self.time_offset_s = float(imu.get("time_offset_s", EGO_TIME_OFFSET_S))
        gyro_still = (not gyro_trusted) or abs(math.degrees(self.yaw_rate)) < EGO_STANDING_GYRO_DPS
        mount = math.radians(MOUNT_YAW_DEG)                                          # boresight left of forward -> a point
        pts = [(d["azimuth_rad"] - mount, d["doppler_mps"]) for d in dets           # on it is left of the vehicle axis
               if d["range_m"] >= EGO_MIN_RANGE_M and d["snr_db"] == d["snr_db"]]    # no leakage, no NaN-SNR points
        self.n_points = n = len(pts)
        self.n_inliers, self.az_spread_deg, self.ambiguous = 0, 0.0, False
        self.yaw_rate_radar = None                                                   # per frame, whatever path _fit takes
        v_fit, vx_fit, vm_abs = None, 0.0, None
        if n >= EGO_MIN_POINTS:
            az = np.array([p[0] for p in pts]); vm = np.array([p[1] for p in pts])
            vm_abs = np.abs(vm)
            v_fit, vx_fit = self._fit(az, vm, t, dt, tol, period, self.yaw_rate if gyro_trusted else None)
        # gyro cross-check bookkeeping (§4.6): 15 frames of > 3 deg/s disagreement while MOVING -> gyro_ok False
        if self.yaw_rate_radar is not None and self.gyro_ok_imu and self.state == "MOVING":
            bad = abs(math.degrees(self.yaw_rate_radar - w_imu)) > EGO_XCHECK_TOL_DPS
            self._xchk_bad, self._xchk_good = (self._xchk_bad + 1, 0) if bad else (0, self._xchk_good + 1)
            if self._xchk_bad >= EGO_XCHECK_FRAMES:
                self.gyro_ok = False
            elif self._xchk_good >= EGO_XCHECK_FRAMES:
                self.gyro_ok = True
        self._advance_state(self._candidate_state(n, vm_abs, imu, gyro_still, v_fit))
        if self.state == "STANDING":
            # v := 0 exactly (§4.3), but 'raw' keeps this frame's unclamped fit: STANDING is a one-bin deadband
            # (0.27 m/s on hangar_v9), and alerts.py gates the collision layer on raw, not on the state alone,
            # so a tractor creeping at 0.25 m/s does not silence TTC / corridor (§5)
            self.raw = float(v_fit) if v_fit is not None else 0.0
            self.v, self.vx, self.valid, self.source, self.t_valid = 0.0, 0.0, True, "radar", t
            self.hist.clear()
            if self._radar_still:
                self.v_fit, self.t_fit = 0.0, t                                      # gate reference: standing
        elif v_fit is not None:                              # CREEPING / MOVING — or UNKNOWN with a fresh fit (label lags)
            self.raw = v_fit
            self.hist.append(v_fit)
            self.v, self.vx, self.valid, self.source, self.t_valid = float(np.median(self.hist)), vx_fit, True, "radar", t
        else:
            self._hold(t, imu)
        return self.v


# ---------------------------------------------------------------- background map
class BackgroundMap:
    """Short-memory ego buffer (EGO_MOTION.md §4, last paragraph). An occupancy-count grid of static returns that moves
    with the carrier: the sensor pose is dead-reckoned from the ego estimate (v, vx, yaw rate) in a map frame anchored at
    the first frame, cells decay with forget_s while moving and keep accumulating while standing ('learn while standing').
    It is persistence EVIDENCE only: mark() writes dets[i]['persist'] — frames a static return was seen in that cell
    (±1 cell: the reflection jitters) — and never sets 'background'. A static object in the corridor is an obstacle
    confirmed by persist >= BACKGROUND_PERSIST_FRAMES, not clutter to suppress (§5)."""

    def __init__(self, cell=0.25, forget_s=BACKGROUND_FORGET_S, half_m=20.0, recenter_m=8.0):
        self.cell, self.forget_s, self.half_m, self.recenter_m = cell, forget_s, half_m, recenter_m
        self.n = int(round(2 * half_m / cell))
        self.counts = np.zeros((self.n, self.n), dtype=np.float32)   # [ix, iy] in the map frame
        self.ox = self.oy = -half_m                                   # map-frame coordinates of cell (0, 0)
        self.px, self.py, self.heading = 0.0, 0.0, 0.0                # sensor pose in the map frame (x right, y forward at t0)
        self.frames_seen, self.t_last = 0, None
        self.learning = False                                         # kept for render(): the buffer is always live

    def _cells(self, dets):
        """Detections (sensor frame) -> map-frame cell indices, None when outside the grid."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        out = []
        for d in dets:
            x, y = d["x_m"], d["y_m"]
            mx, my = self.px + x * c - y * s, self.py + x * s + y * c
            ix, iy = int((mx - self.ox) / self.cell), int((my - self.oy) / self.cell)
            out.append((ix, iy) if 0 <= ix < self.n and 0 <= iy < self.n else None)
        return out

    def _shift(self, k, axis):
        """Move the grid contents by -k cells along axis (the origin moves +k cells); the wrapped strip is cleared."""
        self.counts = np.roll(self.counts, -k, axis=axis)
        idx = [slice(None), slice(None)]
        idx[axis] = slice(-k, None) if k > 0 else slice(None, -k)
        self.counts[tuple(idx)] = 0.0

    def _advance(self, dt, v, vx, w):
        """Dead-reckon the pose: the sensor moved (vx, v)*dt in its own frame and yawed w*dt (+ left). When it wanders
        recenter_m from the grid centre, shift the grid by whole cells so the surroundings stay inside."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        self.px += (vx * c - v * s) * dt
        self.py += (vx * s + v * c) * dt
        self.heading += w * dt
        for axis in (0, 1):
            p, o = (self.px, self.ox) if axis == 0 else (self.py, self.oy)
            off = p - (o + self.half_m)
            if abs(off) > self.recenter_m:
                k = int(round(off / self.cell))
                self._shift(k, axis)
                if axis == 0:
                    self.ox += k * self.cell
                else:
                    self.oy += k * self.cell

    def mark(self, dets, t, ego_speed, dt=None, yaw_rate=0.0, ego_vx=0.0, standing=None):
        """Advance the buffer by one frame and fill dets[i]['persist']. standing — the estimator's STANDING state
        (defaults to |ego_speed| < 0.1 for old callers); while standing nothing moves or decays."""
        if standing is None:
            standing = abs(ego_speed) < 0.1
        if dt is None:
            dt = (t - self.t_last) if self.t_last is not None else 0.0
        self.t_last = t
        self.frames_seen += 1
        if not standing and dt > 0:
            self._advance(dt, ego_speed, ego_vx, yaw_rate)
            self.counts *= math.exp(-dt / self.forget_s)                          # forget while moving
        cells = self._cells(dets)
        for d, c in zip(dets, cells):
            if c is not None and d["is_static"]:
                self.counts[c] = min(self.counts[c] + 1.0, BACKGROUND_COUNT_CAP)   # capped: see BACKGROUND_COUNT_CAP
        for d, c in zip(dets, cells):
            if c is None:
                d["persist"] = 0
            else:
                ix, iy = c
                d["persist"] = int(round(float(self.counts[max(0, ix - 1):ix + 2, max(0, iy - 1):iy + 2].max())))
            d["background"] = False                                                # never a suppression mask
        return dets

    @property
    def n_cells(self):
        """Cells holding a confirmed static return (for the on-screen status line)."""
        return int((self.counts >= BACKGROUND_PERSIST_FRAMES).sum())


# ---------------------------------------------------------------- point classification
def _finite(x):
    """True for a real number — the IMU dict carries None for 'unknown' and NaN survives arithmetic silently."""
    return x is not None and math.isfinite(float(x))


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
def cluster_objects(dets, kinds, confs, eps_m=CLUSTER_EPS_M, v_weight=CLUSTER_V_WEIGHT, min_pts=1, bin_mps=None):
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
            # cluster-level static/moving label (EGO_MOTION.md §5): median compensated Doppler of >= 3 members against
            # 0.8*bin. Tighter than the per-point max(0.2, 1.2*bin) threshold and immune to the +-half-bin quantisation
            # that makes single points of one post disagree with each other. None = not enough members to say.
            "is_static": None if (bin_mps is None or len(mem) < 3) else
                         bool(abs(float(np.median([dets[i].get("doppler_rel_mps", dets[i]["doppler_mps"])
                                                   for i in mem]))) < 0.8 * bin_mps),
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
        self.is_static = obj.get("is_static")          # cluster-level static label (EGO_MOTION.md §5), None = unknown
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

    def rotate(self, theta):
        """Rotate the state (position and velocity) about the sensor by theta (rad, counter-clockwise seen from above,
        x right / y forward). Used when the carrier yaws: the world turns the other way in the sensor frame."""
        c, s_ = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s_], [s_, c]])
        M = np.zeros((4, 4)); M[:2, :2] = R; M[2:, 2:] = R
        self.x = M @ self.x
        self.P = M @ self.P @ M.T

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
        if obj.get("is_static") is not None:
            self.is_static = obj["is_static"]           # keep the last cluster that had >= 3 members to judge with
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

    def ground_velocity(self, ego_vx_mps=0.0, ego_vy_mps=0.0):
        """(vx, vy) in a stationary ground frame instead of the platform's own moving frame — i.e. what
        this object's velocity would read on a non-moving radar. x/vx, y/vy, radial_mps and speed_mps are
        deliberately left as platform-relative everywhere else (that's the correct frame for collision/
        closing-speed judgments — see EGO_VELOCITY.md); this is an additional, separate view for telling
        "this object is actually moving" from "it only looks like it's moving because the platform is".

        Takes the full 2D ego-velocity (ego_vx_mps lateral/right, ego_vy_mps forward — same x=right,
        y=forward convention as everywhere else in this file) from ego_velocity.py's EgoVelocityReader.
        Note this is a *different, more complete* correction than the single-scalar-forward assumption
        Pipeline.ego / points_to_detections() still use for Doppler compensation — see EGO_VELOCITY.md
        for why that one stays forward-only (it also feeds a model trained on that exact assumption) while
        this one takes the full vector now that both components are actually available."""
        return float(self.x[2]) + ego_vx_mps, float(self.x[3]) + ego_vy_mps

    def ground_speed_mps(self, ego_vx_mps=0.0, ego_vy_mps=0.0):
        gvx, gvy = self.ground_velocity(ego_vx_mps, ego_vy_mps)
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
                "kind": self.kind, "kind_conf": round(self.kind_conf, 1), "is_static": self.is_static,
                "age_frames": self.age, "hits": self.hits, "misses": self.misses,
                "coasting": self.coasting, "sigma_xy_m": round(float(math.sqrt(self.P[0, 0] + self.P[1, 1])), 3)}


class Tracker:
    def __init__(self, confirm_hits=TRACK_CONFIRM_HITS, confirm_window=TRACK_CONFIRM_WINDOW,
                 max_misses=TRACK_MAX_MISSES, gate=3.5, merge_m=0.5):
        self.confirm_hits, self.confirm_window, self.max_misses = confirm_hits, confirm_window, max_misses
        self.gate, self.merge_m = gate, merge_m
        self.tracks = []

    def step(self, objs, t, dt, yaw_rate=0.0):
        """yaw_rate — carrier turn rate, rad/s, + = turning left (counter-clockwise from above)."""
        for tr in self.tracks:
            if yaw_rate:
                tr.rotate(-yaw_rate * dt)            # the carrier turned left by w*dt -> the world turned right
            tr.predict(dt)
        cand = []
        for i, tr in enumerate(self.tracks):
            for j, o in enumerate(objs):
                g = tr.gate_distance(o)
                if g < self.gate:
                    cand.append((g, i, j))
        cand.sort()
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


def ego_label(ego):
    """One-line carrier-motion status for the on-screen views."""
    if not ego or not ego.get("valid"):
        return "ego: n/a"
    src = ego.get("source", "-")
    if src == "radar":
        src += f" {ego.get('n_inliers', 0)}/{ego.get('n_points', 0)} pts" + (" ?" if ego.get("ambiguous") else "")
    return f"ego {ego['v']:+.2f} m/s ({src})" if ego.get("moving") else f"ego: standing ({src})"


def render(dets, kinds, tracks, bg, meters=DRAW_METERS, title="", only_ids=None, fps=None, ego=None):
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
    if ego is not None:                                       # carrier motion, second status line at the top
        cv2.putText(img, ego_label(ego), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
    # legend
    y = 46 if ego is not None else 30
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
    def __init__(self, cfg, model=None, ego_speed=0.0, use_background=USE_BACKGROUND, ego_mode=None):
        self.cfg, self.model, self.ego = cfg, model, ego_speed
        self.ego_mode = ego_mode or EGO_MODE
        self.ego_est = EgoEstimator()
        bin_ = cfg.get("doppler_res_mps")
        # per-point static threshold from the profile: ≈ 3σ of the ±½-bin quantisation floor (EGO_MOTION.md §5)
        self.static_doppler_mps = max(0.2, 1.2 * bin_) if bin_ else STATIC_DOPPLER_MPS
        self.ego_est.bin = bin_ or self.ego_est.bin
        self.ego_info = self._fixed_info(ego_speed, self.ego_mode, valid=self.ego_mode == "fixed")
        self.ego_yaw_rate_dps = 0.0   # deg/s, + = left; carried through for the overlay/JSONL (EGO_VELOCITY.md)
        self.period = cfg.get("frame_period_s") or 0.1
        global DOPPLER_PERIOD
        DOPPLER_PERIOD = cfg.get("doppler_period_mps")
        self.bg = BackgroundMap(BACKGROUND_CELL_M, BACKGROUND_FORGET_S) if use_background else None
        self.tracker = Tracker()
        self.last_frame_num, self.t = None, 0.0

    def _fixed_info(self, v, source, valid=True):
        """ego_info with the estimator's full key set (EGO_MOTION.md §4.7) for a speed that did not come from the fit."""
        state = ("STANDING" if abs(v) < EGO_STATIONARY_MPS else "MOVING") if valid else "UNKNOWN"
        return {**self.ego_est.info(), "v": float(v), "vx": 0.0, "raw": float(v), "valid": valid,
                "moving": valid and abs(v) >= EGO_STATIONARY_MPS, "state": state, "ttc_bound": False,
                "n_inliers": 0, "n_points": 0, "az_spread_deg": 0.0, "ambiguous": False, "source": source}

    def process(self, frame, ego_speed=None, yaw_rate=0.0, imu=None):
        """ego_speed — carrier speed for this frame from an external source (m/s, forward +), wins over EGO_MODE;
        None: "radar" estimates it from the frame's stationary targets, "fixed" keeps the constant.
        yaw_rate — carrier turn rate (rad/s, + = left) from a gyro; 0 when unknown (legacy callers: tracker only).
        imu — the dict from ego_velocity.EgoVelocityReader.sample_at(t) for this frame (yaw_rate rad/s + left,
        bias-removed; pitch, roll, acc_fwd, vy_imu, zupt, gyro_ok, age_s) or None. When present it feeds the
        estimator (EGO_MOTION.md §4: gyro-aided fit, STANDING support, IMU bridge) and its yaw_rate replaces the
        yaw_rate argument for the tracker — but only while the sample is fresh AND gyro_ok (the reader interpolates
        and holds, it does not zero), and in radar mode the yaw finally published is the one the fit actually used,
        so a dead link or a failed §4.6 cross-check leaves the tracker and the corridor at 'not turning'."""
        # time — from the radar frame number: dropped USB packets don't compress the tracker's time
        if self.last_frame_num is not None:
            gap = frame["frame"] - self.last_frame_num
            dt = self.period * (gap if 0 < gap < 100 else 1)
        else:
            dt = self.period
        self.last_frame_num = frame["frame"]
        self.t += dt
        w = float(yaw_rate or 0.0)
        # only a FRESH, calibrated sample steers anything (§5): sample_at() keeps returning an interpolated/held
        # yaw with gyro_ok False after the link dies, and feeding that on forever spins the tracker, winds up the
        # background buffer's heading and bends the corridor out of the way of a real obstacle
        if imu is not None and imu.get("gyro_ok") and _finite(imu.get("yaw_rate")):
            w = float(imu["yaw_rate"])

        dets = points_to_detections(frame, self.cfg, 0.0, self.static_doppler_mps)   # raw first: the estimate needs uncompensated Doppler
        if ego_speed is not None:
            self.ego = float(ego_speed)
            self.ego_info = self._fixed_info(self.ego, "external")
        elif self.ego_mode == "radar":
            self.ego = self.ego_est.estimate(dets, self.t, dt, DOPPLER_PERIOD, self.cfg.get("doppler_res_mps"), imu)
            self.ego_info = self.ego_est.info()
            if imu is not None:                      # the yaw the fit actually used — 0 when the sample was stale or
                w = self.ego_est.yaw_rate            # the §4.6 cross-check rejected the gyro (a legacy caller's
                                                     # explicit yaw_rate= argument is left alone)
        self.ego_info["yaw_rate"] = w
        self.ego_yaw_rate_dps = math.degrees(w)
        # compensate with the FULL sensor-frame ego velocity (§5): forward v and the fit's lateral
        # vx = -w*LEVER_ARM_X_M, both rotated back from the vehicle frame the fit lives in. Forward-only leaves
        # w*LEVER_ARM_X_M*sin(az) on every stationary return, so a headland turn labels the wide-azimuth clutter
        # 'moving' and BackgroundMap stops accumulating exactly when the corridor evidence is needed.
        vx_ego = float(self.ego_info.get("vx", 0.0) or 0.0)
        if self.ego != 0.0 or vx_ego != 0.0:                             # ~0 for a stationary target afterwards
            mount = math.radians(MOUNT_YAW_DEG)
            for d in dets:
                a = d["azimuth_rad"] - mount                             # sensor azimuth -> vehicle frame
                d["doppler_rel_mps"] = d["doppler_mps"] + self.ego * math.cos(a) + vx_ego * math.sin(a)
                d["is_static"] = abs(d["doppler_rel_mps"]) < self.static_doppler_mps
        if self.bg is not None:                                          # persistence evidence, moves with the carrier
            dets = self.bg.mark(dets, self.t, self.ego, dt, w, self.ego_info.get("vx", 0.0),
                                standing=self.ego_info.get("state") == "STANDING")
        kinds, confs = predict_points(self.model, dets, self.ego, abs(self.ego) > 0.1)
        objs = cluster_objects(dets, kinds, confs, bin_mps=self.cfg.get("doppler_res_mps"))
        tracks = self.tracker.step(objs, self.t, dt, yaw_rate=w)
        noise = noise_floor(frame, self.cfg)
        return {"t": self.t, "dt": dt, "dets": dets, "kinds": kinds, "confs": confs,
                "objs": objs, "tracks": tracks, "noise": noise, "ego": dict(self.ego_info),
                "ego_speed_mps": self.ego, "ego_yaw_rate_dps": self.ego_yaw_rate_dps}


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
                # the reader (EGO_MOTION.md §5) interpolates a full IMU sample to the frame time and applies
                # EGO_TIME_OFFSET_S / EGO_GYRO_BIAS_DPS; the radar Doppler fit stays the primary speed. Its ring buffer
                # is stamped with time.time() here (the reader's default clock), so the frame time must be the same base.
                imu = ego_reader.sample_at(time.time())
                out = pipe.process(frame, imu=imu)                       # speed: EGO_MODE (radar)
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
                        "ego_yaw_rate_dps": out["ego_yaw_rate_dps"], "ego": out["ego"],
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
