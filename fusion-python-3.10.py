# =====================================================================
#  fusion.py — camera (YOLO / YOLO-World, ByteTrack tracks) + radar (iwr1642_live.Pipeline, EKF tracks)
#             → matching based on azimuth (from camera viewpoint, accounting for parallax) and bbox height
#             → radar track state: class from camera; range and velocity from radar
#             → radar maintains the bounding box when the camera loses the object (+ reason for loss)
#             → merging of multiple radar tracks corresponding to a single object
#             → CSV + camera image with overlay + top-down view
#
#  Launch:  python fusion.py                          — live camera + radar (or --dump x.bin)
#           python fusion.py --selftest --dump x.bin   — no camera: camera emulated from radar data
#           offline using record_sync recording:  python fusion_offline.py rec_<timestamp>
#
#  Radar — via iwr1642_live only (tracks with IDs). No custom TLV parsing here.
# =====================================================================
import argparse
import os
import csv
import math
import threading
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

import iwr1642_live as radar
import radar_filter
import alerts

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
try:
    import torch
except ImportError:
    torch = None

with open("configs.json", "r") as file:
    code_config = json.load(file)            # Read/edit configs.json
# ---------------------------------------------------------------- SETTINGS
CAMERA_INDEX = code_config["CAMERA_INDEX"]   # 0 — built-in camera; 1 — external USB
DETECTOR = code_config["DETECTOR"]           # "yolov8n" — COCO, fast; "yolo-world" — open vocabulary (WORLD_CLASSES),
                                             # first run downloads weights + CLIP (~340 MB), ~0.25 s/frame on CPU
USE_GPU = code_config.get("USE_GPU", 1)      # 1 — run YOLO on the GPU (CUDA) if one is available; 0 — force CPU
USE_NCNN = code_config.get("USE_NCNN", 0)    # 1 — export/run YOLO via the NCNN backend instead of plain PyTorch;
                                             # much faster on ARM/Raspberry Pi CPUs (no CUDA there anyway). CPU-only:
                                             # ignores USE_GPU/YOLO_DEVICE. First run exports and caches a
                                             # <weights>_ncnn_model/ folder next to the .pt weights (needs internet
                                             # once, to fetch the pnnx converter); later runs load the cached export.
YOLO_IMGSZ = code_config.get("YOLO_IMGSZ", 640)   # inference resolution passed to model.track(); an int (square) or
                                                  # [h, w] (each must be a multiple of 32) — e.g. [320, 416] on a Pi
YOLO_EVERY_N = max(1, int(code_config.get("YOLO_EVERY_N", 1)))  # run YOLO on every N-th camera frame only; in between
                                                  # the last boxes are reused (ByteTrack keeps its ids across calls). On a
                                                  # Pi 4 N=2 roughly doubles the loop/display rate at the same detection rate
CAM_WIDTH = int(code_config.get("CAM_WIDTH", 640))       # capture size requested from the camera (it may pick the nearest)
CAM_HEIGHT = int(code_config.get("CAM_HEIGHT", 480))
CAM_MJPG = int(code_config.get("CAM_MJPG", 0))           # 1 — ask the webcam for MJPG: many USB cams only give 30 fps in MJPG
_ds = float(code_config.get("DISPLAY_SCALE", 1.0))
DISPLAY_SCALE = 1.0 if _ds <= 0 else min(1.0, max(0.1, _ds))  # shrink BOTH on-screen windows (fewer pixels for VNC);
                                                  # <= 0 means off, never upscale. Overlay text is tuned for a 640-wide
                                                  # frame: keep CAM_WIDTH >= 640, shrink with DISPLAY_SCALE instead
YOLO_WEIGHTS = {"yolov8n": "yolov8n.pt", "yolo-world": "yolov8s-worldv2.pt"}
YOLO_CONF = 0.4
#YOLO_KEEP = {0: ("person", 1.70), 1: ("bicycle", 1.10), 2: ("car", 1.50), 3: ("motorcycle", 1.20),
#             5: ("bus", 3.00), 7: ("truck", 3.00), 16: ("dog", 0.55), 17: ("horse", 1.60),
#             18: ("sheep", 0.90), 19: ("cow", 1.40), 39: ("bottle", 0.25), 41: ("cup", 0.12)}
#WORLD_CLASSES = {"person": 1.70, "car": 1.50, "truck": 3.00, "tractor": 2.80, "dog": 0.55, "cow": 1.40,
#                 "aluminum can": 0.12, "bottle": 0.25, "chair": 0.90, "pole": 2.00, "box": 0.40}

# DETECT HUMAN ONLY
YOLO_KEEP = {0: ("person", 1.70)}
WORLD_CLASSES = {"person": 1.70}

CAM_HFOV_DEG = 70.0                  # camera horizontal field of view
CAM_YAW_DEG = 0.0                    # yaw = radar_azimuth − camera_azimuth for the same object (median over pairs);
                                     # positive if the radar axis is rotated LEFT of the camera axis. Refined by calibration.
RADAR_TO_CAMERA_M = {"right": 0.0,\
                     "up": 0.0, "forward": 0.0}   # where the radar is relative to the lens; read from meta.json
MAX_DT_S = 0.15                      # allowed desync between camera and radar frames (camera newer than radar: this;
                                     # camera older: this + measured YOLO latency + camera period, see Fusion.sync_window)
SYNC_HARD_CAP_S = 0.6                # never match a camera frame older than this, however slow the detector is
AZ_SIGMA_DEG = 4.0                   # expected azimuth error between sensors
RANGE_REL_SIGMA = 0.35               # relative range error from bbox height (±35 %)
CLIPPED_RANGE_SIGMA = 1.2            # ...if the bbox hits the frame edge — height is clipped, range is only an upper bound
GATE = 3.0                           # matching threshold in sigmas
HISTORY_BONUS = 1.0                  # cost discount if a detection with the same cam_id already matched this track
PAIR_CONFIRM = 5                     # matches needed for a track to become an "object with a class"
FORGET_S = 2.0                       # track memory without radar — this many seconds
HOLD_MAX_S = 30.0                    # camera lost the object, radar is tracking: how many seconds to hold the box
HOLD_FOV_MARGIN_DEG = 8.0            # object is "in frame" if azimuth is within FOV with margin ≥ 2·AZ_SIGMA
MERGE_DIST_M = 0.9                   # an unpaired radar track closer than this to an object...
MERGE_DV_MPS = 0.8                   # ...with a similar speed (wraparound-aware) and the same azimuth — is part of it, not a separate object
RADAR_DOT_SUPPRESS_MARGIN_PX = 40    # a "radar only" dot within this many px (horizontally) of a shown object's box is a duplicate, not a separate detection
SHOW_ONLY_INTERESTING = True
MOVING_MPS = 0.25
RADAR_STALE_S = 0.5                  # radar hasn't updated for this long — consider it lost (banner, tracks not drawn)
CORRIDOR_HALF_WIDTH_M = 1.5          # while the carrier moves, a stationary object inside this lateral band ahead is an
CORRIDOR_AHEAD_M = 20.0              # obstacle in the path (post, parked machine), not background — shown and alerted
EGO_SERIAL_PORT = code_config.get("EGO_SERIAL_PORT", "")   # $RDEGO source (ESP32 with BNO085/GNSS); "" = radar-only ego,
                                                            # "same" = the alert node's port (one USB cable both ways)
EGO_BAUD = code_config.get("EGO_BAUD", 115200)
EGO_STALE_S = 1.0                    # no $RDEGO for this long — fall back to the radar's own estimate
CSV_FOLDER = "logs"
CSV_PATH = os.path.join(CSV_FOLDER, f"fusion_{datetime.now():%Y%m%d_%H%M%S}.csv")
SHOW_WINDOW = True
SHOW_FPS = code_config["SHOW_FPS"]   # 1 — draw the fps counter on the radar/fusion displays; 0 — off

ALERTS_ENABLED = code_config.get("ALERTS_ENABLED", 1)          # 1 — evaluate and send alerts.py events; 0 — off
ALERT_SERIAL_PORT = code_config.get("ALERT_SERIAL_PORT", "")   # e.g. "COM8" / "/dev/ttyUSB1" — the MCU's port; "" — console only, no serial sink
ALERT_BAUD = code_config.get("ALERT_BAUD", 115200)
PROXIMITY_WARN_M = code_config.get("PROXIMITY_WARN_M", 3.0)          # object closer than this — PROXIMITY_WARNING
PROXIMITY_CRITICAL_M = code_config.get("PROXIMITY_CRITICAL_M", 1.5)  # ...and this — PROXIMITY_CRITICAL
CLOSING_SPEED_ALERT_MPS = code_config.get("CLOSING_SPEED_ALERT_MPS", 2.0)  # FAST_APPROACH threshold
RADAR_ONLY_CONFIRM_S = code_config.get("RADAR_ONLY_CONFIRM_S", 1.0)  # radar-only object must persist this long before alerting (avoids alerting on a single-frame clutter blip)
ALERT_RESEND_S = code_config.get("ALERT_RESEND_S", 2.0)              # heartbeat interval for an alert that's still active

Q_SHARP_DROP = 0.45        # sharpness below 45 % of the reference
Q_CONTRAST_DROP = 0.45     # contrast below 45 % of the reference

YOLO_DEVICE = "cpu"
if USE_GPU:
    if torch is not None and torch.cuda.is_available():
        YOLO_DEVICE = "cuda:0"
    else:
        print("⚠️ USE_GPU=1 but no CUDA GPU available (torch missing or no CUDA build) — running YOLO on CPU")

# ---------------------------------------------------------------- camera geometry
class CameraModel:
    """Pixel ↔ azimuth via focal length in pixels. The radar and camera sit at different points: all camera
    quantities are computed from the lens point (parallax); yaw is the axis rotation."""

    def __init__(self, width, height, hfov_deg=CAM_HFOV_DEG, yaw_deg=CAM_YAW_DEG, radar_to_cam=None):
        self.w, self.h = width, height
        self.set_hfov(hfov_deg)
        self.yaw = yaw_deg
        rc = radar_to_cam or RADAR_TO_CAMERA_M
        self.dx, self.dy = rc.get("right", 0.0), rc.get("forward", 0.0)   # radar relative to the lens, m

    def set_hfov(self, hfov_deg):
        self.hfov = hfov_deg
        self.f = (self.w / 2) / math.tan(math.radians(hfov_deg / 2))

    # --- a point in the radar frame → how the camera sees it
    def cam_view(self, x_r, y_r):
        """(azimuth in the radar frame, but from the camera point; depth from the camera). x is right, y is forward."""
        xc, yc = x_r + self.dx, y_r + self.dy
        return math.degrees(math.atan2(xc, yc)), yc

    def azimuth_of_u(self, u):
        """Pixel azimuth converted to the radar frame (yaw applied), from the camera point."""
        return math.degrees(math.atan2(u - self.w / 2, self.f)) + self.yaw

    def u_of_cam_azimuth(self, az_from_cam_deg):
        return self.w / 2 + self.f * math.tan(math.radians(az_from_cam_deg - self.yaw))

    def range_from_bbox(self, bbox_h_px, obj_height_m):
        return self.f * obj_height_m / max(bbox_h_px, 1.0)

    def bbox_clipped(self, x1, y1, x2, y2):
        return y1 <= 1 or y2 >= self.h - 2

    def bbox_side_clipped(self, x1, y1, x2, y2):
        return x1 <= 1 or x2 >= self.w - 2


class Calibrator:
    def __init__(self, min_pairs=30):
        self.pairs, self.min_pairs = [], min_pairs

    def add(self, az_cam_raw, az_radar_from_cam):
        self.pairs.append(az_radar_from_cam - az_cam_raw)
        if len(self.pairs) > 500:
            self.pairs = self.pairs[-500:]

    def yaw(self):
        return float(np.median(self.pairs)) if len(self.pairs) >= self.min_pairs else None


# ---------------------------------------------------------------- frame quality
def frame_quality(gray, bbox):
    if gray is None or cv2 is None:
        return None
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = bbox
    # central 60 % of the box: the edges of a virtual box often capture background, not the object/occluder
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.6, (y2 - y1) * 0.6
    x1, x2 = int(max(0, min(w - 1, cx - bw / 2))), int(max(0, min(w, cx + bw / 2)))
    y1, y2 = int(max(0, min(h - 1, cy - bh / 2))), int(max(0, min(h, cy + bh / 2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    roi = gray[y1:y2, x1:x2]
    small = cv2.resize(gray, (w // 4, h // 4)) if w >= 64 else gray
    return {"sharp": float(cv2.Laplacian(roi, cv2.CV_64F).var()), "contrast": float(roi.std()), "bright": float(roi.mean()),
            "g_sharp": float(cv2.Laplacian(small, cv2.CV_64F).var()), "g_contrast": float(small.std())}


def lost_reason(ref, now):
    """exposure → whole frame (haze/fog/defocus) → object area (occluded) → detector miss."""
    if now is None:
        return "no frame"
    if ref is None:
        return "unknown"
    if now["bright"] > 235 or now["bright"] < 20:
        return "over/under-exposed"
    if now["g_sharp"] / max(ref["g_sharp"], 1e-6) < Q_SHARP_DROP:
        return "haze / smoke / defocus"
    if now["contrast"] / max(ref["contrast"], 1e-6) < Q_CONTRAST_DROP and now["sharp"] / max(ref["sharp"], 1e-6) < Q_SHARP_DROP:
        return "occluded"
    return "detector miss"


# ---------------------------------------------------------------- data
@dataclass
class CamDet:
    t: float
    cam_id: int
    cls: str
    conf: float
    x1: int; y1: int; x2: int; y2: int
    obj_h_m: float = 1.7

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def bh(self): return self.y2 - self.y1


@dataclass
class TrackMem:
    """Object memory keyed by the radar track (not by ID pair — ByteTrack changes IDs, the radar track is more stable)."""
    radar_id: int
    hits: int = 0
    cam_id: Optional[int] = None
    last_t: float = 0.0
    last_cam_t: float = 0.0
    classes: dict = field(default_factory=lambda: defaultdict(float))
    last_bbox: Optional[tuple] = None
    last_bbox_range: float = 0.0
    last_bbox_side_clipped: bool = False
    last_full_width_px: float = 0.0     # width of the last box not clipped by the frame edge
    last_full_width_range: float = 0.0
    last_az_from_cam: float = 0.0
    ref_quality: Optional[dict] = None
    reasons: deque = field(default_factory=lambda: deque(maxlen=10))
    # history of (1/r, v_top, v_bottom) while the camera can see it: v_edge = v_h + k/r → horizon and box scale
    edge_hist: deque = field(default_factory=lambda: deque(maxlen=60))

    @property
    def cls(self):
        return max(self.classes, key=self.classes.get) if self.classes else "unknown"

    def edge_model(self, which):
        """Linear regression v_edge = a + b·(1/r) over the history; None if there's too little history or too little range spread."""
        pts = [(inv_r, vt if which == "top" else vb) for inv_r, vt, vb in self.edge_hist if (vt if which == "top" else vb) is not None]
        if len(pts) < 4:
            return None
        X = np.array([p[0] for p in pts]); Y = np.array([p[1] for p in pts])
        if X.max() - X.min() < 0.08:                       # ranges barely changed — slope is undetermined
            return None
        b, a = np.polyfit(X, Y, 1)
        return a, b


# ---------------------------------------------------------------- fusion
class Fusion:
    def __init__(self, cam: CameraModel, csv_path=CSV_PATH):
        self.cam = cam
        self.calib = Calibrator()
        self.mem: dict[int, TrackMem] = {}
        self.cam_buf = deque(maxlen=60)
        self.lock = threading.Lock()
        self.csv = open(csv_path, "w", newline="", encoding="utf-8") if csv_path else None
        self.writer = csv.writer(self.csv) if self.csv else None
        if self.writer:
            self.writer.writerow(["t", "radar_frame", "radar_id", "range_m", "range_near_m", "azimuth_deg", "radial_mps",
                                  "snr_db", "n_points", "radar_kind", "cam_id", "cam_class", "cam_conf",
                                  "cam_az_deg", "cam_range_est_m", "cam_clipped", "dt_s", "hits", "state", "fused_class",
                                  "lost_reason", "absorbed_by", "ego_mps", "abs_radial_mps"])
        self.last_fused = []
        self.ego = {"v": 0.0, "valid": False, "moving": False}   # carrier motion for the current radar frame (set by the caller)
        self.cam_lag = None        # EMA of (publish time - capture time) = detector latency, measured
        self.cam_period = None     # EMA of the gap between camera observations

    def push_camera(self, t, dets, frame=None, t_pub=None):
        """t — capture time of the frame; t_pub — the moment the detections are ready (after YOLO). The camera
        observation only becomes visible to the radar thread now, i.e. (t_pub - t) later than it was taken —
        on a Pi that is ~100-150 ms, more than MAX_DT_S, so the match window must expect the camera to lag."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if (frame is not None and cv2 is not None) else None
        with self.lock:
            if t_pub is not None:
                lag = max(0.0, t_pub - t)
                self.cam_lag = lag if self.cam_lag is None else 0.8 * self.cam_lag + 0.2 * lag
            if self.cam_buf:
                gap = t - self.cam_buf[-1][0]
                if 0 < gap < 2.0:
                    self.cam_period = gap if self.cam_period is None else 0.8 * self.cam_period + 0.2 * gap
            self.cam_buf.append((t, dets, gray))

    def sync_window(self):
        """How much older than the radar frame a camera observation may be and still count as simultaneous:
        the base budget + the detector latency + one camera period (the newest observation is at most that old)."""
        return min(SYNC_HARD_CAP_S, MAX_DT_S + (self.cam_lag or 0.0) + (self.cam_period or 0.0))

    def nearest_camera_t(self, t):
        with self.lock:
            best = min(self.cam_buf, key=lambda it: abs(it[0] - t), default=None)
            late = self.sync_window()
        if best is None:
            return None
        d = best[0] - t                                   # < 0: the camera observation is older than the radar frame
        return best if -late <= d <= MAX_DT_S else None

    def _nearest_camera(self, t):
        best = self.nearest_camera_t(t)
        if best is None:
            return None, None, None
        return best[1], best[0] - t, best[2]

    def _virtual_bbox(self, m: TrackMem, r_new, az_from_cam):
        """Box from radar when the camera can't see. Horizontal — track azimuth (from the camera point), width —
        from the range ratio. Vertical — from the edge model v = v_h + k/r learned while the camera could see;
        if there's no model — scale from the frame center (no scenario assumptions about camera height)."""
        x1, y1, x2, y2 = m.last_bbox
        scale = m.last_bbox_range / max(r_new, 0.3)
        if m.last_bbox_side_clipped and m.last_full_width_px > 0:
            bw = m.last_full_width_px * m.last_full_width_range / max(r_new, 0.3)   # don't scale a clipped width
        elif m.last_bbox_side_clipped:
            bw = (y2 - y1) * scale * 0.42                                          # human proportions
        else:
            bw = (x2 - x1) * scale
        cu = self.cam.u_of_cam_azimuth(az_from_cam)
        inv_new, inv_old = 1.0 / max(r_new, 0.3), 1.0 / max(m.last_bbox_range, 0.3)
        edges = []
        for which, v_old in (("top", y1), ("bottom", y2)):
            mdl = m.edge_model(which)
            if mdl is not None:
                a, b = mdl
                edges.append(a + b * inv_new)
            else:
                v_h = self.cam.h / 2
                edges.append(v_h + (v_old - v_h) * scale)
        ny1, ny2 = min(edges), max(edges)
        # the box doesn't extend past the frame: what's beyond the edge the camera wouldn't show anyway
        return (int(max(0, cu - bw / 2)), int(max(0, ny1)), int(min(self.cam.w - 1, cu + bw / 2)), int(min(self.cam.h - 1, ny2)))

    def on_radar(self, t, radar_frame, tracks):
        cam_dets, dt, gray = self._nearest_camera(t)
        cam_dets = cam_dets or []
        cam_view = [self.cam.cam_view(float(tr.x[0]), float(tr.x[1])) for tr in tracks]   # (az_from_cam, depth)

        # 1) matching: cost = geometry − history bonus; a clipped bbox means an inaccurate range
        cand = []
        for i, tr in enumerate(tracks):
            az_tr, depth_tr = cam_view[i]
            m = self.mem.get(tr.id)
            for j, d in enumerate(cam_dets):
                az_cam = self.cam.azimuth_of_u(d.cx)
                r_cam = self.cam.range_from_bbox(d.bh, d.obj_h_m)
                sig_r = CLIPPED_RANGE_SIGMA if self.cam.bbox_clipped(d.x1, d.y1, d.x2, d.y2) else RANGE_REL_SIGMA
                c_az = (az_tr - az_cam) / AZ_SIGMA_DEG
                c_r = (depth_tr - r_cam) / (sig_r * max(r_cam, 1.0))
                cost = math.sqrt(c_az ** 2 + c_r ** 2)
                if m is not None and m.cam_id == d.cam_id and m.hits > 0:
                    cost -= HISTORY_BONUS
                if cost < GATE:
                    cand.append((cost, i, j))
        cand.sort()
        used_t, used_c, matched = set(), set(), {}
        for cost, i, j in cand:
            if i in used_t or j in used_c:
                continue
            used_t.add(i); used_c.add(j); matched[i] = j

        # 2) track memory
        for i, j in matched.items():
            tr, d = tracks[i], cam_dets[j]
            m = self.mem.setdefault(tr.id, TrackMem(tr.id))
            m.hits += 1; m.last_t = t; m.last_cam_t = t; m.cam_id = d.cam_id
            m.classes[d.cls] += d.conf
            m.last_bbox = (d.x1, d.y1, d.x2, d.y2); m.last_bbox_range = tr.range_m
            m.last_bbox_side_clipped = self.cam.bbox_side_clipped(d.x1, d.y1, d.x2, d.y2)
            if not m.last_bbox_side_clipped:
                m.last_full_width_px, m.last_full_width_range = float(d.x2 - d.x1), tr.range_m
            m.last_az_from_cam = cam_view[i][0]
            m.reasons.clear()
            inv_r = 1.0 / max(tr.range_m, 0.3)
            m.edge_hist.append((inv_r, None if d.y1 <= 1 else float(d.y1), None if d.y2 >= self.cam.h - 2 else float(d.y2)))
            q = frame_quality(gray, m.last_bbox)
            if q is not None:
                m.ref_quality = q if m.ref_quality is None else {k: 0.8 * m.ref_quality[k] + 0.2 * q[k] for k in q}
            self.calib.add(self.cam.azimuth_of_u(d.cx) - self.cam.yaw, cam_view[i][0])
        alive = {tr.id for tr in tracks}
        for rid in list(self.mem):
            m = self.mem[rid]
            if rid in alive:
                m.last_t = t
                if t - m.last_cam_t > HOLD_MAX_S:
                    del self.mem[rid]
            elif t - m.last_t > FORGET_S:
                del self.mem[rid]

        # 3) state
        fused = []
        for i, tr in enumerate(tracks):
            m = self.mem.get(tr.id)
            confirmed = m is not None and m.hits >= PAIR_CONFIRM
            d = cam_dets[matched[i]] if i in matched else None
            az_from_cam, depth = cam_view[i]
            in_fov = abs(az_from_cam - self.cam.yaw) < self.cam.hfov / 2 - HOLD_FOV_MARGIN_DEG
            left_frame = False
            if confirmed and d is None and m.last_bbox_side_clipped:
                # the last box was against the side edge, and the azimuth moved further toward the edge — the object left the frame
                left_frame = abs(az_from_cam) > abs(m.last_az_from_cam) + 0.5
            if confirmed and d is not None:
                state = "both"
            elif confirmed and in_fov and not left_frame:
                state = "hold"
            elif confirmed:
                state = "out-of-frame"
            else:
                state = "radar-only"
            bbox, reason = None, ""
            if state == "both":
                bbox = (d.x1, d.y1, d.x2, d.y2)
            elif state == "hold" and m.last_bbox is not None:
                bbox = self._virtual_bbox(m, tr.range_m, az_from_cam)
                m.reasons.append(lost_reason(m.ref_quality, frame_quality(gray, bbox)))
                reason = max(set(m.reasons), key=list(m.reasons).count)
            fused.append({
                "radar_id": tr.id, "range_m": tr.range_m, "range_near_m": tr.range_near_m,
                "azimuth_deg": tr.azimuth_deg, "az_from_cam": az_from_cam, "depth_m": depth,
                "radial_mps": tr.radial_mps, "speed_mps": tr.speed_mps, "coasting": tr.coasting, "sigma_m": tr.sigma_xy_m,
                "cam_id": (d.cam_id if d is not None else m.cam_id) if confirmed else None,
                "fused_class": m.cls if confirmed else ("radar-only" if d is None else "pairing"),
                "hits": m.hits if m else 0, "matched_now": d is not None,
                # relative (sensor-frame) values above are what collision logic needs; these two say what the object itself does
                "ego_mps": self.ego["v"],
                "abs_radial_mps": tr.radial_mps + self.ego["v"] * math.cos(math.atan2(float(tr.x[0]), float(tr.x[1]))),
                "state": state, "bbox": bbox, "lost_reason": reason,
                "hold_s": (t - m.last_cam_t) if state == "hold" else 0.0,
                "absorbed_by": None, "x_m": float(tr.x[0]), "y_m": float(tr.x[1]),
            })

        # 4) absorbing duplicates: only tracks with no detection/memory of their own, measured now, at the same azimuth
        anchors = [f for f in fused if f["state"] in ("both", "hold")]
        for idx_f, f in enumerate(fused):
            if f["state"] != "radar-only" or f["matched_now"] or f["hits"] > 0 or f["coasting"]:
                continue
            for a in anchors:
                if math.hypot(f["x_m"] - a["x_m"], f["y_m"] - a["y_m"]) < MERGE_DIST_M and \
                        abs(radar.doppler_diff(f["radial_mps"], a["radial_mps"])) < MERGE_DV_MPS and \
                        abs(f["az_from_cam"] - a["az_from_cam"]) < 2 * AZ_SIGMA_DEG:
                    f["absorbed_by"] = a["radar_id"]
                    a["range_near_m"] = min(a["range_near_m"], f["range_near_m"])
                    break

        # 5) CSV
        if self.writer:
            for i, tr in enumerate(tracks):
                f = fused[i]
                d = cam_dets[matched[i]] if i in matched else None
                self.writer.writerow([round(t, 3), radar_frame, tr.id, round(tr.range_m, 2), round(f["range_near_m"], 2),
                                      round(tr.azimuth_deg, 1), round(tr.radial_mps, 2), round(tr.snr, 1), tr.n_points, tr.kind,
                                      d.cam_id if d else "", d.cls if d else "", round(d.conf, 2) if d else "",
                                      round(self.cam.azimuth_of_u(d.cx), 1) if d else "",
                                      round(self.cam.range_from_bbox(d.bh, d.obj_h_m), 2) if d else "",
                                      int(self.cam.bbox_clipped(d.x1, d.y1, d.x2, d.y2)) if d else "",
                                      round(dt, 3) if dt is not None else "", f["hits"], f["state"], f["fused_class"],
                                      f["lost_reason"], f["absorbed_by"] if f["absorbed_by"] is not None else "",
                                      round(f["ego_mps"], 2), round(f["abs_radial_mps"], 2)])
            for j, d in enumerate(cam_dets):
                if j not in used_c:
                    self.writer.writerow([round(t, 3), radar_frame, "", "", "", "", "", "", "", "", d.cam_id, d.cls,
                                          round(d.conf, 2), round(self.cam.azimuth_of_u(d.cx), 1),
                                          round(self.cam.range_from_bbox(d.bh, d.obj_h_m), 2),
                                          int(self.cam.bbox_clipped(d.x1, d.y1, d.x2, d.y2)),
                                          round(dt, 3) if dt is not None else "", 0, "cam-only", "cam-only", "", ""])
        self.last_fused = fused
        return fused, cam_dets, matched

    def apply_calibration(self):
        yaw = self.calib.yaw()
        if yaw is not None:
            self.cam.yaw = yaw
        return yaw

    def close(self):
        if self.csv:
            self.csv.close()


# ---------------------------------------------------------------- camera
def load_detector(kind=DETECTOR):
    weights = YOLO_WEIGHTS[kind]
    model = YOLO(weights)
    keep = YOLO_KEEP
    if kind == "yolo-world":
        names = list(WORLD_CLASSES)
        model.set_classes(names)          # must happen before an NCNN export below bakes in the class set
        keep = {i: (n, WORLD_CLASSES[n]) for i, n in enumerate(names)}
    if USE_NCNN:
        # NCNN export is validated here against yolov8n (the Pi-recommended detector); yolo-world's open-vocab
        # CLIP head may not export cleanly to NCNN via ultralytics — untested combination, use at your own risk
        ncnn_dir = os.path.splitext(weights)[0] + "_ncnn_model"
        # a size-specific export (e.g. yolov8n_ncnn_model_256x320) wins when it matches YOLO_IMGSZ: NCNN exports are
        # fixed-size, so lowering YOLO_IMGSZ in configs.json only speeds things up if a matching model dir exists
        sz = YOLO_IMGSZ if isinstance(YOLO_IMGSZ, (list, tuple)) else [YOLO_IMGSZ, YOLO_IMGSZ]
        sized_dir = f"{ncnn_dir}_{sz[0]}x{sz[1]}"
        if os.path.isdir(sized_dir):
            ncnn_dir = sized_dir
        if not os.path.isdir(ncnn_dir):
            print(f"Exporting {weights} to NCNN at imgsz={YOLO_IMGSZ} (needs internet the first time, to fetch the pnnx converter) ...")
            model.export(format="ncnn", imgsz=YOLO_IMGSZ)
        model = YOLO(ncnn_dir)
        print(f"YOLO running on: NCNN ({ncnn_dir}), imgsz={YOLO_IMGSZ}")
    else:
        model.to(YOLO_DEVICE)
        print(f"YOLO running on: {YOLO_DEVICE}")
    return model, keep


def yolo_detections(model, frame, t, keep=YOLO_KEEP):
    kwargs = dict(conf=YOLO_CONF, persist=True, verbose=False, tracker="bytetrack.yaml", imgsz=YOLO_IMGSZ)
    if not USE_NCNN:
        kwargs["device"] = YOLO_DEVICE    # the NCNN backend is CPU-only and doesn't take a device= override
    res = model.track(frame, **kwargs)[0]
    dets = []
    if res.boxes is None:
        return dets
    ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [-1] * len(res.boxes)
    for b, tid in zip(res.boxes, ids):
        cls_id = int(b.cls[0])
        if cls_id not in keep:
            continue
        name, h = keep[cls_id]
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        dets.append(CamDet(t, tid, name, float(b.conf[0]), x1, y1, x2, y2, h))
    return dets


# ---------------------------------------------------------------- rendering
def _range_label(img, x, y, text, color, scale=0.9):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    cv2.rectangle(img, (x - 3, y - th - 6), (x + tw + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 2)


def _dashed_rect(img, p1, p2, color, thick=2, dash=12):
    x1, y1 = p1; x2, y2 = p2
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        L = math.hypot(bx - ax, by - ay); n = max(int(L / dash), 1)
        for k in range(0, n, 2):
            s0, s1 = k / n, min((k + 1) / n, 1.0)
            cv2.line(img, (int(ax + (bx - ax) * s0), int(ay + (by - ay) * s0)),
                     (int(ax + (bx - ax) * s1), int(ay + (by - ay) * s1)), color, thick)


def draw_overlay(frame, fused, cam_dets, matched_idx, cam: CameraModel, tracks, radar_stale=False, fps=None):
    """Green box — both sensors; cyan — camera lost it, radar is tracking (dashed — radar on prediction);
    thin orange — camera only (range from bbox height, "≤" if the box is clipped by the edge);
    red circle — moving radar with no pair. The number on the box is the range to the nearest point (radar)."""
    img = frame.copy()
    if fps is not None:
        _range_label(img, cam.w - 130, 26, f"{fps:4.1f} fps", (200, 200, 200), 0.55)
    if radar_stale:
        _range_label(img, 8, 30, "RADAR LOST / STALE — camera only", (0, 60, 255), 0.7)
        fused = []
    fused_cam_ids = {f["cam_id"] for f in fused if f.get("state") == "both"}
    # horizontal spans of every object box shown this frame: a "radar only" dot landing in one of
    # these is (almost always) the same physical object, not a separate detection — the dot has no
    # real vertical position of its own (radar has no elevation, v is just frame-center), so only
    # the horizontal span is meaningful to compare against
    object_x_spans = [(d.x1, d.x2) for d in cam_dets if d.cam_id not in fused_cam_ids]
    for f in fused:
        if f.get("absorbed_by") is None and f.get("state") in ("both", "hold") and f.get("bbox"):
            object_x_spans.append((f["bbox"][0], f["bbox"][2]))
    nearest = None
    for d in cam_dets:
        if d.cam_id in fused_cam_ids:
            continue
        r_cam = cam.range_from_bbox(d.bh, d.obj_h_m)
        lim = "<=" if cam.bbox_clipped(d.x1, d.y1, d.x2, d.y2) else "~"
        cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), (0, 140, 255), 1)
        cv2.putText(img, f"{d.cls} (camera only)", (d.x1, max(d.y1 - 30, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
        _range_label(img, d.x1, max(d.y1 - 6, 24), f"{lim}{r_cam:.1f} m", (0, 140, 255), 0.7)
    for f in fused:
        if f.get("absorbed_by") is not None:
            continue
        st = f.get("state", "radar-only")
        r, vr = f["range_near_m"], f["radial_mps"]
        if st in ("both", "hold") and f.get("bbox"):
            x1, y1, x2, y2 = f["bbox"]
            color = (0, 255, 0) if st == "both" else (255, 200, 0)
            if st == "hold" and f.get("coasting"):
                _dashed_rect(img, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if st == "both" else 3)
            head = f"#{f['radar_id']} {f['fused_class']}" + ("  RADAR HOLD" if st == "hold" else "")
            rtxt = f"{r:.1f} m" + (f" ±{f['sigma_m']:.1f}" if st == "hold" and f["sigma_m"] > 0.15 else "")
            if y1 > 75:
                cv2.putText(img, head, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                _range_label(img, x1, y1 - 8, rtxt, color, 1.0)
            else:
                cv2.putText(img, head, (x1 + 6, y1 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                _range_label(img, x1 + 6, y1 + 100, rtxt, color, 1.0)
            sub = f"{vr:+.1f} m/s"
            if st == "both":
                d = next((c for c in cam_dets if c.cam_id == f["cam_id"]), None)
                if d is not None:
                    lim = "<=" if cam.bbox_clipped(d.x1, d.y1, d.x2, d.y2) else "~"
                    sub += f"   cam {lim}{cam.range_from_bbox(d.bh, d.obj_h_m):.1f} m"
            else:
                sub += f"   camera lost {f['hold_s']:.0f}s: {f.get('lost_reason') or '?'}"
            cv2.putText(img, sub, (x1, min(y2 + 18, cam.h - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if nearest is None or r < nearest[0]:
                nearest = (r, vr, f["fused_class"])
            continue
        if st == "out-of-frame":
            continue
        if SHOW_ONLY_INTERESTING and abs(vr) < MOVING_MPS and f.get("speed_mps", 0) < MOVING_MPS:
            continue
        u = int(cam.u_of_cam_azimuth(f["az_from_cam"])); v = int(cam.h / 2)
        if 0 <= u < cam.w:
            margin = RADAR_DOT_SUPPRESS_MARGIN_PX * cam.w / 640.0          # the constant was tuned for a 640-wide frame
            if any(x1 - margin <= u <= x2 + margin for x1, x2 in object_x_spans):
                continue   # same horizontal area as an object already shown — same physical thing, not a new detection
            cv2.circle(img, (u, v), 9, (0, 0, 255), 2)
            _range_label(img, u + 12, v + 6, f"{r:.1f} m", (0, 0, 255), 0.6)
            cv2.putText(img, f"radar only {vr:+.1f} m/s", (u + 12, v + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            if nearest is None or r < nearest[0]:
                nearest = (r, vr, "radar")
    if nearest is not None and not radar_stale:
        r, vr, cls = nearest
        closing = -vr
        ttc = r / closing if closing > 0.1 else None
        panel = f"NEAREST {cls}: {r:.1f} m   {'closing' if closing > 0.1 else 'receding' if closing < -0.1 else 'static'} {abs(closing):.1f} m/s"
        if ttc is not None:
            panel += f"   TTC {ttc:.1f} s"
        col = (0, 60, 255) if (ttc is not None and ttc < 3) else (255, 255, 255)
        _range_label(img, 8, 30, panel, col, 0.6)
    return img


def in_corridor(f):
    """Object inside the band the carrier is about to drive through (sensor frame: x right, y forward)."""
    az = math.radians(f["azimuth_deg"])
    x, y = f["range_m"] * math.sin(az), f["range_m"] * math.cos(az)
    return abs(x) <= CORRIDOR_HALF_WIDTH_M and 0 < y <= CORRIDOR_AHEAD_M


def interesting_ids(fused, ego=None):
    if not SHOW_ONLY_INTERESTING:
        return None
    moving = bool(ego and ego.get("moving"))
    return {f["radar_id"] for f in fused if f.get("absorbed_by") is None and
            (f.get("state") in ("both", "hold") or abs(f["radial_mps"]) >= MOVING_MPS or f.get("speed_mps", 0) >= MOVING_MPS
             or (moving and in_corridor(f)))}      # a stationary post in the path is an obstacle once we move


class EgoLink:
    """Reads $RDEGO,<v_mps>,<yaw_dps>,<heading_deg>,<quality>*cs lines (NMEA-style XOR checksum, like $RDALT) from the
    ESP32 carrying the BNO085 (+ GNSS later). Empty fields mean "unknown". speed() is None when unknown or stale so the
    radar's own estimate takes over; yaw_rate() is 0 when unknown. Can share the alert node's serial object."""

    def __init__(self, port, baud=115200, shared=None):
        self.ser, self.t_rx, self.v, self.w, self.heading, self.quality = None, None, None, 0.0, None, 0
        self.lock = threading.Lock()
        self.buf = b""
        if shared is not None:
            self.ser = shared
        elif port:
            try:
                import serial
                self.ser = serial.Serial(port, baud, timeout=0)
            except Exception as e:
                print(f"warning: ego link {port} not opened ({e}) - using the radar's own speed estimate")
        if self.ser is not None:
            threading.Thread(target=self._reader, daemon=True).start()

    @staticmethod
    def _checksum_ok(line):
        if not line.startswith("$") or "*" not in line:
            return False
        body, cs = line[1:].split("*", 1)
        x = 0
        for ch in body:
            x ^= ord(ch)
        try:
            return x == int(cs[:2], 16)
        except ValueError:
            return False

    def _reader(self):
        while True:
            try:
                chunk = self.ser.read(256)
            except Exception:
                time.sleep(0.5); continue
            if not chunk:
                time.sleep(0.01); continue
            self.buf += chunk
            while b"\n" in self.buf:
                raw, self.buf = self.buf.split(b"\n", 1)
                line = raw.decode("ascii", errors="ignore").strip()
                if not line.startswith("$RDEGO,") or not self._checksum_ok(line):
                    continue
                f = line[1:].split("*", 1)[0].split(",")
                try:
                    v = float(f[1]) if len(f) > 1 and f[1] else None
                    w = math.radians(float(f[2])) if len(f) > 2 and f[2] else 0.0
                    hd = float(f[3]) if len(f) > 3 and f[3] else None
                    q = int(f[4]) if len(f) > 4 and f[4] else 0
                except ValueError:
                    continue
                with self.lock:
                    self.v, self.w, self.heading, self.quality, self.t_rx = v, w, hd, q, time.time()

    def fresh(self):
        return self.t_rx is not None and (time.time() - self.t_rx) <= EGO_STALE_S

    def speed(self):
        with self.lock:
            return self.v if (self.fresh() and self.v is not None and self.quality > 0) else None

    def yaw_rate(self):
        with self.lock:
            return self.w if self.fresh() else 0.0


def display_scaled(img):
    """Shrink a window image by DISPLAY_SCALE (display only — nothing maps window pixels back to frame pixels)."""
    if DISPLAY_SCALE >= 1.0:
        return img
    return cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- live mode
def run_live(dump=None):
    if cv2 is None:
        raise SystemExit("pip install opencv-python")
    if YOLO is None:
        raise SystemExit("pip install ultralytics")
    radar.DUMP_FILE = dump or radar.DUMP_FILE
    radar.SHOW_WINDOW = False
    cfg = radar.parse_cfg(radar.CFG_FILE)
    model_r = None
    try:
        import joblib; model_r = joblib.load(radar.MODEL_PATH)      # the team's own model
    except Exception:
        pass
    pipe = radar.Pipeline(cfg, model_r, radar.EGO_SPEED_MPS)
    # DirectShow exists only on Windows; on Linux/Pi use V4L2 (CAP_DSHOW constant exists everywhere, so hasattr() is not a valid test)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"camera {CAMERA_INDEX} did not open — check CAMERA_INDEX in configs.json (ls /dev/video*)")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # newest frame, not a 3-4 frame old queue: the capture stamp below must be honest
    if CAM_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # before the size: V4L2 renegotiates the format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC)) & 0xFFFFFFFF
    fcc_s = fcc.to_bytes(4, "little").decode("ascii", errors="replace").strip("\x00") or "?"
    print(f"camera: {w}x{h} {fcc_s} @ {cap.get(cv2.CAP_PROP_FPS) or 0:.0f} fps nominal · YOLO every {YOLO_EVERY_N} frame(s)")
    if CAM_MJPG and fcc_s != "MJPG":
        print(f"warning: CAM_MJPG=1 but the camera negotiated {fcc_s} — no MJPG at {w}x{h} (v4l2-ctl --list-formats-ext)")
    cam = CameraModel(w, h)
    fus = Fusion(cam)
    yolo, keep = load_detector()

    alert_engine = None
    if ALERTS_ENABLED:
        sinks = [alerts.ConsoleAlertSink()]
        if ALERT_SERIAL_PORT:
            sinks.append(alerts.SerialAlertSink(ALERT_SERIAL_PORT, ALERT_BAUD))
        alert_engine = alerts.AlertEngine(sinks, resend_s=ALERT_RESEND_S, radar_only_confirm_s=RADAR_ONLY_CONFIRM_S,
                                          proximity_warn_m=PROXIMITY_WARN_M, proximity_critical_m=PROXIMITY_CRITICAL_M,
                                          closing_speed_mps=CLOSING_SPEED_ALERT_MPS)
    shared = None
    if EGO_SERIAL_PORT == "same" and alert_engine is not None:
        shared = next((s.ser for s in sinks if getattr(s, "ser", None) is not None), None)
    ego_link = EgoLink(None if EGO_SERIAL_PORT == "same" else EGO_SERIAL_PORT, EGO_BAUD, shared=shared)
    t_start = time.time()
    clock = lambda: time.time() - t_start
    snap = {"t": -1e9, "fused": [], "dets": [], "matched": {}, "tracks": [], "rdets": [], "rkinds": [], "radar_fps": 0.0,
            "ego": {"v": 0.0, "valid": False, "moving": False, "n_inliers": 0, "source": "-"}}
    state = {"snap": snap, "error": None}
    stop = threading.Event()

    def radar_thread():
        try:
            buffer = b""
            period = cfg.get("frame_period_s") or 0.1
            t_next = None
            radar_fps = radar.FpsMeter()
            for chunk in radar.byte_source():
                if stop.is_set():
                    break
                buffer += chunk
                gen = radar.frames_from_bytes(buffer)
                while True:
                    try:
                        fr = next(gen)
                    except StopIteration as s:
                        buffer = s.value or b""
                        break
                    if fr is None:
                        continue
                    if dump:
                        t_next = clock() if t_next is None else t_next + period
                        while clock() < t_next:
                            time.sleep(0.005)
                    t_now = clock()
                    out = pipe.process(fr, ego_speed=ego_link.speed(), yaw_rate=ego_link.yaw_rate())
                    fus.ego = out["ego"]
                    radar_fps.tick()
                    best = fus.nearest_camera_t(t_now)
                    dt_sync = (best[0] - t_now) if best is not None else 0.0
                    filtered = radar_filter.sync_and_filter(out["tracks"], dt_sync, max_dt=fus.sync_window())
                    fused, dets, matched = fus.on_radar(t_now, fr["frame"], filtered)
                    state["snap"] = {"t": t_now, "fused": fused, "dets": dets, "matched": matched,
                                     "tracks": filtered, "rdets": out["dets"], "rkinds": out["kinds"],
                                     "radar_fps": radar_fps.fps, "ego": out["ego"]}
        except BaseException as e:                               # a thread dying shouldn't be silent
            state["error"] = f"{type(e).__name__}: {e}"
            print("RADAR STOPPED:", state["error"], flush=True)

    threading.Thread(target=radar_thread, daemon=True).start()
    n = 0
    dets = []                                                 # last YOLO result, reused on the frames YOLO skips
    fusion_fps = radar.FpsMeter()                             # camera loop rate
    det_fps = radar.FpsMeter()                                # YOLO rate (what actually limits detection latency)
    t_report = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = clock()
            if n % YOLO_EVERY_N == 0:
                # detection frame: fresh boxes + the camera-side bookkeeping (frame quality, association history).
                # Skipped frames deliberately don't call push_camera: the fusion then keeps matching the radar against
                # the last *real* camera observation and its true timestamp instead of a re-stamped stale one.
                dets = yolo_detections(yolo, frame, t, keep)
                fus.push_camera(t, dets, frame, t_pub=clock())
                det_fps.tick()
            n += 1
            fusion_fps.tick()
            if t - t_report >= 5.0:                          # headless-friendly rate report (the windows show it too)
                t_report = t
                print(f"rate: loop {fusion_fps.fps:4.1f} fps · YOLO {det_fps.fps:4.1f}/s · radar {state['snap']['radar_fps']:4.1f} fps"
                      f" · cam lag {(fus.cam_lag or 0) * 1000:3.0f} ms · sync window {fus.sync_window() * 1000:3.0f} ms"
                      f" · {radar.ego_label(state['snap']['ego'])}", flush=True)
            if n % 50 == 0:
                yaw = fus.apply_calibration()
                if yaw is not None:
                    print(f"calibration: yaw radar↔camera = {yaw:+.1f}° ({len(fus.calib.pairs)} pairs)")
            s = state["snap"]
            stale = (t - s["t"]) > RADAR_STALE_S or state["error"] is not None
            if alert_engine is not None:
                alert_engine.evaluate(t, s["fused"], dets, cam, stale)
            if SHOW_WINDOW:
                img = draw_overlay(frame, s["fused"], dets, s["matched"], cam, s["tracks"], radar_stale=stale,
                                   fps=fusion_fps.fps if SHOW_FPS else None)
                cv2.imshow("fusion", display_scaled(img))
                cv2.imshow("radar", display_scaled(radar.render(s["rdets"], s["rkinds"], [] if stale else s["tracks"], pipe.bg,
                                                 meters=radar.DRAW_METERS, only_ids=interesting_ids(s["fused"], s["ego"]),
                                                 title="RADAR STALE" if stale else "",
                                                 fps=s["radar_fps"] if SHOW_FPS else None, ego=s["ego"])))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        stop.set(); cap.release(); fus.close()
        if alert_engine is not None:
            alert_engine.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("CSV:", CSV_PATH)


# ---------------------------------------------------------------- self-test without a camera
def selftest(dump, true_yaw_deg=4.0, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    cfg = {"range_res_m": 0.044, "doppler_res_mps": 0.125, "num_doppler_bins": 16, "frame_period_s": 0.1,
           "doppler_period_mps": 2.0}
    radar.Track._next_id = 1
    pipe = radar.Pipeline(cfg, None, 0.0, use_background=True)
    truth_cam = CameraModel(1280, 720, hfov_deg=70.0, yaw_deg=true_yaw_deg, radar_to_cam={"right": 0, "forward": 0})
    cam = CameraModel(1280, 720, hfov_deg=70.0, yaw_deg=0.0, radar_to_cam={"right": 0, "forward": 0})
    csv_path = "fusion_selftest.csv"
    fus = Fusion(cam, csv_path)
    frames = [f for f in radar.frames_from_bytes(open(dump, "rb").read()) if f]
    stats = defaultdict(int)
    for k, fr in enumerate(frames):
        out = pipe.process(fr)
        t = out["t"]
        tracks = radar_filter.sync_and_filter(out["tracks"], 0.0)
        dets = []
        for tr in tracks:
            if tr.range_m < 0.7 or rng.random() < 0.2:
                continue
            az, _ = truth_cam.cam_view(float(tr.x[0]), float(tr.x[1]))
            u = truth_cam.u_of_cam_azimuth(az) + rng.normal(0, 12)
            bh = truth_cam.f * 1.7 / tr.range_m * rng.uniform(0.8, 1.2)
            v = 360 + 0.3 * bh
            dets.append(CamDet(t + rng.normal(0, 0.03), 100 + tr.id, "person", 0.8,
                               int(u - bh * 0.2), int(v - bh / 2), int(u + bh * 0.2), int(v + bh / 2)))
        if rng.random() < 0.15:
            dets.append(CamDet(t, 999, "car", 0.5, 100, 300, 260, 420, 1.5))
        fus.push_camera(t, dets)
        fused, cam_dets, matched = fus.on_radar(t, fr["frame"], tracks)
        stats["radar_frames"] += 1
        stats["tracks"] += len(out["tracks"])
        stats["rejected"] += len(out["tracks"]) - len(tracks)
        stats["matched_now"] += sum(1 for f in fused if f["matched_now"])
        stats["fused_confirmed"] += sum(1 for f in fused if f["cam_id"] is not None)
        stats["class_person"] += sum(1 for f in fused if f["fused_class"] == "person")
        stats["absorbed"] += sum(1 for f in fused if f["absorbed_by"] is not None)
        if k == 40:
            yaw = fus.apply_calibration()
            stats["yaw_est_at_40"] = yaw if yaw is not None else float("nan")
    fus.close()
    yaw_final = fus.calib.yaw()
    import pandas as pd
    df = pd.read_csv(csv_path)
    dup = df[df.radar_id.notna()].duplicated(subset=["radar_frame", "radar_id"]).sum()
    res = {"radar_frames": stats["radar_frames"], "total_tracks (frame-tracks)": stats["tracks"],
           "rejected_tracks (frame-tracks)": stats["rejected"],
           "matched_with_camera_now": stats["matched_now"], "confirmed (frame-tracks)": stats["fused_confirmed"],
           "got_class_person": stats["class_person"], "duplicates_absorbed (frame-tracks)": stats["absorbed"],
           "true_yaw": true_yaw_deg, "yaw_estimate_at_frame_40": round(stats["yaw_est_at_40"], 2),
           "yaw_estimate_at_end": round(yaw_final, 2) if yaw_final is not None else None,
           "csv_rows": len(df), "duplicates (radar_frame, radar_id)": int(dup),
           "cam_only_rows": int((df.state == "cam-only").sum()), "objects_in_memory_at_end": len(fus.mem)}
    if verbose:
        for k_, v_ in res.items():
            print(f"  {k_}: {v_}")
    return res, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.dump or "radar_dump_hand.bin")
    else:
        run_live(a.dump)
