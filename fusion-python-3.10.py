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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

import iwr1642_live as radar
import radar_filter

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ---------------------------------------------------------------- SETTINGS
CAMERA_INDEX = 0
DETECTOR = "yolo-world"       # "yolov8n" — COCO, fast; "yolo-world" — open vocabulary (WORLD_CLASSES),
                           # first run downloads weights + CLIP (~340 MB), ~0.25 s/frame on CPU
YOLO_WEIGHTS = {"yolov8n": "yolov8n.pt", "yolo-world": "yolov8s-worldv2.pt"}
YOLO_CONF = 0.4
YOLO_KEEP = {0: ("person", 1.70), 1: ("bicycle", 1.10), 2: ("car", 1.50), 3: ("motorcycle", 1.20),
             5: ("bus", 3.00), 7: ("truck", 3.00), 16: ("dog", 0.55), 17: ("horse", 1.60),
             18: ("sheep", 0.90), 19: ("cow", 1.40), 39: ("bottle", 0.25), 41: ("cup", 0.12)}
WORLD_CLASSES = {"person": 1.70, "car": 1.50, "truck": 3.00, "tractor": 2.80, "dog": 0.55, "cow": 1.40,
                 "aluminum can": 0.12, "bottle": 0.25, "chair": 0.90, "pole": 2.00, "box": 0.40}

CAM_HFOV_DEG = 70.0        # camera horizontal field of view
CAM_YAW_DEG = 0.0          # yaw = radar_azimuth − camera_azimuth for the same object (median over pairs);
                           # positive if the radar axis is rotated LEFT of the camera axis. Refined by calibration.
RADAR_TO_CAMERA_M = {"right": 0.0, "up": 0.0, "forward": 0.0}   # where the radar is relative to the lens; read from meta.json
MAX_DT_S = 0.15            # allowed desync between camera and radar frames
AZ_SIGMA_DEG = 4.0         # expected azimuth error between sensors
RANGE_REL_SIGMA = 0.35     # relative range error from bbox height (±35 %)
CLIPPED_RANGE_SIGMA = 1.2  # ...if the bbox hits the frame edge — height is clipped, range is only an upper bound
GATE = 3.0                 # matching threshold in sigmas
HISTORY_BONUS = 1.0        # cost discount if a detection with the same cam_id already matched this track
PAIR_CONFIRM = 5           # matches needed for a track to become an "object with a class"
FORGET_S = 2.0             # track memory without radar — this many seconds
HOLD_MAX_S = 30.0          # camera lost the object, radar is tracking: how many seconds to hold the box
HOLD_FOV_MARGIN_DEG = 8.0  # object is "in frame" if azimuth is within FOV with margin ≥ 2·AZ_SIGMA
MERGE_DIST_M = 0.9         # an unpaired radar track closer than this to an object...
MERGE_DV_MPS = 0.8         # ...with a similar speed (wraparound-aware) and the same azimuth — is part of it, not a separate object
SHOW_ONLY_INTERESTING = True
MOVING_MPS = 0.25
RADAR_STALE_S = 0.5        # radar hasn't updated for this long — consider it lost (banner, tracks not drawn)
CSV_FOLDER = "logs"
CSV_PATH = os.path.join(CSV_FOLDER, f"fusion_{datetime.now():%Y%m%d_%H%M%S}.csv")
SHOW_WINDOW = True

Q_SHARP_DROP = 0.45        # sharpness below 45 % of the reference
Q_CONTRAST_DROP = 0.45     # contrast below 45 % of the reference


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
                                  "lost_reason", "absorbed_by"])
        self.last_fused = []

    def push_camera(self, t, dets, frame=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if (frame is not None and cv2 is not None) else None
        with self.lock:
            self.cam_buf.append((t, dets, gray))

    def nearest_camera_t(self, t):
        with self.lock:
            best = min(self.cam_buf, key=lambda it: abs(it[0] - t), default=None)
        return best if best is not None and abs(best[0] - t) <= MAX_DT_S else None

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
                                      f["lost_reason"], f["absorbed_by"] if f["absorbed_by"] is not None else ""])
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
    model = YOLO(YOLO_WEIGHTS[kind])
    if kind == "yolo-world":
        names = list(WORLD_CLASSES)
        model.set_classes(names)
        return model, {i: (n, WORLD_CLASSES[n]) for i, n in enumerate(names)}
    return model, YOLO_KEEP


def yolo_detections(model, frame, t, keep=YOLO_KEEP):
    res = model.track(frame, conf=YOLO_CONF, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
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


def draw_overlay(frame, fused, cam_dets, matched_idx, cam: CameraModel, tracks, radar_stale=False):
    """Green box — both sensors; cyan — camera lost it, radar is tracking (dashed — radar on prediction);
    thin orange — camera only (range from bbox height, "≤" if the box is clipped by the edge);
    red circle — moving radar with no pair. The number on the box is the range to the nearest point (radar)."""
    img = frame.copy()
    if radar_stale:
        _range_label(img, 8, 30, "RADAR LOST / STALE — camera only", (0, 60, 255), 0.7)
        fused = []
    fused_cam_ids = {f["cam_id"] for f in fused if f.get("state") == "both"}
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


def interesting_ids(fused):
    if not SHOW_ONLY_INTERESTING:
        return None
    return {f["radar_id"] for f in fused if f.get("absorbed_by") is None and
            (f.get("state") in ("both", "hold") or abs(f["radial_mps"]) >= MOVING_MPS or f.get("speed_mps", 0) >= MOVING_MPS)}


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
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam = CameraModel(w, h)
    fus = Fusion(cam)
    yolo, keep = load_detector()
    t_start = time.time()
    clock = lambda: time.time() - t_start
    snap = {"t": -1e9, "fused": [], "dets": [], "matched": {}, "tracks": [], "rdets": [], "rkinds": []}
    state = {"snap": snap, "error": None}
    stop = threading.Event()

    def radar_thread():
        try:
            buffer = b""
            period = cfg.get("frame_period_s") or 0.1
            t_next = None
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
                    out = pipe.process(fr)
                    best = fus.nearest_camera_t(t_now)
                    dt_sync = (best[0] - t_now) if best is not None else 0.0
                    filtered = radar_filter.sync_and_filter(out["tracks"], dt_sync)
                    fused, dets, matched = fus.on_radar(t_now, fr["frame"], filtered)
                    state["snap"] = {"t": t_now, "fused": fused, "dets": dets, "matched": matched,
                                     "tracks": filtered, "rdets": out["dets"], "rkinds": out["kinds"]}
        except BaseException as e:                               # a thread dying shouldn't be silent
            state["error"] = f"{type(e).__name__}: {e}"
            print("RADAR STOPPED:", state["error"], flush=True)

    threading.Thread(target=radar_thread, daemon=True).start()
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = clock()
            dets = yolo_detections(yolo, frame, t, keep)
            fus.push_camera(t, dets, frame)
            n += 1
            if n % 50 == 0:
                yaw = fus.apply_calibration()
                if yaw is not None:
                    print(f"calibration: yaw radar↔camera = {yaw:+.1f}° ({len(fus.calib.pairs)} pairs)")
            s = state["snap"]
            stale = (t - s["t"]) > RADAR_STALE_S or state["error"] is not None
            if SHOW_WINDOW:
                img = draw_overlay(frame, s["fused"], dets, s["matched"], cam, s["tracks"], radar_stale=stale)
                cv2.imshow("fusion", img)
                cv2.imshow("radar", radar.render(s["rdets"], s["rkinds"], [] if stale else s["tracks"], pipe.bg,
                                                 meters=radar.DRAW_METERS, only_ids=interesting_ids(s["fused"]),
                                                 title="RADAR STALE" if stale else ""))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        stop.set(); cap.release(); fus.close()
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
