"""fusion_offline.py — прогон слияния по папке record_sync (rec_*/): видео + радар на одной оси времени.

    python fusion_offline.py rec_20260915_180817 [--out fused.mp4] [--meters 8] [--detector yolo-world]
                              [--blind a,b | --miss a,b | --occlude a,b]

Имитации потери камеры (секунды a..b):
  --blind    весь кадр размыт, детекций нет            → ожидаем причину «haze / smoke / defocus»
  --miss     кадр чистый, детекций нет                 → «detector miss»
  --occlude  над РЕАЛЬНЫМ человеком (YOLO на чистом кадре) нарисован «лист», детектор работает по кадру с листом
             → «occluded»; заодно меряется IoU виртуальной рамки с правдой
Время камеры — camera_times.csv; время радара — radar_times.csv (кусок байтов, в котором лежит последний байт кадра).
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath('E:\Kuliah\Skoltech - Engineering Systems\Innovation Workshops\fusion\radar_pack')))
import iwr1642_live as radar
import fusion as F


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
            k = np.searchsorted(offs, last_byte, side="right") - 1        # кусок, содержащий последний байт кадра
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec"); ap.add_argument("--out", default=None); ap.add_argument("--meters", type=int, default=10)
    ap.add_argument("--blind"); ap.add_argument("--miss"); ap.add_argument("--occlude")
    ap.add_argument("--detector", default=None)
    a = ap.parse_args()
    rec = a.rec.rstrip("/\\")
    win = lambda s: tuple(float(x) for x in s.split(",")) if s else None
    blind, miss, occl = win(a.blind), win(a.miss), win(a.occlude)

    meta = json.load(open(os.path.join(rec, "meta.json"), encoding="utf-8"))
    cfg_path = meta.get("cfg_file", radar.CFG_FILE)
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(cfg_path))
    cfg = radar.parse_cfg(cfg_path)
    rc_cm = meta.get("radar_to_camera_cm", {}) or {}
    radar_to_cam = {k: rc_cm.get(k, 0.0) / 100.0 for k in ("right", "up", "forward")}
    rframes = radar_frames_with_time(rec)
    ctimes = camera_times(rec)
    cap = cv2.VideoCapture(os.path.join(rec, "camera.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"радар: {len(rframes)} кадров, {rframes[0]['t']:.2f}–{rframes[-1]['t']:.2f} с · камера: {len(ctimes)} кадров {w}x{h}"
          f" · радар относительно камеры {radar_to_cam} · период Доплера {cfg.get('doppler_period_mps')}")

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
        while ri < len(rframes) and rframes[ri]["t"] <= t:                   # радар до этого момента
            fr = rframes[ri]; ri += 1
            out = pipe.process(fr)
            fused, dets, matched = fus.on_radar(fr["t"], fr["frame"], out["tracks"])
            latest.update(fused=fused, dets=dets, matched=matched, tracks=out["tracks"], rdets=out["dets"], rkinds=out["kinds"])
            stats["radar_frames"] += 1
            stats["matched_frames"] += any(f["matched_now"] for f in fused)
            stats["confirmed_frames"] += any(f["cam_id"] is not None for f in fused)
        in_blind = bool(blind and blind[0] <= t <= blind[1])
        in_miss = bool(miss and miss[0] <= t <= miss[1])
        in_occl = bool(occl and occl[0] <= t <= occl[1])
        truth_box = None
        if in_occl:
            truth = [d for d in F.yolo_detections(yolo, frame, t, keep) if d.cls == "person"]   # правда — до листа
            if truth:
                held = [f for f in latest["fused"] if f.get("state") in ("both", "hold") and f.get("bbox")]
                if held:                                       # правда — человек, ближайший к удерживаемой рамке
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
                print(f"  t={t:5.1f}s  yaw радар↔камера = {yaw:+.1f}° по {len(fus.calib.pairs)} парам")
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
                cv2.rectangle(left, truth_box[:2], truth_box[2:], (255, 255, 255), 1)   # правда — тонкой белой
        cv2.putText(left, f"t={t:5.1f}s  yaw={cam.yaw:+.1f}", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        right = radar.render(latest["rdets"], latest["rkinds"], latest["tracks"], pipe.bg, meters=a.meters,
                             title=f"radar t={t:.1f}s", only_ids=F.interesting_ids(latest["fused"]))
        canvas = np.zeros((max(h, right.shape[0]), w + right.shape[1], 3), dtype=np.uint8)
        canvas[:h, :w] = left; canvas[:right.shape[0], w:] = right
        vw.write(canvas)
    vw.release(); fus.close()
    yaw = fus.calib.yaw()
    print("\nсводка:")
    print(f"  кадров камеры {stats['cam_frames']}, с человеком по YOLO {stats['cam_person']} ({100*stats['cam_person']/max(stats['cam_frames'],1):.0f} %)")
    print(f"  кадров радара {stats['radar_frames']}: совпадение сейчас {stats['matched_frames']}, с подтверждённым объектом {stats['confirmed_frames']}")
    print(f"  кадров, где рамку держал радар: {stats['hold_frames']}; поглощено дублей: {stats['absorbed']}; вышло из кадра (кадро-треков): {stats['out_of_frame']}")
    for mode, cnt in reasons.items():
        print(f"  причины потери камеры [{mode}]: {dict(cnt)}")
    if ious:
        ious = np.array(ious)
        print(f"  виртуальная рамка vs правда (occlude): IoU средн {ious.mean():.2f}, медиана {np.median(ious):.2f}, доля IoU<0.5: {(ious<0.5).mean():.0%} (n={len(ious)})")
    print(f"  разворот радар↔камера (калибровка): {yaw:+.1f}° по {len(fus.calib.pairs)} парам" if yaw is not None else "  калибровка: мало пар")
    print(f"  объектов в памяти в конце: {len(fus.mem)}; {[(m.radar_id, m.cam_id, m.cls, m.hits) for m in fus.mem.values() if m.hits >= F.PAIR_CONFIRM]}")
    print(f"  видео: {out_path}; CSV: {os.path.join(rec, 'fusion.csv')}")


if __name__ == "__main__":
    main()
