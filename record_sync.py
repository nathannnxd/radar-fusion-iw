"""record_sync.py — simultaneous recording of the laptop camera, the IWR1642 radar and the ESP32+IMU ($EGOVEL)
on a shared clock.

    python record_sync.py [--tag standstill] [--no-camera] [--detect]

Exit — the q key in the camera window, or Ctrl+C. Output: folder rec_<time>[_<tag>]/ :
    camera.mp4          video from the camera
    camera_times.csv    frame number -> recording time (s from start)
    radar.bin           raw radar stream (same as dump_radar.py)
    radar_times.csv     byte offset -> time (s from start) — used to align radar frames with the video
    run.jsonl.gz        ONE gzip JSON-lines log of everything on the shared clock (EGO_MOTION.md §6):
                          {"kind":"meta", ...}                                  first record: ports, cfg, tag, calibration keys
                          {"kind":"radar","t":..,"frame":N,"b64":"..."}         one complete TI packet (base64) per radar frame
                          {"kind":"ego","t":..,"line":"$EGOVEL,..."}            raw $EGOVEL sentence + Pi receive time
                          {"kind":"cam","t":..,"frame":N[,"dets":[...]]}        camera frame time (+ detections with --detect)
    meta.json           parameters: ports, resolution, FPS, radar-camera distance

Does no analysis — only records. Processing happens later: fusion_offline.py --log rec_*/run.jsonl.gz
(or the old way: fusion_offline.py rec_* over the video + radar.bin). The .bin/.csv dump is kept unchanged so the
old replay and dump_radar.py-style tools keep working; the .jsonl.gz is the input of the ego-motion tools
(tools/calib_time_offset.py, tools/calib_mount.py — see tools/drills.md for which drill produces what).
"""
import argparse
import base64
import csv
import gzip
import json
import os
import threading
import time
from datetime import datetime

import cv2

try:
    import serial
except ImportError:
    serial = None

# ---------------------------------------------------------------- SETTINGS
import json as _json
_cfg = _json.load(open("configs.json"))
CAMERA_INDEX = _cfg["CAMERA_INDEX"]
CAM_WIDTH, CAM_HEIGHT = 640, 480
CLI_PORT = _cfg["CLI_PORT"]
DATA_PORT = _cfg["DATA_PORT"]
CFG_FILE = _cfg["RADAR_CONFIG"]
EGO_SERIAL_PORT = _cfg.get("EGO_SERIAL_PORT", "")   # ESP32+IMU UART (EGO_VELOCITY.md); "" — no $EGOVEL records
EGO_BAUD = _cfg.get("EGO_BAUD", 115200)
SEND_CFG = True             # False if the radar is already streaming
RADAR_TO_CAMERA_CM = {"right": 0.0, "up": 0.0, "forward": 0.0}   # where the radar is relative to the camera lens
MAX_SECONDS = 600           # safety net: stop after 10 minutes (the 10-min standstill drill fits exactly)
SHOW_PREVIEW = True         # False — no window (Jupyter / headless); stop via a STOP file in the launch folder or MAX_SECONDS
CALIB_KEYS = ("LEVER_ARM_X_M", "LEVER_ARM_Y_M", "MOUNT_YAW_DEG", "EGO_TIME_OFFSET_S", "EGO_GYRO_BIAS_DPS")   # EGO_MOTION.md §3

OUT_DIR = f"rec_{datetime.now():%Y%m%d_%H%M%S}"
MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_LEN = 40


class JsonlLog:
    """One gzip JSON-lines file shared by the radar / ego / camera threads; one line per record."""

    def __init__(self, path):
        self.f = gzip.open(path, "wt", encoding="utf-8")
        self.lock = threading.Lock()
        self.n = 0

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        with self.lock:
            self.f.write(line + "\n")
            self.n += 1

    def close(self):
        with self.lock:
            self.f.close()


def send_config():
    # NOTE: keep the returned CLI port OPEN while reading the data port. On Linux the XDS110 resets the
    # data port to 115200 the moment the CLI port is closed (Windows doesn't care) -> garbage instead of frames.
    ser = serial.Serial(CLI_PORT, 115200, timeout=1)
    with open(CFG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("%"):
                ser.write((line + "\n").encode())
                time.sleep(0.05)
    print("✓ config sent to", CLI_PORT)
    return ser


def split_frames(buf):
    """Complete TI packets at the front of buf -> (list of packet bytes, unprocessed remainder). Same framing rule as
    iwr1642_live.frames_from_bytes (MAGIC, total length at byte 12) — duplicated here so the recorder stays free of the
    pipeline's imports (pandas, the model, configs) and cannot break the .bin dump if the pipeline does."""
    out = []
    while True:
        i = buf.find(MAGIC)
        if i < 0:
            return out, buf[-7:] if len(buf) > 7 else buf
        buf = buf[i:]
        if len(buf) < HEADER_LEN:
            return out, buf
        total = int.from_bytes(buf[12:16], "little")
        if total < HEADER_LEN or total > 65536:
            buf = buf[8:]
            continue
        if len(buf) < total:
            return out, buf
        out.append(buf[:total])
        buf = buf[total:]


def radar_writer(stop, clock, out_dir, stats, log=None):
    """Writes bytes from the data port to radar.bin and the time of each chunk to radar_times.csv; additionally every
    complete frame goes to the .jsonl.gz as base64 with the time of the chunk that completed it (the same rule
    fusion_offline.radar_frames_with_time uses for the .bin, so both replays see identical timestamps)."""
    with serial.Serial(DATA_PORT, 921600, timeout=0.01) as ser, \
         open(os.path.join(out_dir, "radar.bin"), "wb") as fb, \
         open(os.path.join(out_dir, "radar_times.csv"), "w", newline="") as ft:
        w = csv.writer(ft); w.writerow(["byte_offset", "t_s"])
        offset, pending = 0, b""
        while not stop.is_set():
            chunk = ser.read(max(1, ser.in_waiting))     # short reads: timestamp ±10 ms, not ±100
            if chunk:
                t = clock()
                fb.write(chunk)
                w.writerow([offset, f"{t:.4f}"])
                offset += len(chunk)
                stats["radar_bytes"] = offset
                stats["radar_frames"] += chunk.count(MAGIC)
                if log is not None:
                    frames, pending = split_frames(pending + chunk)
                    for pkt in frames:
                        log.write({"kind": "radar", "t": round(t, 4), "frame": int.from_bytes(pkt[20:24], "little"),
                                   "b64": base64.b64encode(pkt).decode("ascii")})
    print("radar: stopped")


def ego_writer(stop, clock, stats, log):
    """Minimal $EGOVEL reader: raw sentence text + Pi receive time into the .jsonl.gz, nothing parsed — parsing and
    checksum happen at replay through ego_velocity.EgoVelocityReader, so a reader bug never costs a recording.
    Deliberately does not import ego_velocity / iwr1642_live: the recorder must not depend on the pipeline."""
    try:
        # exclusive: recording while the live pipeline holds the same port would split the $EGOVEL byte
        # stream between the two readers (both then see truncated sentences) — fail loudly instead
        ser = serial.Serial(EGO_SERIAL_PORT, EGO_BAUD, timeout=0.02, exclusive=True)
    except Exception as e:
        print(f"⚠️ ego-velocity port {EGO_SERIAL_PORT} not opened ({e}) — no $EGOVEL records")
        return
    buf = b""
    with ser:
        while not stop.is_set():
            chunk = ser.read(max(1, ser.in_waiting))
            if not chunk:
                continue
            t = clock()                                   # 50 Hz lines, ~40 B each: one line per chunk, stamp ±2 ms
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.strip().decode("ascii", "replace")
                if text.startswith("$"):
                    log.write({"kind": "ego", "t": round(t, 4), "line": text})
                    stats["ego_lines"] += 1
            if len(buf) > 4096:
                buf = buf[-256:]                          # a stuck/noisy line with no '\n' — don't grow forever
    print("ego: stopped")


def load_detector_optional():
    """--detect: YOLO from the fusion module, loaded lazily (the only heavy import here, and only when asked).
    Returns (module, model, keep) or None — a missing/broken detector never stops a recording."""
    try:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("fusion_live", os.path.join(here, "fusion-python-3.10.py"))
        F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
        model, keep = F.load_detector(F.DETECTOR)
        print(f"✓ detector {F.DETECTOR} loaded — camera detections go to run.jsonl.gz")
        return F, model, keep
    except Exception as e:
        print(f"⚠️ detector not loaded ({e}) — camera records without detections")
        return None


def main():
    global OUT_DIR, SHOW_PREVIEW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="drill name, appended to the folder name and stored in the log (tools/drills.md)")
    ap.add_argument("--no-camera", action="store_true", help="radar + IMU only (headless Pi on the tractor)")
    ap.add_argument("--detect", action="store_true", help="run the fusion detector and log camera detections")
    ap.add_argument("--no-preview", action="store_true", help="no OpenCV window")
    a = ap.parse_args()
    tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in a.tag.strip())
    if tag:
        OUT_DIR += f"_{tag}"
    if a.no_preview:
        SHOW_PREVIEW = False
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = writer = None
    w = h = 0; fps_nominal = 0.0
    if not a.no_camera:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        if not cap.isOpened():
            raise SystemExit(f"camera {CAMERA_INDEX} did not open (use --no-camera for a radar+IMU-only run)")
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera is not returning frames")
        h, w = frame.shape[:2]
        fps_nominal = cap.get(cv2.CAP_PROP_FPS)
        if not fps_nominal or fps_nominal <= 0 or fps_nominal > 120:
            fps_nominal = 30.0                            # camera doesn't report fps (sometimes -1) — otherwise VideoWriter won't write
        print(f"camera: {w}x{h} @ {fps_nominal:.0f} fps (nominal)")
    det = load_detector_optional() if (a.detect and cap is not None) else None

    if serial is None:
        raise SystemExit("pip install pyserial")
    cli_ser = None                                    # held open until the end of main (see send_config)
    try:
        cli_ser = send_config() if SEND_CFG else serial.Serial(CLI_PORT, 115200, timeout=1)
    except Exception as e:
        print(f"⚠️ config not sent ({e}) — fine if the radar is already streaming")

    t_start = time.perf_counter()
    clock = lambda: time.perf_counter() - t_start
    stop = threading.Event()
    stats = {"radar_bytes": 0, "radar_frames": 0, "ego_lines": 0}
    log = JsonlLog(os.path.join(OUT_DIR, "run.jsonl.gz"))
    log.write({"kind": "meta", "t": 0.0, "tag": tag, "wall_t0": time.time(), "started": datetime.now().isoformat(timespec="seconds"),
               "cfg_file": CFG_FILE, "cli_port": CLI_PORT, "data_port": DATA_PORT,
               "ego_port": EGO_SERIAL_PORT, "ego_baud": EGO_BAUD, "camera": None if cap is None else [w, h],
               "radar_to_camera_cm": RADAR_TO_CAMERA_CM,
               "calib": {k: _cfg.get(k) for k in CALIB_KEYS}})    # what the estimator was configured with at record time
    threads = [threading.Thread(target=radar_writer, args=(stop, clock, OUT_DIR, stats, log), daemon=True)]
    if EGO_SERIAL_PORT:
        threads.append(threading.Thread(target=ego_writer, args=(stop, clock, stats, log), daemon=True))
    else:
        print("ego: EGO_SERIAL_PORT is empty — no $EGOVEL records (set it in configs.json for the IMU drills)")
    for th in threads:
        th.start()

    ft = wt = None
    if cap is not None:
        writer = cv2.VideoWriter(os.path.join(OUT_DIR, "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_nominal, (w, h))
        if not writer.isOpened():
            raise SystemExit("VideoWriter did not open — check the mp4v codec / folder permissions")
        ft = open(os.path.join(OUT_DIR, "camera_times.csv"), "w", newline="")
        wt = csv.writer(ft); wt.writerow(["frame", "t_s"])
    n = 0
    print(f"=== recording to {OUT_DIR}/ · q in the window, a STOP file, or Ctrl+C — stop ===")
    try:
        while clock() < MAX_SECONDS:
            if cap is None:                               # radar + IMU only: nothing to grab, just report
                time.sleep(0.5)
                t = clock()
                if int(t * 2) % 10 == 0:
                    print(f"REC {t:6.1f}s  radar frames {stats['radar_frames']}  ego lines {stats['ego_lines']}", flush=True)
                if os.path.exists("STOP"):
                    break
                continue
            ok, frame = cap.read()
            t = clock()                                   # time right after getting the frame
            if not ok:
                continue
            writer.write(frame)
            wt.writerow([n, f"{t:.4f}"])
            rec = {"kind": "cam", "t": round(t, 4), "frame": n}
            if det is not None:
                F, model, keep = det
                rec["dets"] = [{"cls": d.cls, "conf": round(float(d.conf), 3), "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)]}
                               for d in F.yolo_detections(model, frame, t, keep)]
            log.write(rec)
            n += 1
            view = frame.copy()
            cv2.putText(view, f"REC {t:6.1f}s  cam {n}  radar frames {stats['radar_frames']}  ego {stats['ego_lines']}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if stats["radar_frames"] == 0 and t > 3:
                cv2.putText(view, "RADAR: NO FRAMES (cfg? port?)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if EGO_SERIAL_PORT and stats["ego_lines"] == 0 and t > 3:
                cv2.putText(view, "IMU: NO $EGOVEL LINES (port? wiring?)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if SHOW_PREVIEW:
                try:
                    cv2.imshow("record_sync (q = stop)", view)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                except cv2.error:
                    SHOW_PREVIEW = False              # OpenCV without GUI (headless) — record without a window
            elif n % 30 == 0:
                print(f"REC {t:6.1f}s  cam {n}  radar frames {stats['radar_frames']}  ego lines {stats['ego_lines']}", flush=True)
            if os.path.exists("STOP"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for th in threads:
            th.join(timeout=2)
        log.close()
        if writer is not None:
            writer.release()
        if ft is not None:
            ft.close()
        if cap is not None:
            cap.release()
        if cli_ser is not None:
            cli_ser.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        dur = clock()
        json.dump({"camera_index": CAMERA_INDEX if cap is not None else None, "width": w, "height": h, "fps_nominal": fps_nominal,
                   "camera_frames": n, "fps_actual": round(n / max(dur, 1e-3), 2),
                   "radar_frames": stats["radar_frames"], "radar_bytes": stats["radar_bytes"],
                   "ego_lines": stats["ego_lines"], "tag": tag, "log": "run.jsonl.gz",
                   "duration_s": round(dur, 2), "cli_port": CLI_PORT, "data_port": DATA_PORT, "ego_port": EGO_SERIAL_PORT,
                   "cfg_file": CFG_FILE, "radar_to_camera_cm": RADAR_TO_CAMERA_CM},
                  open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\ndone: {dur:.1f} s · camera frames {n} ({n / max(dur, 1e-3):.1f} fps) · "
              f"radar frames {stats['radar_frames']} ({stats['radar_frames'] / max(dur, 1e-3):.1f} fps) · "
              f"ego lines {stats['ego_lines']} ({stats['ego_lines'] / max(dur, 1e-3):.0f} Hz) · log records {log.n}")
        print(f"folder {OUT_DIR}/ — zip it up and send the whole thing")


if __name__ == "__main__":
    main()
