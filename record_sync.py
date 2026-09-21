"""record_sync.py — simultaneous recording of the laptop camera and the IWR1642 radar on a shared clock.

    python record_sync.py

Exit — the q key in the camera window, or Ctrl+C. Output: folder rec_<time>/ :
    camera.mp4          video from the camera
    camera_times.csv    frame number -> recording time (s from start)
    radar.bin           raw radar stream (same as dump_radar.py)
    radar_times.csv     byte offset -> time (s from start) — used to align radar frames with the video
    meta.json           parameters: ports, resolution, FPS, radar-camera distance

Does no analysis — only records. Processing happens later: fusion.py --video ... --dump ...
"""
import csv
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
SEND_CFG = True             # False if the radar is already streaming
RADAR_TO_CAMERA_CM = {"right": 0.0, "up": 0.0, "forward": 0.0}   # where the radar is relative to the camera lens
MAX_SECONDS = 600           # safety net: stop after 10 minutes
SHOW_PREVIEW = True         # False — no window (Jupyter / headless); stop via a STOP file in the launch folder or MAX_SECONDS

OUT_DIR = f"rec_{datetime.now():%Y%m%d_%H%M%S}"
MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"


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


def radar_writer(stop, clock, out_dir, stats):
    """Writes bytes from the data port to radar.bin and the time of each chunk to radar_times.csv."""
    with serial.Serial(DATA_PORT, 921600, timeout=0.01) as ser, \
         open(os.path.join(out_dir, "radar.bin"), "wb") as fb, \
         open(os.path.join(out_dir, "radar_times.csv"), "w", newline="") as ft:
        w = csv.writer(ft); w.writerow(["byte_offset", "t_s"])
        offset = 0
        while not stop.is_set():
            chunk = ser.read(max(1, ser.in_waiting))     # short reads: timestamp ±10 ms, not ±100
            if chunk:
                t = clock()
                fb.write(chunk)
                w.writerow([offset, f"{t:.4f}"])
                offset += len(chunk)
                stats["radar_bytes"] = offset
                stats["radar_frames"] += chunk.count(MAGIC)
    print("radar: stopped")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"camera {CAMERA_INDEX} did not open")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera is not returning frames")
    h, w = frame.shape[:2]
    fps_nominal = cap.get(cv2.CAP_PROP_FPS)
    if not fps_nominal or fps_nominal <= 0 or fps_nominal > 120:
        fps_nominal = 30.0                            # camera doesn't report fps (sometimes -1) — otherwise VideoWriter won't write
    print(f"camera: {w}x{h} @ {fps_nominal:.0f} fps (nominal)")

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
    stats = {"radar_bytes": 0, "radar_frames": 0}
    th = threading.Thread(target=radar_writer, args=(stop, clock, OUT_DIR, stats), daemon=True)
    th.start()

    writer = cv2.VideoWriter(os.path.join(OUT_DIR, "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_nominal, (w, h))
    if not writer.isOpened():
        raise SystemExit("VideoWriter did not open — check the mp4v codec / folder permissions")
    ft = open(os.path.join(OUT_DIR, "camera_times.csv"), "w", newline="")
    wt = csv.writer(ft); wt.writerow(["frame", "t_s"])
    n = 0
    global SHOW_PREVIEW
    print(f"=== recording to {OUT_DIR}/ · q in the window, a STOP file, or Ctrl+C — stop ===")
    try:
        while clock() < MAX_SECONDS:
            ok, frame = cap.read()
            t = clock()                                   # time right after getting the frame
            if not ok:
                continue
            writer.write(frame)
            wt.writerow([n, f"{t:.4f}"])
            n += 1
            view = frame.copy()
            cv2.putText(view, f"REC {t:6.1f}s  cam {n}  radar frames {stats['radar_frames']}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if stats["radar_frames"] == 0 and t > 3:
                cv2.putText(view, "RADAR: NO FRAMES (cfg? port?)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if SHOW_PREVIEW:
                try:
                    cv2.imshow("record_sync (q = stop)", view)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                except cv2.error:
                    SHOW_PREVIEW = False              # OpenCV without GUI (headless) — record without a window
            elif n % 30 == 0:
                print(f"REC {t:6.1f}s  cam {n}  radar frames {stats['radar_frames']}", flush=True)
            if os.path.exists("STOP"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); th.join(timeout=2)
        writer.release(); ft.close(); cap.release()
        if cli_ser is not None:
            cli_ser.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        dur = clock()
        json.dump({"camera_index": CAMERA_INDEX, "width": w, "height": h, "fps_nominal": fps_nominal,
                   "camera_frames": n, "fps_actual": round(n / max(dur, 1e-3), 2),
                   "radar_frames": stats["radar_frames"], "radar_bytes": stats["radar_bytes"],
                   "duration_s": round(dur, 2), "cli_port": CLI_PORT, "data_port": DATA_PORT,
                   "cfg_file": CFG_FILE, "radar_to_camera_cm": RADAR_TO_CAMERA_CM},
                  open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\ndone: {dur:.1f} s · camera frames {n} ({n / max(dur, 1e-3):.1f} fps) · "
              f"radar frames {stats['radar_frames']} ({stats['radar_frames'] / max(dur, 1e-3):.1f} fps)")
        print(f"folder {OUT_DIR}/ — zip it up and send the whole thing")


if __name__ == "__main__":
    main()
