"""record_sync.py — одновременная запись камеры ноутбука и радара IWR1642 с общими часами.

    python record_sync.py

Выход — клавиша q в окне камеры или Ctrl+C. На выходе папка rec_<время>/ :
    camera.mp4          видео с камеры
    camera_times.csv    номер кадра → время записи (с от старта)
    radar.bin           сырой поток радара (как dump_radar.py)
    radar_times.csv     смещение в байтах → время (с от старта) — по нему кадры радара привязываются к видео
    meta.json           параметры: порты, разрешение, FPS, расстояние радар–камера

Ничего не анализирует — только пишет. Обработка потом: fusion.py --video ... --dump ...
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

# ---------------------------------------------------------------- НАСТРОЙКИ
CAMERA_INDEX = 0            # 0 — встроенная камера; 1 — внешняя USB
CAM_WIDTH, CAM_HEIGHT = 640, 480
CLI_PORT = "COM5"           # командный порт (User UART)
DATA_PORT = "COM6"          # порт данных (Auxiliary)
CFG_FILE = "profile_sdk3.cfg"
SEND_CFG = True             # False, если радар уже стримит
RADAR_TO_CAMERA_CM = {"right": 0.0, "up": 0.0, "forward": 0.0}   # где радар относительно объектива камеры
MAX_SECONDS = 600           # предохранитель: остановиться через 10 минут
SHOW_PREVIEW = True         # False — без окна (Jupyter / headless); остановка: файл STOP в папке запуска или MAX_SECONDS

OUT_DIR = f"rec_{datetime.now():%Y%m%d_%H%M%S}"
MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"


def send_config():
    with serial.Serial(CLI_PORT, 115200, timeout=1) as ser, open(CFG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("%"):
                ser.write((line + "\n").encode())
                time.sleep(0.05)
    print("✓ конфиг отправлен в", CLI_PORT)


def radar_writer(stop, clock, out_dir, stats):
    """Пишет байты из порта данных в radar.bin и время каждого куска в radar_times.csv."""
    with serial.Serial(DATA_PORT, 921600, timeout=0.01) as ser, \
         open(os.path.join(out_dir, "radar.bin"), "wb") as fb, \
         open(os.path.join(out_dir, "radar_times.csv"), "w", newline="") as ft:
        w = csv.writer(ft); w.writerow(["byte_offset", "t_s"])
        offset = 0
        while not stop.is_set():
            chunk = ser.read(max(1, ser.in_waiting))     # короткие чтения: метка времени ±10 мс, не ±100
            if chunk:
                t = clock()
                fb.write(chunk)
                w.writerow([offset, f"{t:.4f}"])
                offset += len(chunk)
                stats["radar_bytes"] = offset
                stats["radar_frames"] += chunk.count(MAGIC)
    print("радар: остановлен")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"камера {CAMERA_INDEX} не открылась")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("камера не отдаёт кадры")
    h, w = frame.shape[:2]
    fps_nominal = cap.get(cv2.CAP_PROP_FPS)
    if not fps_nominal or fps_nominal <= 0 or fps_nominal > 120:
        fps_nominal = 30.0                            # камера не сообщает fps (бывает -1) — иначе VideoWriter не пишет
    print(f"камера: {w}x{h} @ {fps_nominal:.0f} fps (номинал)")

    if serial is None:
        raise SystemExit("pip install pyserial")
    if SEND_CFG:
        try:
            send_config()
        except Exception as e:
            print(f"⚠️ конфиг не отправлен ({e}) — если радар уже стримит, нормально")

    t_start = time.perf_counter()
    clock = lambda: time.perf_counter() - t_start
    stop = threading.Event()
    stats = {"radar_bytes": 0, "radar_frames": 0}
    th = threading.Thread(target=radar_writer, args=(stop, clock, OUT_DIR, stats), daemon=True)
    th.start()

    writer = cv2.VideoWriter(os.path.join(OUT_DIR, "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_nominal, (w, h))
    if not writer.isOpened():
        raise SystemExit("VideoWriter не открылся — проверьте кодек mp4v / права на папку")
    ft = open(os.path.join(OUT_DIR, "camera_times.csv"), "w", newline="")
    wt = csv.writer(ft); wt.writerow(["frame", "t_s"])
    n = 0
    global SHOW_PREVIEW
    print(f"=== запись в {OUT_DIR}/ · q в окне, файл STOP или Ctrl+C — стоп ===")
    try:
        while clock() < MAX_SECONDS:
            ok, frame = cap.read()
            t = clock()                                   # время сразу после получения кадра
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
                    SHOW_PREVIEW = False              # OpenCV без GUI (headless) — пишем без окна
            elif n % 30 == 0:
                print(f"REC {t:6.1f}s  cam {n}  radar frames {stats['radar_frames']}", flush=True)
            if os.path.exists("STOP"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); th.join(timeout=2)
        writer.release(); ft.close(); cap.release()
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
        print(f"\nготово: {dur:.1f} с · кадров камеры {n} ({n / max(dur, 1e-3):.1f} fps) · "
              f"кадров радара {stats['radar_frames']} ({stats['radar_frames'] / max(dur, 1e-3):.1f} fps)")
        print(f"папка {OUT_DIR}/ — заархивировать и прислать целиком")


if __name__ == "__main__":
    main()
