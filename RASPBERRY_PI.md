# Running this project on a Raspberry Pi 4 or 5

This branch (`claude/raspberry-pi-4-5`) is the Pi configuration of the project: the same radar +
camera fusion pipeline, with `configs.json` and the YOLO loading path (`fusion-python-3.10.py`) set
up for what's actually achievable on a Pi's CPU — no CUDA on either board, and YOLO run through the
NCNN backend at a reduced input resolution instead of stock PyTorch at 640×640. Everything here
applies to both boards identically — same aarch64 architecture, same OS, same lack of a CUDA GPU —
except where a section below calls out a Pi 4 vs. Pi 5 difference explicitly (mainly power draw and
expected FPS).

**Not yet validated on real hardware.** Everything here was checked by: compiling every changed file,
running targeted unit tests against stubbed `ultralytics`/`torch` modules to verify the control flow
(export-once/cache/reload, kwargs passed to `model.track()`), confirming real aarch64 wheel
availability for every dependency against the live PyPI index, and reasoning from the FPS analysis
done earlier in this project. It has **not** been run on an actual Raspberry Pi 4 or 5 with the real
radar and camera attached — treat the performance numbers below as estimates, and expect to spend
time on serial port names, camera index, and USB power in particular (see below).

## 1. OS and Python version

Use **64-bit Raspberry Pi OS (Bookworm) or newer**, with its default Python (3.11).

Why this matters: `torch`'s official PyPI wheels for Linux/aarch64 only go back to Python 3.10
(`cp310`) for current releases — there is no `cp39`-aarch64 wheel for recent `torch` versions, so
Python 3.9 (Raspberry Pi OS **Bullseye**'s default) will try to build PyTorch from source, which is
slow and failure-prone on a Pi. Every other dependency in `requirements-rpi.txt` (numpy, pandas,
scikit-learn, opencv-python, lightgbm, ncnn, ultralytics) has aarch64 wheels going back further.

- **On Bookworm (Python 3.11, recommended):** everything in `requirements-rpi.txt` installs from
  prebuilt wheels, no pinning needed.
- **Stuck on Bullseye (Python 3.9):** pin `torch<=2.2.2` and `torchvision<=0.17.2` (the last releases
  with a `cp39`-aarch64 wheel) — `pip install torch==2.2.2 torchvision==0.17.2` — before installing the
  rest. `ultralytics` only requires `torch>=1.8.0`, so this is compatible. **This only applies to a
  Pi 4** — the Pi 5 doesn't support Bullseye at all, so on a Pi 5 you're on Bookworm/Python 3.11 by
  default and this doesn't come up.

Confirm you're on 64-bit before installing anything:
```
uname -m        # must print aarch64, not armv7l
```

## 2. System packages

```
sudo apt update
sudo apt install -y python3-pip python3-venv libatlas-base-dev
# only if you'll use the live on-screen windows (SHOW_WINDOW = True / cv2.imshow):
sudo apt install -y libgtk-3-0 libqt5gui5
```
If you're running headless (no monitor — e.g. mounted on a robot/vehicle), set `SHOW_WINDOW = False`
in `fusion-python-3.10.py` / `iwr1642_live.py` and you can skip the GTK/Qt packages and use
`opencv-python-headless` instead of `opencv-python` in `requirements-rpi.txt`.

## 3. Python dependencies

```
python3 -m venv ~/radar-fusion-venv
source ~/radar-fusion-venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rpi.txt
```
This can take a while the first time (torch alone is a large download) — that's expected, not a hang.

## 4. Find your actual serial ports and camera index

`configs.json` on this branch ships with placeholder values — **you must verify these against your own
Pi**, they will not be right by default:

```json
"CLI_PORT": "/dev/ttyACM0",
"DATA_PORT": "/dev/ttyACM1",
"CAMERA_INDEX": 0,
```

- **Radar (IWR1642 via its XDS110 debug probe):** plug it in, then run:
  ```
  ls /dev/serial/by-id/          # stable names — prefer these over ttyACM0/1 if you have multiple USB-serial devices
  dmesg | grep -i tty            # shows which ttyACM/ttyUSB node was just assigned
  ```
  The XDS110 probe on the IWR1642 EVM typically enumerates as **two** `/dev/ttyACM*` devices (CDC-ACM,
  not FTDI/CP210x) — one for commands (`CLI_PORT`), one for data (`DATA_PORT`). Which one is which can
  swap depending on enumeration order; if the radar doesn't start streaming, try swapping the two in
  `configs.json`.
- **Serial permissions:** the pi user needs to be in the `dialout` group, or `pyserial` will fail with
  a permission error:
  ```
  sudo usermod -aG dialout $USER
  # log out and back in for this to take effect
  ```
- **Camera:**
  ```
  v4l2-ctl --list-devices        # or: ls /dev/video*
  ```
  Set `CAMERA_INDEX` to whichever `/dev/videoN` your USB camera landed on (usually `0` if it's the only
  camera; the Pi's own camera connector, if enabled, can also claim index 0 — check both).
- **A third serial port, if you're using the alert system (see §9):** `ALERT_SERIAL_PORT` in
  `configs.json` is a *separate* port from `CLI_PORT`/`DATA_PORT` above — it's the connection to
  whatever MCU is receiving alerts, not the radar. On a Pi with only one built-in USB-serial-capable
  header, this typically means a second USB-serial adapter; same `dialout` group requirement applies.

## 5. What changed on this branch vs. the main pipeline

- **`USE_NCNN`** (`configs.json`, default `1`) — runs YOLO through Ultralytics' NCNN backend instead of
  stock PyTorch. This is the one change that makes the CV part of the pipeline viable on a Pi's CPU;
  see the FPS estimate below. `USE_GPU` is left wired up but harmless — there's no CUDA GPU on either
  a Pi 4 or a Pi 5, so it always falls back to CPU regardless of that setting (default set to `0` here
  just to skip the pointless `torch.cuda.is_available()` warning at startup).
- **`YOLO_IMGSZ`** (`configs.json`, default `[320, 416]`) — the inference resolution passed to
  `model.track()`. Previously hardcoded to the library default (640×640) with no way to change it.
  Must be multiples of 32 in each dimension.
- **First run with `USE_NCNN=1` needs internet access once** — Ultralytics downloads a `pnnx` converter
  binary the first time you export a model to NCNN format. After that first export, the result is
  cached in a `yolov8n_ncnn_model/` folder next to the `.pt` weights and reused on every subsequent run
  with no network needed.
- **`DETECTOR`** defaulted to `"yolov8n"` instead of `"yolo-world"` — the open-vocabulary detector's
  CLIP text tower is too slow for a Pi even before considering NCNN (see the earlier cost analysis:
  ~0.25 s/frame on a *desktop* CPU). NCNN export of `yolo-world` is wired up in the code but **untested**
  — the open-vocab head may not export cleanly; `yolov8n` is the validated path here.
- **Fixed a pre-existing bug in `fusion_offline.py`**, unrelated to the Pi specifically: it had a
  hardcoded Windows path left over from development and imported a module (`fusion`) that doesn't exist
  under that name (the actual file is `fusion-python-3.10.py`, which isn't a valid Python module name).
  This meant `fusion_offline.py` couldn't run on *any* platform before this fix, not just the Pi.
- **`alerts.py`** — ported over from the main branch, unmodified (no Pi-specific concerns — it's pure
  `pyserial`). See §9.

## 6. Expected performance

**Pi 4:** carried over from the earlier cost analysis in this project (YOLOv8n + NCNN + 320×416,
"ignore all other optimizations"): an estimated **~10–18 FPS**, most likely around **12–15 FPS**, on
the camera/fusion side. This is extrapolated from published Ultralytics/community Pi 4 benchmarks and
FLOP-scaling with resolution — not a direct measurement, since no Pi 4 is available to benchmark
against in this environment.

**Pi 5:** its Cortex-A76 cores at 2.4 GHz are commonly benchmarked at roughly 2–3× the Pi 4's
Cortex-A72 throughput for CPU-bound compute. Scaling the Pi 4 estimate above by that factor gives a
rough **~25–40 FPS** ballpark. Treat this one with extra caution — it's a multiplier stacked on top of
an already-extrapolated number, not a measurement or even a direct benchmark citation. "Meaningfully
faster than the Pi 4, likely real-time-usable" is the safe takeaway; don't plan around the exact figure.

Actual results on either board will depend on your specific NCNN build, thermal throttling under
sustained load, and USB camera capture overhead. The radar-processing and sensor-fusion math (EKF,
DBSCAN, GNN association) are not a bottleneck at any resolution on either board — they were already
estimated at low-single-digit milliseconds per frame on much weaker hardware than a Pi 4's Cortex-A72,
let alone a Pi 5's Cortex-A76.

## 7. Power

Sustained YOLO inference on all cores, plus a USB camera and a USB-connected radar board, can draw
more than older/cheaper "phone charger" USB power supplies provide — on either board.

- **Pi 4:** use the official 5V/3A USB-C supply (or better).
- **Pi 5:** wants more — use the official 5V/5A USB-C PD supply. A supply that was adequate for a Pi 4
  can under-power a Pi 5 under sustained load.

If you see random USB disconnects (the radar or camera dropping out), check the power supply first —
that's not a code bug.

## 9. Alert system (MCU output)

This branch includes `alerts.py` — see `ALERTS.md` for the full wire format, alert code table, and
a generic C reference parser. Nothing about it is Pi-specific (it's pure `pyserial`, already a
dependency here), so it works identically to the desktop version; the only Pi-specific note is the
serial port one in §4 above. Leave `ALERT_SERIAL_PORT` empty in `configs.json` to test it
console-only before wiring up any actual MCU.

## 10. Still to do / verify on real hardware

Applies to whichever board you're actually deploying on (Pi 4 or Pi 5):

- [ ] Confirm actual `/dev/ttyACM*` assignment and swap order if the radar doesn't stream
- [ ] Confirm `CAMERA_INDEX`
- [ ] Time the actual NCNN export step (first run) and confirm internet access works for the `pnnx` download
- [ ] Measure real FPS and compare against the estimate for your board above (~12–15 FPS on a Pi 4, ~25–40 FPS on a Pi 5)
- [ ] Check CPU temperature / throttling under sustained load (`vcgencmd measure_temp`); consider a fan/heatsink if throttling
- [ ] If `yolo-world` is needed after all, test whether its NCNN export actually works before relying on it
