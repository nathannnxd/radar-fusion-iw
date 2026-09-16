# IWR1642 Radar + Camera: Recording and Fusion

All files are in one folder. Ports: **COM5 (commands) / COM6 (data)**, and the `profile_sdk3.cfg` configuration is already set up.

### 1) Recording (camera + radar, synchronized clocks)

```bash
python record_sync.py
```

→ Creates a `rec_<time>/` folder containing:

* `camera.mp4`
* `radar.bin`
* timestamps
* `meta.json`

**Stop:** press `q` in the window, or create a `STOP` file in the launch folder, or use `MAX_SECONDS`.

From Jupyter: `SHOW_PREVIEW = False`.

### 2) Offline fusion (video + radar → fused video + CSV)

```bash
python fusion_offline.py rec_<time> [--meters 8]
```

Simulate camera loss to test object retention:

* `--blind 14,26` — haze/smoke
* `--miss 14,26` — detector miss
* `--occlude 14,26` — occlusion by a leaf

Open-vocabulary detector (e.g. **can, tractor**):

```bash
--detector yolo-world
```

(The first run downloads approximately 340 MB.)

### 3) Live mode (camera + radar in real time, two windows)

```bash
python fusion.py
```

Self-test without a camera:

```bash
python fusion.py --selftest --dump <dump.bin>
```

### 4) Radar only, live window / from a dump

```bash
python iwr1642_live.py
```

(`DUMP_FILE` is specified at the top of the file.)

### 5) Raw radar dump

```bash
python dump_radar.py --sec 10
```

## Files

* `record_sync.py` — recording
* `fusion_offline.py` — fusion using a recorded folder
* `fusion.py` — live fusion + self-test; matching and calibration logic
* `iwr1642_live.py` — radar: TLV parsing, point cloud, background map, tracks (EKF), visualization, `/radar/tracks` contract
* `dump_radar.py` — records the raw radar stream to a `.bin` file
* `profile_sdk3.cfg` — radar configuration for SDK 3.x firmware (**9 m, ±1 m/s — indoor use; a different configuration is needed for field/outdoor use**)

## Object states in the video

* **Green bounding box** — detected by both camera and radar
* **Cyan bounding box** — camera lost the object; radar continues tracking it
  Label: reason — `occluded` / `haze` / `detector miss`
* **Dashed outline** — radar also has no measurement
* **Red circle** — moving radar target without a camera match

The number displayed on the bounding box is the **distance to the nearest point of the object**, as measured by the radar.

`RADAR_TO_CAMERA_CM` in `record_sync.py` (the position of the radar board relative to the camera lens) is used during fusion to account for **parallax**.
