IWR1642 radar + camera: recording and fusion
=========================================
All files go in one folder. Ports COM5 (commands) / COM6 (data), config profile_sdk3.cfg — already set up.

Install:  pip install pyserial opencv-python numpy pandas scikit-learn ultralytics

1) Record (camera + radar, shared clock):        python record_sync.py
   -> folder rec_<time>/  (camera.mp4, radar.bin, timestamps, meta.json)
   Stop: q in the window, or a STOP file in the launch folder, or MAX_SECONDS.
   From Jupyter: SHOW_PREVIEW = False.

2) Fuse a recording (video + radar -> fused.mp4, fusion.csv):
                                                python fusion_offline.py rec_<time> [--meters 8]
   Simulate camera loss to test hold-tracking: --blind 14,26 (haze) | --miss 14,26 (miss) | --occlude 14,26 (sheet)
   Open-vocabulary detector (can, tractor): --detector yolo-world  (first run downloads ~340 MB)

3) Live mode (camera + radar in real time, two windows):
                                                python fusion.py
   Self-test without a camera:                  python fusion.py --selftest --dump <dump.bin>

4) Radar only, live window / from a dump:       python iwr1642_live.py   (DUMP_FILE at the top)
5) Raw radar dump:                              python dump_radar.py --sec 10

Files:
  record_sync.py    recording
  fusion_offline.py fusion over a recorded folder
  fusion.py         live fusion + self-test; matching and calibration logic
  iwr1642_live.py   radar: TLV parsing, points, background map, tracks (EKF), rendering, /radar/tracks contract
  dump_radar.py     records the raw radar stream to .bin
  profile_sdk3.cfg  radar config for SDK 3.x firmware (9 m, ±1 m/s — indoor; a different one is needed for the field)

Object states on the video: green box — camera+radar; cyan — camera lost it, radar is holding it
(label: reason — occluded / haze / detector miss); dashed — radar also has no measurement; red circle —
moving radar with no pair. The number on the box is the range to the nearest point of the object (radar).
RADAR_TO_CAMERA_CM in record_sync.py (where the board sits relative to the lens) — used during fusion (parallax).
