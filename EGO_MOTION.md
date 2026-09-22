# Ego-motion for the tractor radar — design contract (branch `ego-motion`)

Decision (2026-09-22): option **E + B** from the research report — reprofile the IWR1642 so the
tractor's own speed never wraps, take the carrier speed from the radar's stationary targets, take the
yaw rate from the BNO085 gyro, and keep the three physics corrections as **calibration constants**, not
Kalman states. MVP first: everything below is testable at home on recordings; the tractor gets one
short drill session.

## 1. Sensors and data path

- Radar: IWR1642BOOST, SDK 3.6 OOB demo point cloud over UART (x, y, z, doppler m/s, snr), ~15 fps.
  Profiles: `radar_configs/hangar_v9.cfg` (v_max 8.7 m/s, bin 0.27 m/s, ~18 m) indoors / hangar,
  `radar_configs/field_v15.cfg` (v_max 14.8 m/s, bin 0.23 m/s, ~40 m) in the field.
  `clutterRemoval 0`, `extendedMaxVelocity 0` (only after TI phase calibration with a corner reflector).
- IMU: BNO08x on the ESP32 over **I2C** (Adafruit_BNO08x library, address 0x4B, INT 4 / RESET 15 — the
  wiring and firmware the team already has in `firmware/ego_velocity/ego_velocity.ino`, see
  `EGO_VELOCITY.md`). The ESP32 streams `$EGOVEL` sentences out **Serial2 (GPIO17 TX2 → Pi GPIO15/pin 10 RXD,
  GPIO16 RX2 ← Pi GPIO14/pin 8 TXD, common GND)** at 115200; its USB `Serial` stays free for flashing and
  for the `$RDALT` alert node (the same board can run both: alerts in over USB, IMU out over Serial2).
  On the Pi the port is `/dev/ttyAMA0` or `/dev/ttyS0` (`EGO_SERIAL_PORT`).
- Pi 4 (Python): `iwr1642_live.py` (radar pipeline), `ego_velocity.py` (IMU link — extended here, replaces
  the interim `EgoLink` in fusion), `fusion-python-3.10.py` (camera fusion + alerts), `record_sync.py`
  (logging), `fusion_offline.py` (replay).
- Speed policy: the **radar Doppler fit is the primary speed**; the ESP32's integrated velocity (`vy_mps`,
  ZUPT-bounded dead reckoning) is only a **bridge** while the radar state is UNKNOWN (≤ 2 s) and an input
  to the STANDING decision (its ZUPT flag) — never the operational speed on its own.

## 2. `$EGOVEL` sentence (ESP32 → Pi, 50 Hz) — v2, backward compatible

```
$EGOVEL,<vx_mps>,<vy_mps>,<yaw_rate_dps>,<seq>,<flags>,<esp_ms>,<pitch_deg>,<roll_deg>,<acc_fwd_mps2>,<gyro_cal>*<XOR>
```
Fields 1–5 are exactly the existing v1 sentence (`EGO_VELOCITY.md`): `vx` lateral **right +**, `vy` forward,
`yaw_rate_dps` **+ = CCW from above = turning left** (same sign `Tracker.step(yaw_rate=...)` expects, in rad/s
inside Python), `seq` 0–255, `flags` bit 0 = ZUPT active. v2 appends:
- `esp_ms` — ESP32 `millis()` at the IMU sample; the Pi maps it to its clock with a running offset
  `median(t_rx_pi − esp_ms/1000)` over the last 2 s minus a fixed 5 ms link latency guess.
- `pitch_deg`, `roll_deg` — from the Game Rotation Vector (no magnetometer: steel chassis); nose-up +.
- `acc_fwd_mps2` — bias-corrected forward linear acceleration (already computed for the ZUPT).
- `gyro_cal` — BNO08x gyroscope accuracy status 0..3.
Empty fields mean "unknown". The reader accepts both the 5-field v1 and the 10-field v2 sentence
(v2 fields default to unknown/0 for v1) — a v1 firmware keeps working, a mismatched field count is still
rejected loudly. Checksum: XOR of the characters between `$` and `*`, two uppercase hex digits.

## 3. Calibration constants (`configs.json`)

| key | meaning | how measured |
|---|---|---|
| `RADAR_CONFIG` | profile file | — |
| `LEVER_ARM_X_M` | radar phase centre forward of the rear-axle centre (yaw rotation point), m | tape measure |
| `LEVER_ARM_Y_M` | radar left (+) of the vehicle centre line, m | tape measure (0 if on the centre line) |
| `MOUNT_YAW_DEG` | radar boresight left (+) of the vehicle forward axis | straight drive fwd/back: the yaw angle that zeroes the lateral component of the fit (`calib_mount.py`) |
| `EGO_TIME_OFFSET_S` | **IMU time − radar frame timestamp** for the same physical instant — exactly the quantity `sample_at` adds to the frame time (negative: the radar's timestamp is the later one, i.e. the radar lags) | `calib_time_offset.py` on a step-turn recording (replay grid search primary, cross-correlation check); re-measure after a profile change |
| `EGO_GYRO_BIAS_DPS` | last standstill gyro-bias estimate (auto-refreshed at every STANDING episode ≥ 3 s, persisted) | automatic |

## 4. Estimator (`iwr1642_live.EgoEstimator`)

Inputs per frame: raw detections (range, azimuth, doppler, snr), `dt`, profile numbers from `parse_cfg`
(use λ at the **start frequency** — the firmware's convention — for `doppler_res_mps` and the period),
and the IMU sample interpolated to the frame time (`t_frame + EGO_TIME_OFFSET_S`): `yaw_rate`
(bias-removed, rad/s), `pitch`, `roll`, `gyro_ok`.

Model for a stationary point i at azimuth θ_i. **One frame throughout: sensor frame x = right, y = forward =
boresight, θ = atan2(x, y) so θ is RIGHT-positive, ω is + for a left turn (CCW from above).**
`doppler_i = -(v_fwd·cos θ_i + v_lat·sin θ_i)` (TI convention: + = receding) where the radar's own velocity is
`v_fwd = v − ω·LEVER_ARM_Y_M` (forward) and `v_lat = −ω·LEVER_ARM_X_M` (lateral, **RIGHT +**, matching θ).
Expanded, a stationary point's Doppler is `−v·cos θ + ω·LEVER_ARM_X_M·sin θ + ω·LEVER_ARM_Y_M·cos θ` — the
lever-arm term `EgoEstimator._fit` subtracts. Rotate azimuths by `MOUNT_YAW_DEG` first.

1. **Gate points**: range ≥ `EGO_MIN_RANGE_M`, finite snr; on slopes (|pitch| or |roll| > 8°) also drop
   points with |z| implausible — keep it simple: log, do not reject.
2. **Bin-scaled constants**: `bin = doppler_res_mps`; inlier tolerance = 0.75·bin + 0.06 m/s.
3. **State machine** (hysteresis 2 frames):
   - `STANDING`: median|doppler| < 1 bin AND < 25 % of points above 1 bin AND |ω_gyro| < 1.5 °/s
     (when gyro ok) → v := 0 exactly; bias averaging window opens.
   - `CREEPING`: |v| in 1..5 bins → accept band widened ×2, `ego_info["ttc_bound"] = True`.
   - `MOVING`: |v| > 5 bins → normal fit.
   - `UNKNOWN`: no valid fit for > `EGO_HOLD_S` → v := last valid for `EGO_HOLD_S`, then 0 with
     `valid=False`.
4. **Fit**: when gyro ok, subtract the lever-arm term (`ω·LEVER_ARM_X_M·sin θ_i` and `ω·LEVER_ARM_Y_M·cos θ_i`)
   and fit the single unknown `v` (grid search on inlier count over the period-wrapped residuals as now,
   then IRLS with a **Cauchy** weight `w = 1/(1+(r/c)²)`, c = tolerance). Without gyro, keep the current
   2-parameter (v, v_lat) fit.
5. **Plausibility gate**: accept only if `|v − v_prev| ≤ a_max·dt + 0.5·bin` with `a_max = 2.5 m/s²`
   (gate width shrinks to `a_max·dt` when inlier fraction < 0.5); inliers ≥ `EGO_MIN_POINTS` (6) and
   inlier fraction ≥ 0.4; **azimuth spread** of inliers (max−min of azimuth) ≥ 25° — below that the fit is
   `ambiguous`, hold last value.
   The pitch/roll of the IMU sample are carried into `info()` with a `slope` flag (|pitch| or |roll| > 8°)
   so a replay can tell a slope-induced bias from a calibration error — logged, never a rejection.
6. **Radar yaw cross-check** (only when gyro ok, state MOVING, inliers ≥ 40 and spread ≥ 60°): run the
   2-parameter fit too; `yaw_rate_radar = v_lat / LEVER_ARM_X_M`; if `|yaw_rate_radar − ω_gyro| > 3 °/s`
   for 15 consecutive frames → `gyro_ok = False` flag in `ego_info` (alert code 130, see §7); never use
   radar yaw as the operational yaw. **After a 130 the operational yaw is 0**: `info()["yaw_rate"]` reports the
   yaw the fit actually used, so the tracker's de-rotation and the corridor curvature (§5) both fall back to
   "straight" rather than following a gyro the system has just declared untrustworthy. The same holds for a
   stale/uncalibrated sample — a yaw rate is used only while `gyro_ok` is true and the sample is fresh.
7. `info()` adds: `state`, `bin`, `yaw_rate` (used, rad/s), `yaw_rate_radar`, `gyro_ok`, `gyro_bias_dps`,
   `time_offset_s`, `n_inliers`, `n_points`, `az_spread_deg`, `ambiguous`.

Tracker / background: `Tracker.step(..., yaw_rate=ω_gyro)` unchanged; `BackgroundMap` becomes a
**short-memory ego buffer**: cells decay with `BACKGROUND_FORGET_S = 5 s` while moving, are translated by
`v·dt` and rotated by `ω·dt` each frame (existing `rotate` path), and are used as *persistence evidence*
(a static return seen ≥ N frames in the same cell raises confidence) — never as a suppression mask.

## 5. Fusion layer (`fusion-python-3.10.py`)

- `ego_velocity.EgoVelocityReader` (extended; the interim `EgoLink` class in fusion is removed): parse the §2
  sentence (v1 and v2); keep a 2-s ring buffer of samples with Pi timestamps; `sample_at(t)` returns the
  linearly interpolated `{yaw_rate (rad/s, + left, bias-removed), pitch, roll, acc_fwd, vy_imu, zupt,
  gyro_ok, age_s}` for `t + EGO_TIME_OFFSET_S`; bias removal with `EGO_GYRO_BIAS_DPS`;
  `gyro_ok = gyro_cal ≥ 2 (or unknown on v1) and age ≤ 0.2 s`; the existing `vx_mps/vy_mps/yaw_rate_dps`
  properties stay for backward compatibility. Add `feed_line(line, t_rx)` for tests.
- Standstill bias refresh: when the radar reports `STANDING` for ≥ 3 s (the IMU's ZUPT flag is a supporting
  input), average the raw gyro Z and store it (`configs.json` write-back on clean exit, that key only).
- `Track.ground_velocity(ego_vx, ego_vy)` (upstream) is fed with the radar-fit speed (`ego_info["v"]`) and
  the fit's lateral component, not the IMU's integrated velocity.
- Corridor: half-width `CORRIDOR_HALF_WIDTH_M` (track + implement/2 + 0.3 m margin); look-ahead
  `max(6 m, v·TTC_WARN_S·1.3)`; when `v ≥ 0.7 m/s` bend the corridor with curvature `κ = ω/v`
  (lateral offset ≈ κ·y²/2), else a straight box.
- Static/moving label after compensation: keep `STATIC_DOPPLER_MPS` as the per-point threshold but derive
  it from the profile: `max(0.2 m/s, 1.2·bin)` per point (≈ 3σ of the ±½-bin floor), and a cluster-level
  label at `0.8·bin` (median of ≥ 3 points, `cluster_objects()[i]["is_static"]`, carried onto the track);
  a tangential mover (person crossing) has zero Doppler and is
  caught only by the tracker's position-derived speed. Compensation uses the **full** sensor-frame ego
  velocity — forward `v` and the fit's lateral `vx = −ω·LEVER_ARM_X_M`, both rotated back by
  `MOUNT_YAW_DEG` — not the forward scalar alone, or every turn labels the outer-azimuth clutter as moving.
  (`points_to_detections`' forward-only `ego_speed_mps` feature for the LightGBM model stays as trained.)
- Alerts (`alerts.py`): TTC tiers `TTC_WARN_S = 3.5`, `TTC_CRITICAL_S = 1.8` computed with the
  ego-compensated closing speed of the track; when `state == STANDING` **and the unclamped fit `raw` is
  below `EGO_STATIONARY_MPS` = 0.15 m/s** alerts are zone-occupancy only — STANDING is a one-bin deadband
  (0.27 m/s on hangar_v9), so the state alone must not silence the collision layer on a creeping tractor.
  While `ttc_bound` (CREEPING, §4.3) the TTC tiers need a closing speed above `2·bin` and 122 TTC_CRITICAL
  is held back to 121 — at 1..5 bins the track's radial speed is mostly quantisation noise.
  A static object inside the corridor is an obstacle by default (warn tier once confirmed by persistence
  ≥ 3 frames or by the camera), never a "background" drop.

## 6. Logging, replay, calibration tools

- `record_sync.py`: logs radar frames (existing), `$EGOVEL` lines with Pi receive time, camera detections;
  one `.jsonl` per run, gzip; `--tag` names the drill.
- `fusion_offline.py --log <file>`: replays with the same code path (`Pipeline.process` + `EgoVelocityReader.sample_at`
  fed from the log), prints the per-frame `ego_info` and a summary (mean/median v per segment, state
  histogram, false-moving rate on points the user marks static via `--static-mask`).
- `tools/calib_time_offset.py <log>`: (a) replay grid search over `EGO_TIME_OFFSET_S ∈ [−0.4, +0.1] s`
  step 10 ms minimising the residual of the lever-arm-compensated fit on turning frames, (b)
  cross-correlation of gyro yaw rate vs radar yaw rate (2-parameter fit) as a check; prints both and the
  frame period; pass criterion: |a − b| < 1 frame.
- `tools/calib_mount.py <log_fwd> <log_back>`: mounting yaw from straight drives.
- `tools/drills.md`: the checklist for the tractor session with pass/fail numbers (from the report).

## 6b. Firmware (`firmware/ego_velocity/ego_velocity.ino`)

Extend the existing sketch, do not rewrite it: enable `SH2_GAME_ROTATION_VECTOR` (50 Hz) next to the
linear-acceleration and calibrated-gyro reports; derive pitch/roll from the quaternion; emit the v2 sentence
with `millis()` and the gyro accuracy status; keep the ZUPT and startup bias logic; add the `$RDALT`
alert-node handling (LEDs/buzzer, from `esp32/alert_node/alert_node.ino`) on the USB `Serial` behind a
`#define ALERT_NODE 1` so one ESP32 can do both jobs. Keep `FORWARD_SIGN`/`RIGHT_SIGN`.

## 7. Alert codes added (see ALERTS.md)

- 130 — IMU degraded (stale, uncalibrated, or radar/gyro yaw disagreement) — severity W.
- 131 — ego estimate lost (UNKNOWN > 2 s while not STANDING) — severity W.
- 141 — obstacle in corridor (now issued for static objects too). NB: §7 originally said "140 stays", but in
  this repository 140 was already `RADAR_SENSOR_STALE`, so the corridor obstacle is **141**
  (`AlertCode.OBSTACLE_IN_CORRIDOR`); ALERTS.md and `tools/drills.md` T5 use 141.

## 8. Tests (`test_ego.py`, `tests/`)

Synthetic: (A) standing scene — state STANDING, v = 0; (B) straight drive at 0.3 / 1 / 4.5 m/s with
wrapped Doppler on the old profile vs unwrapped on hangar_v9 — |v − v_true| < 0.5·bin; (C) turn at
ω = 0.3 rad/s with lever arm 2 m — with gyro, |v − v_true| < 0.5·bin (fix the sign bug in the old test C:
`th = w*dt*k`); (D) one large mover covering 60 % of points — gate holds the previous value; (E) narrow
azimuth spread — `ambiguous`, hold; (F) `$EGOVEL` v1/v2 parsing, checksum, seq drops, interpolation, time offset and bias removal;
(G) `calib_time_offset` recovers an injected offset of −120 ms within 20 ms.
Recorded: `test_stationary_stand.bin` → STANDING ≥ 95 % of frames; `test_stationary_walk.bin` → the
walker is `moving`, the background static.
