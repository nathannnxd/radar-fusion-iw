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
- IMU: BNO085 on the **ESP32 over SPI** (the chip's I2C erratum; the esp32_BNO08x driver is SPI-only).
  ESP32 streams `$RDEGO` sentences on the same USB serial it receives `$RDALT` alerts on.
- Pi 4 (Python): `iwr1642_live.py` (radar pipeline), `fusion-python-3.10.py` (camera fusion + alerts),
  `record_sync.py` (logging), `fusion_offline.py` (replay).

## 2. `$RDEGO` sentence (ESP32 -> Pi, 50 Hz)

```
$RDEGO,<esp_ms>,<yaw_rate_dps>,<pitch_deg>,<roll_deg>,<acc_fwd_mps2>,<gyro_cal>,<v_mps>*<XOR checksum>\r\n
```
- `esp_ms` — ESP32 millis() at the IMU sample (monotonic; the Pi maps it to its own clock by a running
  offset = median(t_rx_pi - esp_ms/1000) over the last 2 s minus a fixed USB latency guess of 5 ms).
- `yaw_rate_dps` — BNO085 *Calibrated Gyroscope* Z, deg/s, **+ = turning left (CCW from above)** —
  the same sign as `Tracker.step(yaw_rate=...)` expects (rad/s inside Python).
- `pitch_deg`, `roll_deg` — from the Game Rotation Vector (no magnetometer: steel chassis); nose-up +.
- `acc_fwd_mps2` — linear acceleration along the tractor's forward axis (gravity removed by the chip).
- `gyro_cal` — BNO085 calibration status 0..3.
- `v_mps` — empty for now (reserved for a later GNSS/Hall speed); empty fields mean "unknown".
Checksum exactly as `$RDALT` (XOR of the characters between `$` and `*`, two hex digits).

## 3. Calibration constants (`configs.json`)

| key | meaning | how measured |
|---|---|---|
| `RADAR_CONFIG` | profile file | — |
| `LEVER_ARM_X_M` | radar phase centre forward of the rear-axle centre (yaw rotation point), m | tape measure |
| `LEVER_ARM_Y_M` | radar left (+) of the vehicle centre line, m | tape measure (0 if on the centre line) |
| `MOUNT_YAW_DEG` | radar boresight left (+) of the vehicle forward axis | straight drive fwd/back: the yaw angle that zeroes the lateral component of the fit (`calib_mount.py`) |
| `EGO_TIME_OFFSET_S` | radar frame timestamp − IMU time for the same physical instant (negative: radar lags) | `calib_time_offset.py` on a step-turn recording (replay grid search primary, cross-correlation check); re-measure after a profile change |
| `EGO_GYRO_BIAS_DPS` | last standstill gyro-bias estimate (auto-refreshed at every STANDING episode ≥ 3 s, persisted) | automatic |

## 4. Estimator (`iwr1642_live.EgoEstimator`)

Inputs per frame: raw detections (range, azimuth, doppler, snr), `dt`, profile numbers from `parse_cfg`
(use λ at the **start frequency** — the firmware's convention — for `doppler_res_mps` and the period),
and the IMU sample interpolated to the frame time (`t_frame + EGO_TIME_OFFSET_S`): `yaw_rate`
(bias-removed, rad/s), `pitch`, `roll`, `gyro_ok`.

Model for a stationary point i at azimuth θ_i (sensor frame, boresight = +y as in the code):
`doppler_i = -(v_sx·cos θ_i + v_sy·sin θ_i)` where the sensor-frame velocity is
`v_sx = v − ω·LEVER_ARM_Y_M` (forward), `v_sy = ω·LEVER_ARM_X_M` (lateral, left +). Rotate azimuths by
`MOUNT_YAW_DEG` first.

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
6. **Radar yaw cross-check** (only when gyro ok, state MOVING, inliers ≥ 40 and spread ≥ 60°): run the
   2-parameter fit too; `yaw_rate_radar = v_lat / LEVER_ARM_X_M`; if `|yaw_rate_radar − ω_gyro| > 3 °/s`
   for 15 consecutive frames → `gyro_ok = False` flag in `ego_info` (alert code 130, see §7); never use
   radar yaw as the operational yaw.
7. `info()` adds: `state`, `bin`, `yaw_rate` (used, rad/s), `yaw_rate_radar`, `gyro_ok`, `gyro_bias_dps`,
   `time_offset_s`, `n_inliers`, `n_points`, `az_spread_deg`, `ambiguous`.

Tracker / background: `Tracker.step(..., yaw_rate=ω_gyro)` unchanged; `BackgroundMap` becomes a
**short-memory ego buffer**: cells decay with `BACKGROUND_FORGET_S = 5 s` while moving, are translated by
`v·dt` and rotated by `ω·dt` each frame (existing `rotate` path), and are used as *persistence evidence*
(a static return seen ≥ N frames in the same cell raises confidence) — never as a suppression mask.

## 5. Fusion layer (`fusion-python-3.10.py`)

- `EgoLink`: parse the §2 sentence; keep a 2-s ring buffer of samples with Pi timestamps; `sample_at(t)`
  returns the linearly interpolated `{yaw_rate, pitch, roll, acc_fwd, gyro_ok, age_s}` for
  `t + EGO_TIME_OFFSET_S`; bias removal with `EGO_GYRO_BIAS_DPS`; `gyro_ok = gyro_cal ≥ 2 and age ≤ 0.2 s`.
- Standstill bias refresh: when the radar reports `STANDING` for ≥ 3 s, average the raw gyro Z and store
  it (`configs.json` write-back on clean exit).
- Corridor: half-width `CORRIDOR_HALF_WIDTH_M` (track + implement/2 + 0.3 m margin); look-ahead
  `max(6 m, v·TTC_WARN_S·1.3)`; when `v ≥ 0.7 m/s` bend the corridor with curvature `κ = ω/v`
  (lateral offset ≈ κ·y²/2), else a straight box.
- Static/moving label after compensation: keep `STATIC_DOPPLER_MPS` as the per-point threshold but derive
  it from the profile: `max(0.2 m/s, 1.2·bin)` per point (≈ 3σ of the ±½-bin floor), and a cluster-level
  label at `0.8·bin` (median of ≥ 3 points); a tangential mover (person crossing) has zero Doppler and is
  caught only by the tracker's position-derived speed.
- Alerts (`alerts.py`): TTC tiers `TTC_WARN_S = 3.5`, `TTC_CRITICAL_S = 1.8` computed with the
  ego-compensated closing speed of the track; when `state == STANDING` alerts are zone-occupancy only.
  A static object inside the corridor is an obstacle by default (warn tier once confirmed by persistence
  ≥ 3 frames or by the camera), never a "background" drop.

## 6. Logging, replay, calibration tools

- `record_sync.py`: logs radar frames (existing), `$RDEGO` lines with Pi receive time, camera detections;
  one `.jsonl` per run, gzip; `--tag` names the drill.
- `fusion_offline.py --log <file>`: replays with the same code path (`Pipeline.process` + `EgoLink`
  fed from the log), prints the per-frame `ego_info` and a summary (mean/median v per segment, state
  histogram, false-moving rate on points the user marks static via `--static-mask`).
- `tools/calib_time_offset.py <log>`: (a) replay grid search over `EGO_TIME_OFFSET_S ∈ [−0.4, +0.1] s`
  step 10 ms minimising the residual of the lever-arm-compensated fit on turning frames, (b)
  cross-correlation of gyro yaw rate vs radar yaw rate (2-parameter fit) as a check; prints both and the
  frame period; pass criterion: |a − b| < 1 frame.
- `tools/calib_mount.py <log_fwd> <log_back>`: mounting yaw from straight drives.
- `tools/drills.md`: the checklist for the tractor session with pass/fail numbers (from the report).

## 7. Alert codes added (see ALERTS.md)

- 130 — IMU degraded (stale, uncalibrated, or radar/gyro yaw disagreement) — severity W.
- 131 — ego estimate lost (UNKNOWN > 2 s while not STANDING) — severity W.
- 140 stays: obstacle in corridor (now issued for static objects too).

## 8. Tests (`test_ego.py`, `tests/`)

Synthetic: (A) standing scene — state STANDING, v = 0; (B) straight drive at 0.3 / 1 / 4.5 m/s with
wrapped Doppler on the old profile vs unwrapped on hangar_v9 — |v − v_true| < 0.5·bin; (C) turn at
ω = 0.3 rad/s with lever arm 2 m — with gyro, |v − v_true| < 0.5·bin (fix the sign bug in the old test C:
`th = w*dt*k`); (D) one large mover covering 60 % of points — gate holds the previous value; (E) narrow
azimuth spread — `ambiguous`, hold; (F) `$RDEGO` parsing, checksum, interpolation and bias removal;
(G) `calib_time_offset` recovers an injected offset of −120 ms within 20 ms.
Recorded: `test_stationary_stand.bin` → STANDING ≥ 95 % of frames; `test_stationary_walk.bin` → the
walker is `moving`, the background static.
