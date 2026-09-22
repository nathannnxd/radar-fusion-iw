# Alert system — sending events to an external MCU

`alerts.py` watches the fused radar+camera state each frame and turns it into a small set of
coded, discrete events — the kind of thing a downstream microcontroller can act on (brake, sound a
horn, flash a light, trigger a safe-stop) without knowing anything about radar point clouds or
camera bounding boxes. Wired into the live pipeline in `fusion-python-3.10.py`'s `run_live()`,
right where `s["fused"]`/`dets`/`stale` are already available each frame — no duplicated state.

This doc is the quick-reference; `alerts.py`'s own module docstring has the full detail (wire
format, checksum, a generic C reference parser) and is the source of truth if the two ever drift.

## Why this exists

Three scenarios you asked for directly, plus a few more that fell out of data the pipeline was
already computing but wasn't surfacing as an actionable signal:

| # | Scenario | Alert code |
|---|---|---|
| 1 | Object detected by radar but not camera | `100 RADAR_ONLY_UNCONFIRMED` |
| 2 | Seen by both, then camera suddenly loses it (radar hold) | `101 CAMERA_LOST_HOLD` |
| 3 | Object too close — radar-only, camera-only, or fused | `110`/`111 PROXIMITY_WARNING`/`PROXIMITY_CRITICAL` |
| + | Object left the camera's view entirely, radar still (briefly) has it | `102 CAMERA_LOST_OUT_OF_FRAME` |
| + | Camera degraded by environment (haze/smoke/defocus/exposure), not just one occluded object | `103 CAMERA_DEGRADED` |
| + | Closing speed exceeds a threshold | `120 FAST_APPROACH` |
| + | Time-to-collision below a tier (`range / closing speed`) | `121`/`122 TTC_WARNING`/`TTC_CRITICAL` |
| + | IMU (gyro) can't be trusted — stale, uncalibrated, or contradicted by the radar yaw cross-check | `130 IMU_DEGRADED` |
| + | Carrier-speed estimate lost (UNKNOWN > 2 s while not standing) | `131 EGO_LOST` |
| + | Radar sensor itself has gone stale/silent | `140 RADAR_SENSOR_STALE` |
| + | Object (static or moving) inside the driving corridor, confirmed by persistence or the camera | `141 OBSTACLE_IN_CORRIDOR` |

Not exhaustive on purpose — `alerts.py`'s docstring explains how to add another code (pick a
number, a severity, call `self._emit(...)` from `AlertEngine.evaluate()`).

### Ego-motion rules (EGO_MOTION.md §5, §7)

`AlertEngine.evaluate(..., ego=out["ego"])` receives the radar pipeline's carrier-motion dict for
the frame (`state`, `v`, `gyro_ok`, ...). What changes with it:

- **TTC tiers** — `121 TTC_WARNING` at `range / closing ≤ TTC_WARN_S` (3.5 s), `122 TTC_CRITICAL`
  at `≤ TTC_CRITICAL_S` (1.8 s). `closing` is the track's closing speed relative to the carrier
  (the radar's radial speed already contains the carrier's own motion: a post straight ahead closes
  at exactly the driving speed, a person walking alongside at ~0). Closing slower than 0.3 m/s gives
  no TTC. While the estimator flags `ttc_bound` (state `CREEPING`, EGO_MOTION.md §4.3) that floor
  rises to `2 · bin` and `122` is held back to `121` — at 1..5 bins the radial speed is mostly
  quantisation noise, and a stationary post must not read as a 1.7 s collision.
- **STANDING → zone occupancy only** — while `ego["state"] == "STANDING"` **and** the unclamped fit
  `ego["raw"]` is under `EGO_STANDING_MAX_MPS` (0.15 m/s) the motion alerts (`120`, `121`, `122`,
  `141`) are suppressed; proximity (`110`/`111`) and the camera/radar state codes (`100`–`103`) keep
  working, so someone stepping next to a parked tractor is still reported. The `raw` condition
  matters because STANDING is a one-bin deadband: on `hangar_v9` (bin 0.27 m/s) a tractor creeping at
  0.25 m/s is still STANDING, and it must not lose its collision layer.
- **`130 IMU_DEGRADED`** (W, `obj_id=0`) — `ego["gyro_ok"]` is False: the `$EGOVEL` link is stale
  (> 0.2 s), the BNO08x gyro reports accuracy < 2, or the radar/gyro yaw cross-check disagreed for
  15 frames. Only evaluated when an IMU link is configured (`EGO_SERIAL_PORT` non-empty) — a
  radar-only setup is not "degraded", it just has no gyro.
- **`131 EGO_LOST`** (W, `obj_id=0`) — the carrier-speed estimate has been `UNKNOWN` for more than
  `EGO_LOST_S` (2 s) while not standing; corridor and TTC logic are effectively blind until it
  recovers.
- **`141 OBSTACLE_IN_CORRIDOR`** (W) — the fusion layer marks a track `obstacle` when it is inside
  the driving corridor (half-width `CORRIDOR_HALF_WIDTH_M`, look-ahead `max(6 m, v·TTC_WARN_S·1.3)`,
  bent by the gyro yaw rate when `v ≥ 0.7 m/s`) and has either persisted ≥ 3 frames or is
  camera-confirmed. A static object in the path is an obstacle by default — never a "background"
  drop. (EGO_MOTION.md §7 refers to this as "140 stays"; `140` was already `RADAR_SENSOR_STALE`
  here, so the corridor alert got the next free number, `141`.)

## Wire format

One line per event, NMEA-0183-style — chosen specifically because it's a decades-old, universally
documented plain-text sensor protocol, not tied to any MCU ecosystem:

```
$RDALT,<code>,<event>,<severity>,<obj_id>,<range_m>,<az_deg>,<speed_mps>,<flags>*<checksum>\r\n
```

- `event`: `S` start, `A` active (heartbeat while still true), `C` clear. **Wait for the explicit
  `C`** — a dropped line must never be mistaken for an all-clear.
- `severity`: `I` info, `W` warning, `C` critical.
- `obj_id`: the radar track id, `1000000 + camera_track_id` for a camera-only object (no radar
  pairing), or `0` for a sensor/system-level alert.
- `checksum`: two hex digits, XOR of every byte between `$` and `*` — the standard NMEA checksum,
  so even a bare-metal MCU with no parsing library can reject a corrupted line.

Example: `$RDALT,111,S,C,7,1.20,-8.40,-2.10,0*7B`

## Tuning (`configs.json`)

| Key | Default | Meaning |
|---|---|---|
| `ALERTS_ENABLED` | `1` | master on/off switch |
| `ALERT_SERIAL_PORT` | `""` | the MCU's port (e.g. `"COM8"`, `"/dev/ttyUSB1"`); empty = console-only, no hardware needed |
| `ALERT_BAUD` | `115200` | serial baud rate |
| `PROXIMITY_WARN_M` | `3.0` | → `110 PROXIMITY_WARNING` |
| `PROXIMITY_CRITICAL_M` | `1.5` | → `111 PROXIMITY_CRITICAL` |
| `CLOSING_SPEED_ALERT_MPS` | `2.0` | → `120 FAST_APPROACH` |
| `TTC_WARN_S` | `3.5` | time-to-collision → `121 TTC_WARNING`; also sizes the corridor look-ahead (EGO_MOTION.md §5) |
| `TTC_CRITICAL_S` | `1.8` | time-to-collision → `122 TTC_CRITICAL` |
| `EGO_SERIAL_PORT` | `""` | (from EGO_VELOCITY.md) non-empty = an IMU link is expected → `130 IMU_DEGRADED` is evaluated |
| `RADAR_ONLY_CONFIRM_S` | `1.0` | how long a radar-only object must persist before alerting (filters out single-frame clutter blips) |
| `ALERT_RESEND_S` | `2.0` | heartbeat interval for an alert that's still active |

## Wiring up your MCU (whichever one you land on)

1. Connect its RX to the Pi/laptop's TX (level-shift if one side is 3.3V and the other 5V).
2. Set `ALERT_SERIAL_PORT` / `ALERT_BAUD` in `configs.json`.
3. Parse lines split on `,`, verify the checksum, act on `code`/`event`/`severity`. See `alerts.py`'s
   docstring for a ~15-line reference parser in plain C that isn't tied to any specific board/HAL.
4. If your MCU doesn't speak serial at all (I2C/CAN/networked), everything hardware-specific is
   isolated in the `AlertSink` classes in `alerts.py` — write your own sink; `AlertEngine` and the
   wire format itself don't need to change.

## Testing without hardware

Leave `ALERT_SERIAL_PORT` empty — alerts still print to the console via `ConsoleAlertSink`, so you
can watch the sentences scroll by and confirm the logic before wiring up any actual MCU. If you do
set a port but nothing's plugged in yet, `SerialAlertSink` fails to open it, prints one warning, and
keeps running console-only — it never crashes the fusion pipeline over an unattached accessory.

## Not yet validated on real hardware

Verified with unit tests covering the start/active/clear state machine, checksum correctness, the
confirm-delay on radar-only alerts, and the trigger conditions of the ego-motion codes `130`, `131`,
`141`, the TTC tiers and the STANDING suppression (`tests/test_alerts.py`) — all against synthetic
data, no real MCU in this environment. The camera/proximity codes (`100`–`103`, `110`/`111`) are
exercised only through the fusion pipeline, not by a dedicated unit test. Worth confirming on real hardware: actual serial timing under
load (alerts share the same process as the camera/radar loop — sends are non-blocking with
`timeout=0`, but a very chatty scene with many simultaneous transitions could still queue up several
sentences in one frame), and your MCU's actual parsing of the checksum/format above.
