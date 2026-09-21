# Ego-velocity sensor — ESP32 + BNO08x IMU

Measures the platform's own forward speed and feeds it into the radar pipeline's ego-motion
compensation, so tracked-object velocities can be understood in two ways: relative to the platform
(what matters for collision/closing-speed alerts) and relative to the ground (what matters for
telling "this object is actually moving" from "it only looks like it's moving because I am").

## Why this exists

The radar pipeline already had an ego-motion concept — `EGO_SPEED_MPS` in `iwr1642_live.py`,
originally "0 on the bench, from odometry on the tractor" — but it was a **hardcoded manual
constant**, and it only fed the point-level static/moving classification
(`points_to_detections()`'s `doppler_rel_mps`/`is_static`, used for the background clutter map).
It was **not** applied to the actual EKF track velocities — `Track.x[2], Track.x[3]`, and therefore
`radial_mps`/`speed_mps`, are computed from the raw (non-ego-compensated) Doppler measurement.

That's not actually a bug worth "fixing" by replacing those fields — a moving platform's collision-
avoidance and proximity alerting (see `ALERTS.md`'s `FAST_APPROACH`/`PROXIMITY_*`) correctly wants
**relative closing speed**: if the platform drives toward a stationary person at 2 m/s, that person's
2 m/s closing speed *is* the thing worth alerting on, regardless of the fact that the person themself
isn't moving in the world. Changing `radial_mps`/`speed_mps` to be ground-frame would make that alert
logic silently wrong (a stationary obstacle dead ahead would report ~0 m/s and never trigger
`FAST_APPROACH`).

So this feature adds ground-frame velocity as a **new, additional** field —
`Track.ground_velocity()` / `Track.ground_speed_mps()` in `iwr1642_live.py`, surfaced as
`ground_speed_mps` in each `fused` entry from `Fusion.on_radar()` — rather than touching the
existing relative-frame fields at all. Nothing that currently reads `radial_mps`/`speed_mps`/`vx_mps`
/`vy_mps` changes behavior.

## Hardware

- **Sensor:** Adafruit BNO08x breakout (absolute orientation IMU — accelerometer + gyroscope +
  onboard sensor fusion), I2C, address `0x4B`.
- **MCU:** ESP32 dev board, running `firmware/ego_velocity/ego_velocity.ino`.
- **Link to the Pi:** a dedicated hardware UART (`Serial2` on the ESP32, GPIO16/17 by default) wired
  directly to the Pi's GPIO UART pins — **not** the ESP32's USB port. Both boards run 3.3V logic, so
  this is a direct wire connection, no level shifter needed (unlike a 5V Arduino).

```
ESP32                          Raspberry Pi (40-pin header)
-----                          ----------------------------
GPIO17 (TX2)  ------------->   GPIO15 / pin 10 (RXD)
GPIO16 (RX2)  <-------------   GPIO14 / pin 8  (TXD)
GND           ------------->   GND (any ground pin)
```

Why a separate UART instead of the ESP32's `Serial`/USB: `Serial`/UART0 is also used by the onboard
USB-serial chip for flashing and the Arduino IDE's serial monitor — sharing it with the Pi link means
debugging over USB and the live data feed to the Pi can conflict. Keeping them on different UART
peripherals means you can plug the ESP32 into a laptop over USB for debug prints at the same time the
GPIO link to the Pi is running, with no contention.

On the Pi side, `EGO_SERIAL_PORT` in `configs.json` needs to point at whatever port that UART becomes
— on Pi OS this is typically `/dev/ttyS0` or `/dev/ttyAMA0` depending on which UART is enabled and
whether Bluetooth is using the primary one; see `RASPBERRY_PI.md`'s serial-port-discovery section for
the general approach (the commands there apply here too, this is just a third serial device).

## Firmware (`firmware/ego_velocity/ego_velocity.ino`)

Builds on the original `gyro-test.ino` (same I2C setup, same BNO08x init). What it adds:

1. **`SH2_LINEAR_ACCELERATION`** instead of just the rotation vector — this report is gravity-
   compensated by the BNO08x's own onboard sensor fusion, so no separate gravity-subtraction math is
   needed on the ESP32 side.
2. **Startup bias calibration** — averages the first ~2 s of readings (assumes the platform is
   stationary at power-on) and subtracts that as a zero-offset from every subsequent sample.
3. **Zero-velocity update (ZUPT)** — the actual drift-bounding mechanism. If both the bias-corrected
   forward acceleration and the yaw rate (`SH2_GYROSCOPE_CALIBRATED`) stay under a small threshold for
   a sustained window, the platform is declared stationary and the integrated velocity is pinned back
   to exactly 0 rather than continuing to integrate.
4. Sends `$EGOVEL,<speed_mps>,<seq>,<flags>*<checksum>\r\n` out `Serial2` on every new linear-
   acceleration sample (~50 Hz) — NMEA-0183-style, same design philosophy as `alerts.py`'s `$RDALT`
   sentences (plain text, checksummed, no parser library needed).

### The honest limitation

This is dead-reckoning: velocity = integral of acceleration, with no independent ground truth (no
wheel encoder, no GPS). ZUPT bounds the error to zero every time the platform genuinely stops, which
is why it's there — but between stops, on a long uninterrupted drive, expect real drift from residual
accelerometer bias and noise. This is good enough to inform Doppler ego-compensation and the
`ground_speed_mps` classification signal (a few tenths of a m/s of error there just nudges a
static/moving judgment) — it is **not** a substitute for a wheel encoder or GPS if you need accurate
absolute speed over a long, continuous drive.

### Mounting requirement

The firmware assumes the BNO08x's own +X axis is rigidly mounted pointing in the direction of travel,
and uses that axis directly — it does **not** use the orientation quaternion to figure out "forward"
dynamically. Mount the board so its X axis points forward. A more robust version that uses the
quaternion to project acceleration onto the true direction of travel regardless of mounting angle is a
reasonable future improvement, left undone here to keep the first version simple to reason about and
debug.

## Pi side (`ego_velocity.py`)

`EgoVelocityReader` — a background thread reading `$EGOVEL` sentences from the configured serial
port, exposing `.speed_mps` (the latest value, or `0.0` if the port never opened / the link has gone
stale). Same defensive philosophy as `alerts.py`'s `SerialAlertSink`: an unattached or misconfigured
sensor never crashes the pipeline, it just silently degrades to exactly the old default behavior
(`EGO_SPEED_MPS = 0.0`).

## Integration points

- `iwr1642_live.Pipeline.ego` is now updated live, once per radar frame, in both `run_live()`
  (`fusion-python-3.10.py`) and standalone `main()` (`iwr1642_live.py`) — `pipe.ego =
  ego_reader.speed_mps` right before `pipe.process(frame)`. This feeds the existing Doppler
  ego-compensation (`points_to_detections`) a live value instead of the old fixed constant.
- `Pipeline.process()`'s returned dict now includes `"ego_speed_mps"` (the value used for that
  frame), so callers don't need to separately track it.
- `Track.ground_velocity(ego_speed_mps)` / `Track.ground_speed_mps(ego_speed_mps)` — new methods,
  additive only (see "Why this exists" above for why the existing relative-frame fields are
  untouched).
- `Fusion.on_radar(..., ego_speed_mps=0.0)` — new optional parameter (default preserves existing
  behavior for `fusion_offline.py`/`selftest()`, which don't pass it); each `fused` entry now
  includes `ground_speed_mps`.
- `draw_overlay(..., ego_speed_mps=None)` — shows the live ego speed in the corner when enabled, and
  appends `(ground X.X)` next to an object's relative speed label when ego speed is non-negligible.

## Configuration (`configs.json`)

| Key | Default | Meaning |
|---|---|---|
| `EGO_SERIAL_PORT` | `""` | the ESP32's UART port; empty = disabled, `EGO_SPEED_MPS` stays `0.0` (today's behavior) |
| `EGO_BAUD` | `115200` | must match the firmware's `PI_UART_BAUD` |

## Testing without hardware

Leave `EGO_SERIAL_PORT` empty — everything else works exactly as before (ego speed reads `0.0`,
`ground_speed_mps` equals the existing relative speed). Set the port once you have the ESP32 wired up
and flashed; `EgoVelocityReader` will start populating live values without any other code change.

## Not yet validated on real hardware

Verified with unit tests: NMEA checksum accept/reject, staleness fallback, sequence-drop detection
(including 255→0 wraparound), a sentence split across two reads, and the ego-compensation math itself
(`Track.ground_velocity()` against hand-computed expected values). No physical ESP32/BNO08x/Pi UART
link available in this environment — worth confirming on real hardware: the actual UART wiring/port
name on your specific Pi image, real-world ZUPT threshold tuning (the values in the firmware are a
reasonable starting point, not derived from real accelerometer noise data), and whether 50 Hz over a
plain wire at 115200 baud holds up next to the radar's own USB-serial traffic.
