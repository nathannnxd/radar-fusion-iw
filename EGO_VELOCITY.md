# Ego-velocity sensor — ESP32 + BNO08x IMU

Measures the platform's own planar motion — forward velocity, lateral velocity, AND yaw
(rotation) rate — and feeds it into the radar pipeline's ego-motion compensation, so
tracked-object velocities can be understood in two ways: relative to the platform (what matters
for collision/closing-speed alerts) and relative to the ground (what matters for telling "this
object is actually moving" from "it only looks like it's moving because I am").

Yaw rate specifically exists to cover something the radar cannot measure at all: a single radar
point only ever gives a radial (range-rate) measurement, so there is no way to recover the
platform's own rotation rate from radar data alone. This sensor is the only source for it in this
project.

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
`ground_speed_mps`/`ground_vx_mps`/`ground_vy_mps` in each `fused` entry from `Fusion.on_radar()` —
rather than touching the existing relative-frame fields at all. Nothing that currently reads
`radial_mps`/`speed_mps`/`vx_mps`/`vy_mps` changes behavior.

A second, separate design decision worth calling out: `Pipeline.ego` — the single scalar that feeds
`points_to_detections()`'s Doppler compensation *and* the LightGBM model's trained `ego_speed_mps`
feature — stays exactly the forward-only scalar it always was, now sourced from the sensor's forward
component (`EgoVelocityReader.vy_mps`) rather than a fixed constant. It deliberately does **not**
pick up the new lateral velocity or yaw rate, because the model was trained expecting that specific
single-scalar-forward semantics; silently changing what's fed into it would corrupt point
classification in a way that's hard to notice. The new lateral velocity and yaw rate are surfaced
everywhere else instead (`Track.ground_velocity()`'s full 2D correction, `Pipeline.ego_yaw_rate_dps`,
the on-screen readout, the JSONL log) — additive, not retrofitted into the one place that can't
safely absorb a semantic change.

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

1. **`SH2_LINEAR_ACCELERATION`** (both X and Y axes) instead of just the rotation vector — this
   report is gravity-compensated by the BNO08x's own onboard sensor fusion, so no separate
   gravity-subtraction math is needed on the ESP32 side.
2. **`SH2_GYROSCOPE_CALIBRATED`** — read directly and sent as-is (deg/s). This is the actual point of
   the file: yaw rate is a direct rate measurement, not an integral, so unlike forward/lateral
   velocity it does not accumulate dead-reckoning drift the same way (see "The honest limitation"
   below for the nuance).
3. **Startup bias calibration** — averages the first ~2 s of forward/lateral acceleration readings
   (assumes the platform is stationary at power-on) and subtracts that as a zero-offset from every
   subsequent sample.
4. **Zero-velocity update (ZUPT)** — the actual drift-bounding mechanism for the velocity estimate.
   If the bias-corrected forward acceleration, lateral acceleration, AND yaw rate all stay under a
   small threshold for a sustained window, the platform is declared stationary and both velocity
   components are pinned back to exactly 0 rather than continuing to integrate.
5. Sends `$EGOVEL,<vx_mps>,<vy_mps>,<yaw_rate_dps>,<seq>,<flags>*<checksum>\r\n` out `Serial2` on
   every new linear-acceleration sample (~50 Hz) — NMEA-0183-style, same design philosophy as
   `alerts.py`'s `$RDALT` sentences (plain text, checksummed, no parser library needed).

### The honest limitation

Forward/lateral velocity is dead-reckoning: velocity = integral of acceleration, with no independent
ground truth (no wheel encoder, no GPS). ZUPT bounds the error to zero every time the platform
genuinely stops, which is why it's there — but between stops, on a long uninterrupted drive, expect
real drift from residual accelerometer bias and noise. This is good enough to inform Doppler
ego-compensation and the `ground_speed_mps` classification signal (a few tenths of a m/s of error
there just nudges a static/moving judgment) — it is **not** a substitute for a wheel encoder or GPS
if you need accurate absolute speed over a long, continuous drive.

Yaw *rate* doesn't have that specific problem (it's a direct reading, not an integral) — but this
firmware does **not** integrate it into a heading/orientation angle, only the instantaneous rate is
sent. Turning "current rotation rate" into "how far has the platform actually turned since frame N"
(which is what you'd need to correct *positions*, not just velocities, during a turn) would require
integrating this rate over time on top, and isn't attempted here.

### Mounting requirement

The firmware assumes the BNO08x is rigidly mounted with its own +X axis pointing in the direction of
travel (forward) and +Y axis pointing right — matching this project's existing radar convention
(x = right, y = forward). It does **not** use the orientation quaternion to figure out orientation
dynamically. If your board's axes don't cooperate with that mounting, `FORWARD_SIGN`/`RIGHT_SIGN` at
the top of the `.ino` can flip either axis's sign instead of requiring a physical remount — see the
firmware's own header comment for how to determine which way to set them. A more robust version that
uses the quaternion to project acceleration onto the true direction of travel regardless of mounting
angle is a reasonable future improvement, left undone here to keep this version simple to reason
about and debug.

## Pi side (`ego_velocity.py`)

`EgoVelocityReader` — a background thread reading `$EGOVEL` sentences from the configured serial
port, exposing `.vx_mps` (lateral), `.vy_mps` (forward), `.yaw_rate_dps`, and `.speed_mps`
(convenience magnitude of vx/vy) — all falling back to `0.0` if the port never opened or the link
has gone stale. Same defensive philosophy as `alerts.py`'s `SerialAlertSink`: an unattached or
misconfigured sensor never crashes the pipeline, it just silently degrades to exactly the old
default behavior (`EGO_SPEED_MPS = 0.0`, no rotation).

## Integration points

- `iwr1642_live.Pipeline.ego` is now updated live, once per radar frame, in both `run_live()`
  (`fusion-python-3.10.py`) and standalone `main()` (`iwr1642_live.py`) — `pipe.ego =
  ego_reader.vy_mps` (forward component only — see "why this exists" above for why not the
  magnitude or the lateral component) right before `pipe.process(frame)`. This feeds the existing
  Doppler ego-compensation (`points_to_detections`) a live value instead of the old fixed constant.
- `Pipeline.ego_yaw_rate_dps` — new attribute, also updated live (`pipe.ego_yaw_rate_dps =
  ego_reader.yaw_rate_dps`), carried through only — not fed into `points_to_detections`/the model.
- `Pipeline.process()`'s returned dict now includes `"ego_speed_mps"` and `"ego_yaw_rate_dps"` (the
  values used for that frame), so callers don't need to separately track them.
- `Track.ground_velocity(ego_vx_mps=0.0, ego_vy_mps=0.0)` / `Track.ground_speed_mps(...)` — now take
  the full 2D ego-velocity instead of a forward-only scalar, since both components are actually
  available now. Additive only (see "Why this exists" above for why the existing relative-frame
  fields are untouched).
- `Fusion.on_radar(..., ego_vx_mps=0.0, ego_vy_mps=0.0)` — replaces the previous single
  `ego_speed_mps` parameter (defaults preserve existing behavior for `fusion_offline.py`/
  `selftest()`, which don't pass either); each `fused` entry now includes `ground_speed_mps`,
  `ground_vx_mps`, and `ground_vy_mps`.
- `draw_overlay(..., ego_speed_mps=None, ego_yaw_rate_dps=None)` — shows the live ego speed and yaw
  rate in the corner when enabled, and appends `(ground X.X)` next to an object's relative speed
  label when ego speed is non-negligible.

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

Verified with unit tests: NMEA checksum accept/reject on the full 5-field sentence, staleness
fallback (independently for vx/vy/yaw_rate/speed), sequence-drop detection (including 255→0
wraparound), rejection of the previous 4-field sentence format (so a firmware/reader version
mismatch fails loudly rather than silently misparsing), and the ego-compensation math itself
(`Track.ground_velocity()` with both forward-only and full 2D corrections, against hand-computed
expected values) — plus an end-to-end wiring smoke test through the real `Pipeline` → `Track` →
`Fusion.on_radar()` → `draw_overlay()` chain. No physical ESP32/BNO08x/Pi UART link available in this
environment — worth confirming on real hardware: the actual UART wiring/port name on your specific Pi
image, real-world ZUPT threshold tuning for both accel axes (the values in the firmware are a
reasonable starting point, not derived from real accelerometer noise data), whether the BNO08x's Y
axis really does read "right" on your specific mounting (confirm before trusting the sign, per the
mounting-requirement note above), and whether 50 Hz over a plain wire at 115200 baud holds up next to
the radar's own USB-serial traffic.
