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
| + | Radar sensor itself has gone stale/silent | `140 RADAR_SENSOR_STALE` |

Not exhaustive on purpose — `alerts.py`'s docstring explains how to add another code (pick a
number, a severity, call `self._emit(...)` from `AlertEngine.evaluate()`).

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
confirm-delay on radar-only alerts, and every alert code's trigger condition — all against synthetic
data, no real MCU in this environment. Worth confirming on real hardware: actual serial timing under
load (alerts share the same process as the camera/radar loop — sends are non-blocking with
`timeout=0`, but a very chatty scene with many simultaneous transitions could still queue up several
sentences in one frame), and your MCU's actual parsing of the checksum/format above.
