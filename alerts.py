"""alerts.py — safety/situational-awareness alert system for the fusion pipeline.

Turns the fused radar+camera state into a small set of discrete, coded events that an external
MCU can act on (trigger braking, a warning light, a horn, a safe-stop, ...) without needing to
understand radar point clouds, camera bounding boxes, or any of this project's own data structures.

    engine = AlertEngine(sinks=[...], **thresholds_from_configs_json)
    ...
    engine.evaluate(t, fused, cam_dets, cam, radar_stale)   # once per camera frame, in run_live()

--------------------------------------------------------------------------------------------------
WIRE PROTOCOL — for whatever MCU ends up on the other end

One ASCII "sentence" per alert event, NMEA-0183-style (the format GPS/marine electronics have used
for decades). Deliberately chosen over JSON or a custom binary struct: it's plain text (readable on
a serial monitor while debugging), doesn't need a parser library, and just about every embedded
ecosystem already has example code for splitting one of these — you can point at any NMEA parsing
tutorial and it transfers directly, regardless of which MCU you end up choosing.

    $RDALT,<code>,<event>,<severity>,<obj_id>,<range_m>,<az_deg>,<speed_mps>,<flags>*<checksum>\r\n

  code      - numeric AlertCode, see the table below
  event     - 'S' start   — condition just became true, first notice
              'A' active  — still true, periodic heartbeat every ALERT_RESEND_S while it persists
              'C' clear   — condition just became false — IMPORTANT: the MCU should treat "no more
                            sentences" as "still active", not as cleared — always wait for the
                            explicit 'C' (e.g. a dropped/garbled line must not look like an all-clear)
  severity  - 'I' info, 'W' warning, 'C' critical
  obj_id    - the radar track id this alert is about, 1000000+camera-track-id for a camera-only
              object with no radar pairing, or 0 for a sensor/system-level alert
  range_m / az_deg / speed_mps - decimal, or an empty field if not applicable to this alert code
  flags     - reserved for future use, currently always 0
  checksum  - two uppercase hex digits: XOR of every byte between '$' and '*' (the standard NMEA
              checksum) — lets even a bare-metal MCU with no library at all reject a line that got
              corrupted on the wire, just by re-XORing what it received

Example:  $RDALT,111,S,C,7,1.20,-8.40,-2.10,0*7B\r\n
          (alert 111 = PROXIMITY_CRITICAL just started, track #7, 1.20 m away, -8.4° azimuth,
           closing at 2.10 m/s)

--------------------------------------------------------------------------------------------------
ALERT CODES

 code | name                     | severity | meaning
 -----|--------------------------|----------|----------------------------------------------------
 100  | RADAR_ONLY_UNCONFIRMED   | info     | an object is tracked by radar but has never been
      |                          |          | confirmed by the camera (state == "radar-only")
 101  | CAMERA_LOST_HOLD         | warning  | was seen by both sensors, camera has now lost it —
      |                          |          | radar is still tracking (state == "hold")
 102  | CAMERA_LOST_OUT_OF_FRAME | info     | as above, but the object has now left the camera's
      |                          |          | field of view entirely (state == "out-of-frame")
 103  | CAMERA_DEGRADED          | warning  | camera lost an object specifically because of an
      |                          |          | environmental cause (haze/smoke/defocus, exposure) —
      |                          |          | not just this one object being occluded
 110  | PROXIMITY_WARNING        | warning  | an object is within PROXIMITY_WARN_M — from radar,
      |                          |          | camera-only range estimate, or fused, whichever sees it
 111  | PROXIMITY_CRITICAL       | critical | as above, within the tighter PROXIMITY_CRITICAL_M
 120  | FAST_APPROACH            | warning  | an object's closing speed exceeds CLOSING_SPEED_ALERT_MPS
 140  | RADAR_SENSOR_STALE       | critical | the radar hasn't produced a frame in RADAR_STALE_S —
      |                          |          | system is effectively blind on that sensor (obj_id=0)

This is not an exhaustive list by design — see the module docstring above for how to add another
condition: pick a code, a severity, and call self._emit(...) from evaluate().

--------------------------------------------------------------------------------------------------
INTEGRATING A NEW MCU

There's no assumption anywhere in this file about which microcontroller is listening. To add one:
  1. Wire its RX pin to the Pi/laptop's TX (through a level shifter if the MCU is 3.3V and the
     serial adapter is 5V, or vice versa — check both boards' logic levels).
  2. Set ALERT_SERIAL_PORT / ALERT_BAUD in configs.json to that port.
  3. On the MCU side, read lines terminated by '\r\n', split on ',', verify the checksum, and act on
     `code`/`event`/`severity`. A minimal reference parser (deliberately generic C, not tied to any
     specific board/HAL):

    void on_line(char *line) {                  // line = "$RDALT,111,S,C,7,1.20,-8.40,-2.10,0*7B"
        if (line[0] != '$') return;
        char *star = strchr(line, '*');
        if (!star) return;
        uint8_t cs = 0;
        for (char *p = line + 1; p < star; p++) cs ^= (uint8_t)*p;
        uint8_t recv_cs = (uint8_t)strtol(star + 1, NULL, 16);
        if (cs != recv_cs) return;               // corrupted line, drop it
        *star = '\0';
        char *field = strtok(line + 1, ",");      // "RDALT"
        int code = atoi(strtok(NULL, ","));
        char event = strtok(NULL, ",")[0];        // 'S' / 'A' / 'C'
        char severity = strtok(NULL, ",")[0];     // 'I' / 'W' / 'C'
        int obj_id = atoi(strtok(NULL, ","));
        // ... remaining fields: range_m, az_deg, speed_mps, flags
        if (code == 111 && event != 'C') trigger_brake();
        if (code == 111 && event == 'C') release_brake();
    }

If serial isn't the right transport for your MCU (e.g. it only speaks I2C/CAN, or it's actually
another networked board), everything MCU-specific is isolated in the AlertSink classes below —
swap SerialAlertSink for your own sink; AlertEngine and the wire format don't need to change.
"""
from dataclasses import dataclass
from typing import Optional

try:
    import serial
except ImportError:
    serial = None


# ---------------------------------------------------------------- alert codes
class AlertCode:
    RADAR_ONLY_UNCONFIRMED = 100
    CAMERA_LOST_HOLD = 101
    CAMERA_LOST_OUT_OF_FRAME = 102
    CAMERA_DEGRADED = 103
    PROXIMITY_WARNING = 110
    PROXIMITY_CRITICAL = 111
    FAST_APPROACH = 120
    RADAR_SENSOR_STALE = 140


SEVERITY = {
    AlertCode.RADAR_ONLY_UNCONFIRMED: "I",
    AlertCode.CAMERA_LOST_HOLD: "W",
    AlertCode.CAMERA_LOST_OUT_OF_FRAME: "I",
    AlertCode.CAMERA_DEGRADED: "W",
    AlertCode.PROXIMITY_WARNING: "W",
    AlertCode.PROXIMITY_CRITICAL: "C",
    AlertCode.FAST_APPROACH: "W",
    AlertCode.RADAR_SENSOR_STALE: "C",
}

CAMERA_DEGRADED_REASONS = {"haze / smoke / defocus", "over/under-exposed"}


# ---------------------------------------------------------------- wire format
@dataclass
class Alert:
    t: float
    code: int
    event: str            # "S" | "A" | "C"
    severity: str          # "I" | "W" | "C"
    obj_id: int
    range_m: Optional[float] = None
    az_deg: Optional[float] = None
    speed_mps: Optional[float] = None

    def to_sentence(self):
        def fmt(x):
            return "" if x is None else f"{x:.2f}"
        body = (f"RDALT,{self.code},{self.event},{self.severity},{self.obj_id},"
                f"{fmt(self.range_m)},{fmt(self.az_deg)},{fmt(self.speed_mps)},0")
        checksum = 0
        for ch in body:
            checksum ^= ord(ch)
        return f"${body}*{checksum:02X}\r\n"


# ---------------------------------------------------------------- sinks (where alerts go)
class AlertSink:
    def send(self, line):
        raise NotImplementedError

    def close(self):
        pass


class ConsoleAlertSink(AlertSink):
    """Prints every alert sentence — useful standalone for debugging without any MCU attached,
    and as a permanent audit trail alongside whatever hardware sink is also configured."""
    def send(self, line):
        print(line.strip())


class SerialAlertSink(AlertSink):
    """Sends each alert sentence out a UART to an external MCU. Opens defensively: if the port
    can't be opened (not attached yet, wrong path), this prints a one-time warning and silently
    no-ops on every send afterward, rather than crashing the whole fusion pipeline over what is,
    from the pipeline's perspective, an optional accessory."""

    def __init__(self, port, baud=115200):
        self.ser = None
        if serial is None:
            print("⚠️ pyserial not available — alert serial sink disabled, alerts will only print to console")
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=0)
        except Exception as e:
            print(f"⚠️ alert serial port {port} not opened ({e}) — alerts will only print to console")

    def send(self, line):
        if self.ser is not None:
            try:
                self.ser.write(line.encode("ascii"))
            except Exception as e:
                print(f"⚠️ alert serial write failed ({e})")

    def close(self):
        if self.ser is not None:
            self.ser.close()


# ---------------------------------------------------------------- engine
class AlertEngine:
    """Evaluates alert conditions once per camera frame and emits start/active/clear events with
    debouncing built in: a condition fires once when it becomes true ('S'), re-fires as a heartbeat
    ('A') every resend_s while it stays true (so a downstream MCU doesn't need to guess whether the
    link is still alive), and fires exactly once ('C') when it becomes false. Without this, every
    per-frame condition check would otherwise flood the MCU with dozens of identical messages a
    second."""

    def __init__(self, sinks, resend_s=2.0, radar_only_confirm_s=1.0,
                 proximity_warn_m=3.0, proximity_critical_m=1.5, closing_speed_mps=2.0):
        self.sinks = sinks
        self.resend_s = resend_s
        self.radar_only_confirm_s = radar_only_confirm_s
        self.proximity_warn_m = proximity_warn_m
        self.proximity_critical_m = proximity_critical_m
        self.closing_speed_mps = closing_speed_mps
        self._active = {}              # (code, obj_id) -> {"last_sent": t}
        self._radar_only_since = {}    # obj_id -> t first seen as radar-only, unconfirmed

    def close(self):
        for sink in self.sinks:
            sink.close()

    def _emit(self, t, code, obj_id, cond_true, range_m=None, az_deg=None, speed_mps=None):
        key = (code, obj_id)
        st = self._active.get(key)
        if cond_true:
            if st is None:
                self._active[key] = {"last_sent": t}
                self._send(Alert(t, code, "S", SEVERITY[code], obj_id, range_m, az_deg, speed_mps))
            elif t - st["last_sent"] >= self.resend_s:
                st["last_sent"] = t
                self._send(Alert(t, code, "A", SEVERITY[code], obj_id, range_m, az_deg, speed_mps))
        elif st is not None:
            del self._active[key]
            self._send(Alert(t, code, "C", SEVERITY[code], obj_id, range_m, az_deg, speed_mps))

    def _send(self, alert):
        line = alert.to_sentence()
        for sink in self.sinks:
            sink.send(line)

    def evaluate(self, t, fused, cam_dets, cam, radar_stale):
        """t: clock() timestamp: fused/cam_dets/radar_stale: the same values already computed each
        frame in run_live() (s["fused"], dets, and stale) — this reuses them, it doesn't recompute
        anything from scratch."""
        # system health: is the radar sensor itself still alive?
        self._emit(t, AlertCode.RADAR_SENSOR_STALE, 0, radar_stale)

        seen_radar_only = set()
        for f in fused:
            if f.get("absorbed_by") is not None:
                continue                                    # a merged duplicate, not a separate object
            obj_id = f["radar_id"]
            state = f.get("state")
            range_m = f.get("range_near_m")
            az = f.get("az_from_cam")
            speed = f.get("radial_mps")
            closing = -speed if speed is not None else None

            # 1) "detected by radar but not camera" — never confirmed by the camera at all. Gated by
            # a short confirm delay (radar_only_confirm_s) so a single-frame clutter blip upstream
            # doesn't turn into an MCU alert.
            if state == "radar-only":
                seen_radar_only.add(obj_id)
                first_t = self._radar_only_since.setdefault(obj_id, t)
                cond = (t - first_t) >= self.radar_only_confirm_s
            else:
                self._radar_only_since.pop(obj_id, None)
                cond = False
            self._emit(t, AlertCode.RADAR_ONLY_UNCONFIRMED, obj_id, cond, range_m, az, speed)

            # 2) "seen by both, then suddenly disappeared from camera" — the radar-hold scenario
            self._emit(t, AlertCode.CAMERA_LOST_HOLD, obj_id, state == "hold", range_m, az, speed)

            # object fully left the frame while radar still (briefly) has it
            self._emit(t, AlertCode.CAMERA_LOST_OUT_OF_FRAME, obj_id, state == "out-of-frame",
                       range_m, az, speed)

            # camera degraded (haze/exposure) rather than just this one object being occluded
            degraded = state == "hold" and f.get("lost_reason") in CAMERA_DEGRADED_REASONS
            self._emit(t, AlertCode.CAMERA_DEGRADED, obj_id, degraded, range_m, az, speed)

            # 3) proximity — covers "as seen by radar only or the fusion" (camera-only handled below)
            in_range = range_m is not None and state != "out-of-frame"
            self._emit(t, AlertCode.PROXIMITY_WARNING, obj_id,
                       in_range and range_m <= self.proximity_warn_m, range_m, az, speed)
            self._emit(t, AlertCode.PROXIMITY_CRITICAL, obj_id,
                       in_range and range_m <= self.proximity_critical_m, range_m, az, speed)

            # closing speed
            self._emit(t, AlertCode.FAST_APPROACH, obj_id,
                       closing is not None and closing >= self.closing_speed_mps, range_m, az, speed)

        # drop radar-only bookkeeping for any id no longer present at all this frame
        for obj_id in list(self._radar_only_since):
            if obj_id not in seen_radar_only:
                self._radar_only_since.pop(obj_id, None)

        # 3b) proximity — "as seen by ... camera only": detections with no radar pairing at all
        fused_cam_ids = {f["cam_id"] for f in fused if f.get("state") == "both"}
        for d in cam_dets:
            if d.cam_id in fused_cam_ids:
                continue                                      # already covered via the fused loop above
            r_cam = cam.range_from_bbox(d.bh, d.obj_h_m)
            cam_obj_id = 1_000_000 + d.cam_id                 # disjoint id range from radar track ids
            self._emit(t, AlertCode.PROXIMITY_WARNING, cam_obj_id, r_cam <= self.proximity_warn_m, r_cam)
            self._emit(t, AlertCode.PROXIMITY_CRITICAL, cam_obj_id, r_cam <= self.proximity_critical_m, r_cam)
