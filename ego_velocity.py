"""ego_velocity.py — reads the platform's own planar motion (forward velocity, lateral velocity,
and yaw rate) from an external IMU (ESP32 + BNO08x, see firmware/ego_velocity/ego_velocity.ino)
over a UART/GPIO link. Forward velocity feeds the radar pipeline's existing Doppler ego-
compensation (iwr1642_live.points_to_detections) and Track.ground_velocity()/ground_speed_mps();
yaw rate exists because the radar cannot measure the platform's own rotation rate at all — a
single radar point only gives a radial measurement, so this is the only source for it in this
project. See EGO_VELOCITY.md for the full design/rationale.

    reader = EgoVelocityReader(EGO_SERIAL_PORT, EGO_BAUD)
    ...
    pipe.ego = reader.vy_mps                 # forward component — once per radar frame, before pipe.process(frame)
    pipe.ego_yaw_rate_dps = reader.yaw_rate_dps
    ...
    reader.close()

--------------------------------------------------------------------------------------------------
WIRE PROTOCOL

Same NMEA-0183-style ASCII sentence approach as alerts.py, for the same reasons (plain text,
checksummed, no parser library needed, MCU-agnostic) — this is the mirror-image direction: sensor
data coming INTO the Pi from the ESP32, instead of alerts going out to an MCU.

    $EGOVEL,<vx_mps>,<vy_mps>,<yaw_rate_dps>,<seq>,<flags>*<checksum>\r\n

  vx_mps       - lateral (right-positive) velocity estimate, from integrating linear acceleration
  vy_mps       - forward velocity estimate, from integrating linear acceleration
                 both signed, from integrating the IMU's linear acceleration with drift
                 correction — see the firmware's own header comment for exactly how, and its real
                 limitations (this is dead-reckoning, not a substitute for a wheel encoder or GPS
                 over any long horizon)
  yaw_rate_dps - direct gyroscope reading (deg/s, + = counterclockwise viewed from above), NOT
                 integrated — unlike vx/vy this does not accumulate dead-reckoning drift over time,
                 though it's still only as accurate as the gyro's own bias calibration
  seq          - 0-255, wraps around; lets this reader notice dropped/corrupted lines without
                 needing synchronized clocks on both ends
  flags        - bit 0: ZUPT (zero-velocity update) is currently active, i.e. the firmware believes
                 the platform is stationary and is holding vx/vy at 0 rather than integrating
                 drift. Other bits reserved.
  checksum     - two uppercase hex digits, XOR of every byte between '$' and '*' (standard NMEA
                 checksum) — a corrupted line is dropped outright, never guessed at

Example:  $EGOVEL,0.05,1.35,-2.10,42,0*3E\r\n
"""
import math
import threading
import time

try:
    import serial
except ImportError:
    serial = None


class EgoVelocityReader:
    """Background thread reading $EGOVEL sentences from the ESP32+IMU over a serial/GPIO UART link.
    Degrades gracefully — matching alerts.py's SerialAlertSink — rather than ever crashing the radar
    pipeline over an optional accessory: if the port can't be opened, or the link goes stale (no
    valid sentence for stale_s), every property reads back 0.0 (i.e. "assume stationary, not
    turning"), which is the same as the pipeline's old hardcoded EGO_SPEED_MPS = 0.0 default — so a
    missing/disconnected sensor silently falls back to exactly today's behavior, not a crash and not
    a stale/wrong value held forever."""

    def __init__(self, port, baud=115200, stale_s=1.0):
        self.stale_s = stale_s
        self._vx = 0.0
        self._vy = 0.0
        self._yaw_rate = 0.0
        self._last_t = None
        self._last_seq = None
        self._dropped = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.ser = None
        if not port:
            return
        if serial is None:
            print("⚠️ pyserial not available — ego-velocity input disabled, EGO_SPEED_MPS stays 0.0")
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=0.2)
        except Exception as e:
            print(f"⚠️ ego-velocity serial port {port} not opened ({e}) — EGO_SPEED_MPS stays 0.0")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(256)
            except Exception:
                time.sleep(0.1)
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._handle_line(line.strip())
            if len(buf) > 4096:
                buf = buf[-256:]                          # a stuck/noisy line with no '\n' — don't grow forever

    def _handle_line(self, line):
        try:
            text = line.decode("ascii")
        except UnicodeDecodeError:
            return
        if not text.startswith("$") or "*" not in text:
            return
        body, _, checksum_hex = text[1:].partition("*")
        try:
            checksum = int(checksum_hex, 16)
        except ValueError:
            return
        cs = 0
        for ch in body:
            cs ^= ord(ch)
        if cs != checksum:
            return                                        # corrupted on the wire — drop, never guess
        fields = body.split(",")
        if len(fields) < 5 or fields[0] != "EGOVEL":
            return
        try:
            vx = float(fields[1])
            vy = float(fields[2])
            yaw_rate = float(fields[3])
            seq = int(fields[4])
        except ValueError:
            return
        with self._lock:
            if self._last_seq is not None:
                expected = (self._last_seq + 1) % 256
                if seq != expected:
                    self._dropped += (seq - expected) % 256
            self._last_seq = seq
            self._vx = vx
            self._vy = vy
            self._yaw_rate = yaw_rate
            self._last_t = time.time()

    def _stale_locked(self):
        """Caller must hold self._lock."""
        return self._last_t is None or (time.time() - self._last_t) > self.stale_s

    @property
    def vx_mps(self):
        """Lateral (right-positive) ego velocity. 0.0 if the link is down/stale."""
        with self._lock:
            return 0.0 if self._stale_locked() else self._vx

    @property
    def vy_mps(self):
        """Forward ego velocity — this is what feeds Pipeline.ego (the existing Doppler
        ego-compensation and the LightGBM model's trained ego_speed_mps feature both expect a
        forward-only scalar, not a magnitude — see EGO_VELOCITY.md). 0.0 if the link is down/stale."""
        with self._lock:
            return 0.0 if self._stale_locked() else self._vy

    @property
    def speed_mps(self):
        """Magnitude of (vx, vy) — convenience for callers that only want a scalar speed, e.g. an
        on-screen readout. Do NOT use this for Pipeline.ego; use vy_mps (see its docstring)."""
        with self._lock:
            return 0.0 if self._stale_locked() else math.hypot(self._vx, self._vy)

    @property
    def yaw_rate_dps(self):
        """Direct gyro yaw-rate reading (deg/s), not integrated — see the module docstring for what
        that does and doesn't mean for drift. 0.0 if the link is down/stale (never a stale nonzero
        value held past staleness, same as vx/vy)."""
        with self._lock:
            return 0.0 if self._stale_locked() else self._yaw_rate

    @property
    def stale(self):
        with self._lock:
            return self._stale_locked()

    @property
    def dropped_count(self):
        with self._lock:
            return self._dropped

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self.ser is not None:
            self.ser.close()
