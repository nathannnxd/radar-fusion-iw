"""ego_velocity.py — reads the platform's own forward speed from an external IMU (ESP32 + BNO08x,
see firmware/ego_velocity/ego_velocity.ino) over a UART/GPIO link, and exposes it as EGO_SPEED_MPS
for the radar pipeline's existing Doppler ego-compensation (iwr1642_live.points_to_detections) and
for Track.ground_velocity()/ground_speed_mps() (see EGO_VELOCITY.md for the full design/rationale).

    reader = EgoVelocityReader(EGO_SERIAL_PORT, EGO_BAUD)
    ...
    pipe.ego = reader.speed_mps      # once per radar frame, before pipe.process(frame)
    ...
    reader.close()

--------------------------------------------------------------------------------------------------
WIRE PROTOCOL

Same NMEA-0183-style ASCII sentence approach as alerts.py, for the same reasons (plain text,
checksummed, no parser library needed, MCU-agnostic) — this is the mirror-image direction: sensor
data coming INTO the Pi from the ESP32, instead of alerts going out to an MCU.

    $EGOVEL,<speed_mps>,<seq>,<flags>*<checksum>\r\n

  speed_mps - the platform's current forward speed estimate, signed (+ forward, - reverse), from
              integrating the IMU's linear acceleration with drift correction — see the firmware's
              own header comment for exactly how, and its real limitations (this is dead-reckoning,
              not a substitute for a wheel encoder or GPS over any long horizon)
  seq       - 0-255, wraps around; lets this reader notice dropped/corrupted lines without needing
              synchronized clocks on both ends
  flags     - bit 0: ZUPT (zero-velocity update) is currently active, i.e. the firmware believes the
              platform is stationary and is holding speed_mps at 0 rather than integrating drift.
              Other bits reserved.
  checksum  - two uppercase hex digits, XOR of every byte between '$' and '*' (standard NMEA
              checksum) — a corrupted line is dropped outright, never guessed at

Example:  $EGOVEL,1.35,42,0*3E\r\n
"""
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
    valid sentence for stale_s), speed_mps reads back 0.0 (i.e. "assume stationary"), which is the
    same as the pipeline's old hardcoded EGO_SPEED_MPS = 0.0 default — so a missing/disconnected
    sensor silently falls back to exactly today's behavior, not a crash and not a stale/wrong value
    held forever."""

    def __init__(self, port, baud=115200, stale_s=1.0):
        self.stale_s = stale_s
        self._speed = 0.0
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
        if len(fields) < 3 or fields[0] != "EGOVEL":
            return
        try:
            speed = float(fields[1])
            seq = int(fields[2])
        except ValueError:
            return
        with self._lock:
            if self._last_seq is not None:
                expected = (self._last_seq + 1) % 256
                if seq != expected:
                    self._dropped += (seq - expected) % 256
            self._last_seq = seq
            self._speed = speed
            self._last_t = time.time()

    @property
    def speed_mps(self):
        """Current forward ego-speed. Falls back to 0.0 (assume stationary) if the link has never
        produced a valid sentence, or hasn't in the last stale_s seconds."""
        with self._lock:
            if self._last_t is None or (time.time() - self._last_t) > self.stale_s:
                return 0.0
            return self._speed

    @property
    def stale(self):
        with self._lock:
            return self._last_t is None or (time.time() - self._last_t) > self.stale_s

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
