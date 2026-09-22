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

Ego-motion extension (EGO_MOTION.md §5): the reader also keeps a 2-s ring buffer of samples on the
Pi clock and answers `sample_at(t)` — the IMU sample linearly interpolated to a radar frame time —
which is what iwr1642_live.Pipeline.process(frame, imu=...) consumes:

    imu = reader.sample_at(t_frame)          # {"yaw_rate", "pitch", "roll", "acc_fwd", "vy_imu", "zupt",
    out = pipe.process(frame, imu=imu)       #  "gyro_ok", "age_s"} — yaw_rate in rad/s, + = left, bias removed
    ...
    reader.begin_standstill()                # radar says STANDING for >= 3 s: average the raw gyro Z ...
    bias = reader.end_standstill()           # ... and adopt it as the new gyro bias (persist EGO_GYRO_BIAS_DPS)

--------------------------------------------------------------------------------------------------
WIRE PROTOCOL

Same NMEA-0183-style ASCII sentence approach as alerts.py, for the same reasons (plain text,
checksummed, no parser library needed, MCU-agnostic) — this is the mirror-image direction: sensor
data coming INTO the Pi from the ESP32, instead of alerts going out to an MCU.

    v1 (5 fields):   $EGOVEL,<vx_mps>,<vy_mps>,<yaw_rate_dps>,<seq>,<flags>*<checksum>\r\n
    v2 (10 fields):  $EGOVEL,<vx_mps>,<vy_mps>,<yaw_rate_dps>,<seq>,<flags>,<esp_ms>,<pitch_deg>,<roll_deg>,
                             <acc_fwd_mps2>,<gyro_cal>*<checksum>\r\n

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
  esp_ms       - (v2) ESP32 millis() at the IMU sample; mapped to the Pi clock with a running median
                 offset (EGO_MOTION.md §2), so the sample time survives USB/UART jitter
  pitch_deg    - (v2) from the Game Rotation Vector (no magnetometer: steel chassis); nose-up +
  roll_deg     - (v2) idem; right side down +
  acc_fwd_mps2 - (v2) bias-corrected forward linear acceleration (the ZUPT input)
  gyro_cal     - (v2) BNO08x gyroscope accuracy status 0..3 (3 = fully calibrated)
  checksum     - two uppercase hex digits, XOR of every byte between '$' and '*' (standard NMEA
                 checksum) — a corrupted line is dropped outright, never guessed at

Empty v2 fields mean "unknown" (numbers default to 0.0, gyro_cal to None); a v1 sentence is the same
as a v2 one with all extra fields unknown, so a v1 firmware keeps working. Any other field count is
a firmware/reader version mismatch and is rejected (counted + one warning), never misparsed.

Example:  $EGOVEL,0.05,1.35,-2.10,42,0*3A\r\n
          $EGOVEL,0.05,1.35,-2.10,42,0,123456,1.2,-0.4,0.02,3*14\r\n
"""
import json
import math
import statistics
import threading
import time
from collections import deque

import numpy as np

try:
    import serial
except ImportError:
    serial = None

try:                                           # same file the other modules read; a missing one only costs the defaults
    with open("configs.json", "r") as file:
        code_config = json.load(file)
except (OSError, ValueError):
    code_config = {}

EGO_TIME_OFFSET_S = float(code_config.get("EGO_TIME_OFFSET_S", 0.0))   # IMU time − radar frame timestamp for the same physical
                                                                        # instant — exactly what sample_at() adds to t (negative:
                                                                        # the radar's stamp is the later one, i.e. the radar lags);
                                                                        # calib_time_offset.py measures it — EGO_MOTION.md §3
EGO_GYRO_BIAS_DPS = float(code_config.get("EGO_GYRO_BIAS_DPS", 0.0))   # last standstill gyro-bias estimate, auto-refreshed — §3
EGO_BUFFER_S = 2.0            # ring buffer / clock-offset window length, s — EGO_MOTION.md §2, §5
EGO_LINK_LATENCY_S = 0.005    # fixed guess for ESP32 sample -> Pi receive latency, subtracted from the clock offset — §2
EGO_GYRO_OK_AGE_S = 0.2       # an interpolated sample farther than this from real data is not "gyro ok" — §5
EGO_GYRO_CAL_OK = 2           # BNO08x gyro accuracy status needed for gyro_ok (unknown on v1 passes) — §5
EGO_BIAS_MIN_SAMPLES = 25     # a standstill window shorter than this (~0.5 s at 50 Hz) does not refresh the bias
EGO_BIAS_MAX_SAMPLES = 3000   # ... and a longer one keeps only the last ~60 s at 50 Hz (bounded memory) — §5
EGO_IMU_NONE = {"yaw_rate": 0.0, "pitch": 0.0, "roll": 0.0, "acc_fwd": 0.0, "vy_imu": 0.0, "zupt": False,
                "gyro_ok": False, "age_s": float("inf"), "gyro_bias_dps": EGO_GYRO_BIAS_DPS,
                "time_offset_s": EGO_TIME_OFFSET_S}   # what sample_at() returns with no link / no data at all


def nmea_checksum_ok(text):
    """'$EGOVEL,...*3E' -> (True, body) if the XOR checksum matches, else (False, None)."""
    if not text.startswith("$") or "*" not in text:
        return False, None
    body, _, checksum_hex = text[1:].partition("*")
    try:
        checksum = int(checksum_hex.strip()[:2], 16)
    except ValueError:
        return False, None
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return cs == checksum, body


class EgoVelocityReader:
    """Background thread reading $EGOVEL sentences from the ESP32+IMU over a serial/GPIO UART link.
    Degrades gracefully — matching alerts.py's SerialAlertSink — rather than ever crashing the radar
    pipeline over an optional accessory: if the port can't be opened, or the link goes stale (no
    valid sentence for stale_s), every property reads back 0.0 (i.e. "assume stationary, not
    turning"), which is the same as the pipeline's old hardcoded EGO_SPEED_MPS = 0.0 default — so a
    missing/disconnected sensor silently falls back to exactly today's behavior, not a crash and not
    a stale/wrong value held forever. sample_at() likewise returns gyro_ok=False with zeros.

    port=None/"" builds a reader with no thread — feed_line(line, t_rx) then plays the wire for tests
    and for fusion_offline.py replay. `clock` is the time base t_rx is stamped with (time.time() by
    default; run_live() passes its own zero-based clock so radar frame times and IMU times agree)."""

    def __init__(self, port, baud=115200, stale_s=1.0, time_offset_s=None, gyro_bias_dps=None, clock=time.time):
        self.stale_s = stale_s
        self.time_offset_s = EGO_TIME_OFFSET_S if time_offset_s is None else float(time_offset_s)
        self.gyro_bias_dps = EGO_GYRO_BIAS_DPS if gyro_bias_dps is None else float(gyro_bias_dps)
        self.bias_refreshed = False            # a standstill window changed the bias -> worth persisting on exit
        self._clock = clock
        self._vx = 0.0
        self._vy = 0.0
        self._yaw_rate = 0.0
        self._last_t = None
        self._last_seq = None
        self._dropped = 0
        self._rejected = 0                     # sentences with a field count that is neither v1 nor v2
        self._version = None                   # 1 / 2 — what the firmware on the other end is sending
        self._samples = deque(maxlen=512)      # ~2 s at 50 Hz with headroom; pruned by time in _push_sample_locked
        self._offsets = deque(maxlen=512)      # (t_rx, t_rx - esp_ms/1000) over the last EGO_BUFFER_S
        self._offset = None                    # median of the above minus EGO_LINK_LATENCY_S; None until the first v2 line
        self._still = None                     # bounded window of raw gyro Z (deg/s) while a standstill window is open
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
            # exclusive: a second opener (record_sync.py running alongside the live pipeline) gets a clear
            # SerialException instead of silently splitting the $EGOVEL byte stream between both processes
            self.ser = serial.Serial(port, baud, timeout=0.2, exclusive=True)
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
        self.feed_line(text, self._clock())

    # ------------------------------------------------------------------ parsing
    def feed_line(self, text, t_rx):
        """Parse one '$EGOVEL,...*cs' line received at Pi time t_rx (the reader thread calls this; tests and
        the offline replay call it directly). Returns the parsed sample dict, or None if the line was dropped."""
        if isinstance(text, bytes):
            text = text.decode("ascii", errors="ignore")
        text = text.strip()
        ok, body = nmea_checksum_ok(text)
        if not ok:
            return None                                   # corrupted on the wire — drop, never guess
        fields = body.split(",")
        if fields[0] != "EGOVEL":
            return None
        n = len(fields) - 1
        if n not in (5, 10):
            with self._lock:
                self._rejected += 1
                bad = self._rejected
            if bad == 1 or bad % 500 == 0:                # loud once, then a reminder — a mismatched firmware never misparses
                print(f"⚠️ $EGOVEL with {n} fields (expected 5 = v1 or 10 = v2) — rejected ({bad} so far); "
                      "flash the matching firmware/ego_velocity/ego_velocity.ino")
            return None
        try:
            vx = float(fields[1])
            vy = float(fields[2])
            yaw_rate = float(fields[3])
            seq = int(fields[4])
            flags = int(fields[5]) if fields[5] else 0
            if n == 10:                                   # v2: empty field = unknown
                esp_ms = int(float(fields[6])) if fields[6] else None
                pitch = float(fields[7]) if fields[7] else 0.0
                roll = float(fields[8]) if fields[8] else 0.0
                acc_fwd = float(fields[9]) if fields[9] else 0.0
                gyro_cal = int(fields[10]) if fields[10] else None
            else:
                esp_ms, pitch, roll, acc_fwd, gyro_cal = None, 0.0, 0.0, 0.0, None
        except ValueError:
            return None
        sample = {"t_rx": float(t_rx), "esp_ms": esp_ms, "vx": vx, "vy": vy, "yaw_raw_dps": yaw_rate, "seq": seq,
                  "flags": flags, "zupt": bool(flags & 0x01), "pitch": pitch, "roll": roll, "acc_fwd": acc_fwd,
                  "gyro_cal": gyro_cal}
        with self._lock:
            self._version = 2 if n == 10 else 1
            if self._last_seq is not None:
                expected = (self._last_seq + 1) % 256
                if seq != expected:
                    self._dropped += (seq - expected) % 256
            self._last_seq = seq
            self._vx = vx
            self._vy = vy
            self._yaw_rate = yaw_rate
            self._last_t = float(t_rx)
            self._push_sample_locked(sample)
            if self._still is not None:
                self._still.append(yaw_rate)
        return sample

    def _push_sample_locked(self, s):
        """Ring buffer + clock offset bookkeeping. Caller must hold self._lock."""
        t_rx = s["t_rx"]
        if s["esp_ms"] is not None:
            # running median of (Pi receive time − ESP32 sample time) over the last EGO_BUFFER_S, minus the fixed link
            # latency guess: the median ignores the occasional late line (USB/UART scheduling) — EGO_MOTION.md §2.
            # millis() wraps after ~49.7 days; a wrap shows up as one wild offset the median also ignores until the
            # window has rolled over.
            self._offsets.append((t_rx, t_rx - s["esp_ms"] / 1000.0))
            while self._offsets and t_rx - self._offsets[0][0] > EGO_BUFFER_S:
                self._offsets.popleft()
            self._offset = statistics.median(o for _, o in self._offsets) - EGO_LINK_LATENCY_S
        self._samples.append(s)
        while self._samples and t_rx - self._samples[0]["t_rx"] > EGO_BUFFER_S:
            self._samples.popleft()

    def _sample_time_locked(self, s):
        """Pi-clock time of a sample: esp_ms mapped through the running offset (v2), else the receive time (v1)."""
        if s["esp_ms"] is not None and self._offset is not None:
            return s["esp_ms"] / 1000.0 + self._offset
        return s["t_rx"]

    # ------------------------------------------------------------------ interpolated lookup for the estimator
    def sample_at(self, t):
        """IMU state at Pi time t + time_offset_s (EGO_TIME_OFFSET_S: IMU time − radar frame timestamp), linearly
        interpolated between the two neighbouring samples of the ring buffer; held at the nearest sample outside
        the buffer. yaw_rate is rad/s, + = left, gyro bias removed. gyro_ok needs a calibrated gyro (status >= 2,
        or unknown on a v1 firmware) AND real data within EGO_GYRO_OK_AGE_S of the requested time — EGO_MOTION.md §5.
        Two extra keys, gyro_bias_dps and time_offset_s, carry the constants in effect for the estimator's info()."""
        t_q = float(t) + self.time_offset_s
        with self._lock:
            if not self._samples:
                return {**EGO_IMU_NONE, "gyro_bias_dps": self.gyro_bias_dps, "time_offset_s": self.time_offset_s}
            ss = list(self._samples)
            ts = np.array([self._sample_time_locked(s) for s in ss])
            bias = self.gyro_bias_dps
        order = np.argsort(ts, kind="stable")             # v1/v2 mix or an offset step could leave the buffer unsorted
        ts = ts[order]; ss = [ss[i] for i in order]
        k = int(np.searchsorted(ts, t_q))                 # ss[k-1].t <= t_q < ss[k].t
        if k <= 0:
            a, b, w, age = ss[0], ss[0], 0.0, ts[0] - t_q
        elif k >= len(ss):
            a, b, w, age = ss[-1], ss[-1], 0.0, t_q - ts[-1]
        else:
            a, b = ss[k - 1], ss[k]
            span = ts[k] - ts[k - 1]
            w = (t_q - ts[k - 1]) / span if span > 1e-6 else 0.0
            age = min(t_q - ts[k - 1], ts[k] - t_q)   # distance to the NEAREST real sample, not 0: a query bracketed by
                                                     # two samples 1.5 s apart is an interpolation across a dropout, and
                                                     # §5's gyro_ok / EGO_IMU_MAX_AGE_S tests must see that gap
        lerp = lambda key: float((1.0 - w) * a[key] + w * b[key])   # plain floats: these go into logs / JSONL
        near = b if w >= 0.5 else a                       # flags / statuses: nearest sample, no interpolation
        cal = near["gyro_cal"]
        gyro_ok = (cal is None or cal >= EGO_GYRO_CAL_OK) and age <= EGO_GYRO_OK_AGE_S
        return {"yaw_rate": math.radians(lerp("yaw_raw_dps") - bias), "pitch": lerp("pitch"), "roll": lerp("roll"),
                "acc_fwd": lerp("acc_fwd"), "vy_imu": lerp("vy"), "zupt": bool(near["zupt"]),
                "gyro_ok": bool(gyro_ok), "age_s": float(max(age, 0.0)),
                "gyro_bias_dps": bias, "time_offset_s": self.time_offset_s}   # extras: the estimator's info() echoes them

    # ------------------------------------------------------------------ standstill gyro-bias refresh
    def begin_standstill(self):
        """The radar has reported STANDING long enough (EGO_MOTION.md §5: >= 3 s): start averaging the raw gyro Z.
        Idempotent — a second call while a window is open keeps the samples collected so far."""
        with self._lock:
            if self._still is None:
                # bounded: 60 s at 50 Hz — an hour parked must not grow a list, and the bias should reflect the gyro's
                # recent thermal state rather than the oldest data in the window (EGO_MOTION.md §5)
                self._still = deque(maxlen=EGO_BIAS_MAX_SAMPLES)

    @property
    def standstill_open(self):
        with self._lock:
            return self._still is not None

    def end_standstill(self):
        """Close the window opened by begin_standstill(): with enough samples the mean raw gyro Z becomes the new
        bias (removed from every later sample_at() yaw_rate). Returns the bias in effect afterwards (deg/s)."""
        with self._lock:
            samples, self._still = self._still, None
            if samples is not None and len(samples) >= EGO_BIAS_MIN_SAMPLES:
                new = float(np.mean(samples))
                if abs(new - self.gyro_bias_dps) > 1e-6:
                    self.bias_refreshed = True
                self.gyro_bias_dps = new
            return self.gyro_bias_dps

    # ------------------------------------------------------------------ v1 properties (unchanged behaviour)
    def _stale_locked(self):
        """Caller must hold self._lock."""
        return self._last_t is None or (self._clock() - self._last_t) > self.stale_s

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
        """Direct gyro yaw-rate reading (deg/s), not integrated and NOT bias-removed (sample_at() is the
        bias-removed path) — see the module docstring for what that does and doesn't mean for drift. 0.0 if
        the link is down/stale (never a stale nonzero value held past staleness, same as vx/vy)."""
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

    @property
    def rejected_count(self):
        with self._lock:
            return self._rejected

    @property
    def version(self):
        """1 / 2 — sentence format of the firmware on the other end, None before the first valid line."""
        with self._lock:
            return self._version

    @property
    def clock_offset_s(self):
        """Pi time − ESP32 time (s) as currently estimated; None until a v2 sentence has arrived."""
        with self._lock:
            return self._offset

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self.ser is not None:
            self.ser.close()
