"""tests/test_egolink.py — test F of EGO_MOTION.md §8: the $EGOVEL link (ego_velocity.EgoVelocityReader).

v1 and v2 parsing + checksum, seq-drop detection, interpolation, time offset, gyro bias removal, staleness ->
gyro_ok False, the standstill bias refresh. No hardware: the reader is built with port=None (no thread) and fed
with feed_line(line, t_rx).

    python -m pytest -q tests/test_egolink.py
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ego_velocity as ev                                              # noqa: E402


def sentence(body):
    """'EGOVEL,...' -> '$EGOVEL,...*CS\\r\\n' with the NMEA XOR checksum."""
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"${body}*{cs:02X}\r\n"


def v1(vx=0.0, vy=0.0, yaw=0.0, seq=0, flags=0):
    return sentence(f"EGOVEL,{vx:.2f},{vy:.2f},{yaw:.2f},{seq},{flags}")


def v2(vx=0.0, vy=0.0, yaw=0.0, seq=0, flags=0, esp_ms=0, pitch=0.0, roll=0.0, acc=0.0, cal=3):
    f = lambda x, fmt="%.2f": "" if x is None else (fmt % x)
    return sentence(f"EGOVEL,{vx:.2f},{vy:.2f},{yaw:.2f},{seq},{flags},{f(esp_ms, '%d')},{f(pitch)},{f(roll)},"
                    f"{f(acc, '%.3f')},{f(cal, '%d')}")


def reader(**kw):
    kw.setdefault("time_offset_s", 0.0)
    kw.setdefault("gyro_bias_dps", 0.0)
    kw.setdefault("clock", lambda: 0.0)
    return ev.EgoVelocityReader(None, **kw)


# ---------------------------------------------------------------- parsing + checksum
def test_v1_sentence_parses_and_properties_unchanged():
    r = reader(clock=lambda: 10.0)
    s = r.feed_line(v1(0.05, 1.35, -2.10, seq=42, flags=1), t_rx=10.0)
    assert s is not None and s["seq"] == 42 and s["zupt"] is True
    assert r.version == 1
    assert r.vx_mps == pytest.approx(0.05) and r.vy_mps == pytest.approx(1.35)
    assert r.yaw_rate_dps == pytest.approx(-2.10)                      # raw, not bias-removed (unchanged v1 behaviour)
    assert r.speed_mps == pytest.approx(math.hypot(0.05, 1.35))
    assert r.clock_offset_s is None                                    # no esp_ms on v1 -> no clock mapping
    imu = r.sample_at(10.0)
    assert imu["gyro_ok"] is True                                      # gyro_cal unknown on v1 passes
    assert imu["yaw_rate"] == pytest.approx(math.radians(-2.10))
    assert imu["pitch"] == 0.0 and imu["roll"] == 0.0 and imu["acc_fwd"] == 0.0   # v2 fields default to 0 on v1
    assert imu["vy_imu"] == pytest.approx(1.35) and imu["zupt"] is True


def test_v2_sentence_parses_all_fields():
    r = reader()
    s = r.feed_line(v2(0.1, 2.0, 5.0, seq=7, flags=0, esp_ms=123456, pitch=1.2, roll=-0.4, acc=0.02, cal=3), t_rx=5.0)
    assert s["esp_ms"] == 123456 and s["pitch"] == pytest.approx(1.2) and s["roll"] == pytest.approx(-0.4)
    assert s["acc_fwd"] == pytest.approx(0.02) and s["gyro_cal"] == 3 and s["zupt"] is False
    assert r.version == 2
    # clock mapping: median(t_rx - esp_ms/1000) - 5 ms link latency (EGO_MOTION.md §2)
    assert r.clock_offset_s == pytest.approx(5.0 - 123.456 - 0.005)
    imu = r.sample_at(5.0)
    assert imu["pitch"] == pytest.approx(1.2) and imu["roll"] == pytest.approx(-0.4)
    assert imu["acc_fwd"] == pytest.approx(0.02) and imu["gyro_ok"] is True


def test_v2_empty_fields_mean_unknown():
    r = reader()
    s = r.feed_line(v2(seq=1, esp_ms=None, pitch=None, roll=None, acc=None, cal=None), t_rx=1.0)
    assert s is not None
    assert s["esp_ms"] is None and s["gyro_cal"] is None
    assert s["pitch"] == 0.0 and s["roll"] == 0.0 and s["acc_fwd"] == 0.0
    assert r.clock_offset_s is None
    assert r.sample_at(1.0)["gyro_ok"] is True                         # unknown calibration status passes, like v1


def test_firmware_v2_line_verbatim():
    """The exact bytes firmware/ego_velocity/ego_velocity.ino's send_egovel() puts on the wire, including the
    "attitude not known yet" form where pitch and roll are two empty fields."""
    r = reader()
    assert r.feed_line("$EGOVEL,0.05,1.35,-2.10,42,1,123456,,,0.020,3*0F\r\n", 1.0)["pitch"] == 0.0
    s = r.feed_line("$EGOVEL,0.05,1.35,-2.10,43,0,123476,1.2,-0.4,0.020,2*26\r\n", 1.02)
    assert s["pitch"] == pytest.approx(1.2) and s["roll"] == pytest.approx(-0.4) and s["gyro_cal"] == 2
    assert r.version == 2 and r.dropped_count == 0 and r.rejected_count == 0


def test_bad_checksum_and_garbage_dropped():
    r = reader()
    good = v1(0.5, 1.0, 2.0, seq=3)
    bad = good.replace("*", "*0")[:-3] + "\r\n"                        # corrupt the checksum
    assert r.feed_line(bad, 0.0) is None
    assert r.feed_line("garbage\r\n", 0.0) is None
    assert r.feed_line("$EGOVEL,x,y,z,1,0*00\r\n", 0.0) is None
    assert r.feed_line(sentence("RDALT,111,S,C,7,1.20,-8.40,-2.10,0"), 0.0) is None   # other talker
    assert r.stale and r.vy_mps == 0.0 and r.sample_at(0.0)["gyro_ok"] is False
    assert r.feed_line(good, 0.0) is not None


def test_other_field_counts_rejected_loudly(capsys):
    r = reader()
    assert r.feed_line(sentence("EGOVEL,0.00,0.00,0.00,1"), 0.0) is None            # old 4-field format
    assert r.feed_line(sentence("EGOVEL,0.00,0.00,0.00,1,0,1000"), 0.0) is None     # neither v1 nor v2
    assert r.rejected_count == 2 and r.version is None
    assert "rejected" in capsys.readouterr().out
    assert r.feed_line(v2(seq=1), 0.0) is not None


def test_bytes_accepted_like_the_serial_thread():
    r = reader()
    assert r.feed_line(v1(seq=9).encode("ascii"), 0.0)["seq"] == 9


# ---------------------------------------------------------------- seq drops
def test_seq_drop_detection_incl_wraparound():
    r = reader()
    for seq in (250, 251, 252):
        r.feed_line(v1(seq=seq), 0.0)
    assert r.dropped_count == 0
    r.feed_line(v1(seq=255), 0.0)                                      # 253, 254 missing
    assert r.dropped_count == 2
    r.feed_line(v2(seq=0), 0.0)                                        # 255 -> 0 wrap, no drop, format switch is fine
    assert r.dropped_count == 2
    r.feed_line(v2(seq=3), 0.0)                                        # 1, 2 missing
    assert r.dropped_count == 4


# ---------------------------------------------------------------- interpolation
def test_sample_at_interpolates_between_samples():
    r = reader()
    r.feed_line(v2(vy=1.0, yaw=10.0, seq=1, esp_ms=1000, pitch=2.0, acc=0.5, cal=3), t_rx=1.0)
    r.feed_line(v2(vy=2.0, yaw=20.0, seq=2, esp_ms=1100, pitch=4.0, acc=1.5, cal=3), t_rx=1.1)
    imu = r.sample_at(1.05 - 0.005)                                    # Pi time of ESP 1050 ms (5 ms link latency)
    assert imu["yaw_rate"] == pytest.approx(math.radians(15.0))
    assert imu["vy_imu"] == pytest.approx(1.5) and imu["pitch"] == pytest.approx(3.0)
    # age is the distance to the NEAREST real sample (half of a 100 ms spacing here), not 0: an interpolated
    # value is only as fresh as the data bracketing it — EGO_MOTION.md §5
    assert imu["acc_fwd"] == pytest.approx(1.0) and imu["age_s"] == pytest.approx(0.05) and imu["gyro_ok"] is True
    imu = r.sample_at(1.075 - 0.005)                                   # 3/4 of the way, flags from the nearest sample
    assert imu["yaw_rate"] == pytest.approx(math.radians(17.5))


def test_sample_at_across_a_dropout_is_not_gyro_ok():
    """A query bracketed by two samples 1.5 s apart is an interpolation across a link dropout, not fresh data:
    age must be the gap to the nearest real sample and gyro_ok False (EGO_MOTION.md §5, 0.2 s limit)."""
    r = reader()
    r.feed_line(v2(yaw=20.0, seq=1, esp_ms=1000, cal=3), t_rx=1.0)
    r.feed_line(v2(yaw=-20.0, seq=2, esp_ms=2500, cal=3), t_rx=2.5)
    imu = r.sample_at(1.75 - 0.005)                                    # halfway across the gap
    assert imu["age_s"] == pytest.approx(0.75, abs=1e-3)
    assert imu["gyro_ok"] is False
    imu = r.sample_at(1.1 - 0.005)                                     # 100 ms after the last real sample
    assert imu["age_s"] == pytest.approx(0.1, abs=1e-3) and imu["gyro_ok"] is True


def test_sample_at_uses_esp_clock_not_receive_jitter():
    # every 20 ms on the ESP32, but one line arrives 40 ms late: the median offset keeps the sample where it belongs
    r = reader()
    for k in range(50):
        late = 0.040 if k == 25 else 0.0
        r.feed_line(v2(yaw=float(k), seq=k, esp_ms=1000 + 20 * k, cal=3), t_rx=1.0 + 0.020 * k + late)
    assert r.clock_offset_s == pytest.approx(1.0 - 1.0 - 0.005)
    imu = r.sample_at(1.0 + 0.020 * 25 - 0.005)                       # Pi time of ESP sample 25 (incl. the 5 ms latency)
    assert imu["yaw_rate"] == pytest.approx(math.radians(25.0), abs=1e-6)


def test_ring_buffer_keeps_two_seconds():
    r = reader()
    for k in range(200):                                               # 4 s at 50 Hz
        r.feed_line(v2(yaw=float(k), seq=k % 256, esp_ms=20 * k, cal=3), t_rx=0.020 * k)
    assert len(r._samples) <= 101 + 1
    assert r._samples[0]["esp_ms"] >= 20 * 99                         # nothing older than 2 s survives
    assert r.sample_at(0.020 * 150 - 0.005)["yaw_rate"] == pytest.approx(math.radians(150.0), abs=1e-6)


# ---------------------------------------------------------------- time offset + bias removal
def test_time_offset_shifts_the_lookup():
    r = reader(time_offset_s=-0.100)                                   # radar lags the IMU by 100 ms
    for k in range(20):
        r.feed_line(v2(yaw=float(k), seq=k, esp_ms=1000 + 20 * k, cal=3), t_rx=1.0 + 0.020 * k)
    t_frame = 1.0 + 0.020 * 10 - 0.005                                 # Pi time of ESP sample 10
    assert r.sample_at(t_frame)["yaw_rate"] == pytest.approx(math.radians(5.0), abs=1e-6)   # sample 10 - 100 ms = sample 5
    r.time_offset_s = 0.0
    assert r.sample_at(t_frame)["yaw_rate"] == pytest.approx(math.radians(10.0), abs=1e-6)


def test_gyro_bias_removed_from_sample_at_only():
    r = reader(gyro_bias_dps=0.8)
    r.feed_line(v1(yaw=3.0, seq=1), t_rx=0.0)
    assert r.yaw_rate_dps == pytest.approx(3.0)                        # v1 property: raw
    assert r.sample_at(0.0)["yaw_rate"] == pytest.approx(math.radians(2.2))   # bias removed, + = left, rad/s


def test_bias_default_comes_from_configs_json(monkeypatch):
    monkeypatch.setattr(ev, "EGO_GYRO_BIAS_DPS", -0.5)
    monkeypatch.setattr(ev, "EGO_TIME_OFFSET_S", -0.05)
    r = ev.EgoVelocityReader("")
    assert r.gyro_bias_dps == -0.5 and r.time_offset_s == -0.05


# ---------------------------------------------------------------- staleness / gyro_ok
def test_stale_data_gives_gyro_ok_false():
    r = reader()
    for k in range(10):
        r.feed_line(v2(yaw=1.0, seq=k, esp_ms=20 * k, cal=3), t_rx=0.020 * k)
    t_last = 0.020 * 9 - 0.005
    assert r.sample_at(t_last + 0.15)["gyro_ok"] is True               # within EGO_GYRO_OK_AGE_S: held, still ok
    imu = r.sample_at(t_last + 0.5)
    assert imu["gyro_ok"] is False and imu["age_s"] == pytest.approx(0.5)
    assert imu["yaw_rate"] == pytest.approx(math.radians(1.0))         # value still the last one, caller decides
    assert r.sample_at(t_last - 5.0)["gyro_ok"] is False               # asking far before the buffer is stale too


def test_uncalibrated_gyro_gives_gyro_ok_false():
    r = reader()
    r.feed_line(v2(seq=1, esp_ms=1000, cal=1), t_rx=1.0)
    assert r.sample_at(1.0)["gyro_ok"] is False
    r.feed_line(v2(seq=2, esp_ms=1020, cal=2), t_rx=1.02)
    assert r.sample_at(1.02)["gyro_ok"] is True


def test_no_data_at_all():
    r = reader()
    imu = r.sample_at(123.0)
    assert imu["gyro_ok"] is False and imu["yaw_rate"] == 0.0 and math.isinf(imu["age_s"])
    assert r.vx_mps == r.vy_mps == r.yaw_rate_dps == r.speed_mps == 0.0


def test_v1_properties_go_to_zero_when_stale():
    now = [0.0]
    r = reader(clock=lambda: now[0], stale_s=1.0)
    r.feed_line(v1(0.3, 1.5, 4.0, seq=1), t_rx=0.0)
    assert r.vy_mps == pytest.approx(1.5) and not r.stale
    now[0] = 1.5
    assert r.stale and r.vx_mps == 0.0 and r.vy_mps == 0.0 and r.yaw_rate_dps == 0.0 and r.speed_mps == 0.0


# ---------------------------------------------------------------- standstill bias refresh
def test_standstill_refresh_averages_raw_gyro_z():
    r = reader(gyro_bias_dps=0.0)
    assert not r.standstill_open
    r.begin_standstill()
    assert r.standstill_open
    for k in range(50):
        r.feed_line(v2(yaw=0.3 + (0.1 if k % 2 else -0.1), seq=k, esp_ms=20 * k, cal=3, flags=1), t_rx=0.020 * k)
    bias = r.end_standstill()
    assert bias == pytest.approx(0.3) and r.gyro_bias_dps == pytest.approx(0.3) and r.bias_refreshed
    assert not r.standstill_open
    r.feed_line(v2(yaw=0.3, seq=50, esp_ms=1000, cal=3), t_rx=1.0)
    assert r.sample_at(1.0)["yaw_rate"] == pytest.approx(0.0, abs=1e-9)   # the bias is gone from the estimator's view


def test_standstill_refresh_needs_enough_samples():
    r = reader(gyro_bias_dps=0.2)
    r.begin_standstill()
    for k in range(5):                                                 # far below EGO_BIAS_MIN_SAMPLES
        r.feed_line(v1(yaw=5.0, seq=k), t_rx=0.02 * k)
    assert r.end_standstill() == pytest.approx(0.2) and not r.bias_refreshed
    assert r.end_standstill() == pytest.approx(0.2)                    # closing a closed window is harmless


def test_samples_outside_the_window_do_not_count():
    r = reader(gyro_bias_dps=0.0)
    for k in range(40):
        r.feed_line(v1(yaw=9.0, seq=k), t_rx=0.02 * k)                 # driving: not part of the average
    r.begin_standstill()
    for k in range(40, 80):
        r.feed_line(v1(yaw=0.5, seq=k), t_rx=0.02 * k)
    assert r.end_standstill() == pytest.approx(0.5)
