"""tests/test_calib.py — test G of EGO_MOTION.md §8 plus the log/replay plumbing of §6.

G.  a synthetic step-turn log (stationary world, v = 1 m/s, lever arm 2 m, radar timestamps 120 ms late so
    EGO_TIME_OFFSET_S = -0.120) -> tools/calib_time_offset.py recovers the offset within 20 ms, and its
    cross-correlation check agrees within one frame (PASS).
M.  two synthetic straight drives with the radar mounted 7° left of the vehicle axis -> tools/calib_mount.py
    recovers MOUNT_YAW_DEG within 0.5°.
S.  fusion_offline.py --log replay smoke test on a 20-frame synthetic log (real Pipeline.process path, IMU fed
    through EgoVelocityReader.feed_line/sample_at when present, else the local parser).
P.  $EGOVEL v1/v2 parsing + interpolation of the local fallback parser.

No hardware: the radar packets are built here in the SDK 3 OOB format (detected points + side info TLVs) exactly
as iwr1642_live.parse_frame expects them, the $EGOVEL lines with the NMEA checksum.

    python -m pytest -q tests/test_calib.py
"""
import base64
import gzip
import json
import math
import os
import struct
import sys

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
if not os.path.exists("configs.json"):
    os.chdir(_root)                                      # iwr1642_live reads configs.json from the working directory
import iwr1642_live as radar                             # noqa: E402
import fusion_offline as fo                              # noqa: E402
import calib_time_offset                                 # noqa: E402
import calib_mount                                       # noqa: E402

CFG_PATH = os.path.join(_root, "radar_configs", "hangar_v9.cfg")


@pytest.fixture(scope="module")
def cfg():
    c = radar.parse_cfg(CFG_PATH)
    if not c.get("doppler_res_mps"):
        pytest.skip("radar_configs/hangar_v9.cfg missing — cannot build a profile")
    return c


# ---------------------------------------------------------------- synthetic world
def sentence(body):
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"${body}*{cs:02X}"


def egovel_v2(vy, yaw_dps, seq, t_s, zupt=False, cal=3):
    return sentence(f"EGOVEL,0.00,{vy:.2f},{yaw_dps:.3f},{seq % 256},{int(zupt)},{int(round(t_s * 1000))},0.00,0.00,0.000,{cal}")


def sdk3_packet(frame_num, points, snr_db=20.0):
    """SDK 3.x OOB demo packet: header (MAGIC + 8 uint32) + detected-points TLV (x, y, z, v float32) + side-info TLV."""
    pts = b"".join(struct.pack("<4f", x, y, z, v) for x, y, z, v in points)
    side = b"".join(struct.pack("<2h", int(snr_db * 10), 300) for _ in points)
    tlvs = struct.pack("<2I", radar.TLV_DETECTED_POINTS, len(pts)) + pts + struct.pack("<2I", radar.TLV_SIDE_INFO, len(side)) + side
    total = radar.HEADER_LEN + len(tlvs)
    header = radar.MAGIC + struct.pack("<8I", 3 << 24, total, 0xA1642, frame_num, 0, len(points), 2, 0)
    return header + tlvs


def step_turn_dps(tau, peak=25.0):
    """0 -> +peak -> 0 -> -peak -> 0 with 0.4 s ramps; ~8 s total. Sharp enough to time, slow enough for a tractor."""
    segs = [(1.0, 0.0, 0.0), (0.4, 0.0, peak), (1.5, peak, peak), (0.4, peak, 0.0), (1.0, 0.0, 0.0),
            (0.4, 0.0, -peak), (1.5, -peak, -peak), (0.4, -peak, 0.0), (1.4, 0.0, 0.0)]
    t0 = 0.0
    for dur, a, b in segs:
        if tau < t0 + dur:
            return a + (b - a) * (tau - t0) / dur
        t0 += dur
    return 0.0


def synth_log(path, cfg, n_frames, v_mps=1.0, yaw_fn=lambda tau: 0.0, lx=2.0, ly=0.0, mount_yaw_deg=0.0,
              radar_lag_s=0.0, gyro_noise_dps=0.15, n_points=60, seed=1, imu_hz=50.0):
    """Stationary world seen from a moving, turning sensor. Doppler follows EGO_MOTION.md §4:
    dop = -((v - w·ly)·cos θ_v - w·lx·sin θ_v) with θ_v = θ_sensor - mount_yaw, quantised to the profile's bin and
    wrapped by its period. Radar frames are stamped tau + radar_lag_s (the radar's timestamp is later than the IMU's
    for the same instant -> EGO_TIME_OFFSET_S = -radar_lag_s); $EGOVEL v2 lines at imu_hz stamped at tau."""
    rng = np.random.default_rng(seed)
    binw, period, fp = cfg["doppler_res_mps"], cfg["doppler_period_mps"], cfg["frame_period_s"]
    recs = [{"kind": "meta", "t": 0.0, "tag": "synthetic", "cfg_file": CFG_PATH}]
    m = math.radians(mount_yaw_deg)
    for k in range(n_frames):
        tau = k * fp
        w = math.radians(yaw_fn(tau))
        rng_m = rng.uniform(2.0, 15.0, n_points)
        th_s = rng.uniform(-math.radians(60), math.radians(60), n_points)      # sensor azimuth, right +
        th_v = th_s - m
        dop = -((v_mps - w * ly) * np.cos(th_v) - w * lx * np.sin(th_v))
        dop = np.round(dop / binw) * binw
        dop -= np.round(dop / period) * period
        pts = [(float(r * math.sin(a)), float(r * math.cos(a)), 0.0, float(d)) for r, a, d in zip(rng_m, th_s, dop)]
        recs.append({"kind": "radar", "t": round(tau + radar_lag_s, 4), "frame": k + 1,
                     "b64": base64.b64encode(sdk3_packet(k + 1, pts)).decode("ascii")})
    n_imu = int(n_frames * fp * imu_hz) + int(imu_hz)
    for i in range(n_imu):
        tau = i / imu_hz
        yaw = yaw_fn(tau) + rng.normal(0.0, gyro_noise_dps)
        recs.append({"kind": "ego", "t": round(tau, 4), "line": egovel_v2(v_mps, yaw, i, tau)})
    recs.sort(key=lambda r: r["t"])
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return path


# ---------------------------------------------------------------- P. parser / interpolation
def test_parse_egovel_v1_v2_and_checksum():
    v1 = sentence("EGOVEL,0.05,1.35,-2.10,42,0")
    d = fo.parse_egovel(v1)
    assert d and d["vy"] == 1.35 and d["yaw_rate_dps"] == -2.10 and d["seq"] == 42 and d["gyro_cal"] is None
    d2 = fo.parse_egovel(egovel_v2(1.0, 12.5, 7, 3.2, zupt=True, cal=2))
    assert d2 and d2["zupt"] and d2["gyro_cal"] == 2 and d2["esp_ms"] == 3200
    assert fo.parse_egovel(v1[:-1] + "0") is None                          # corrupted checksum
    assert fo.parse_egovel("$EGOVEL,1,2,3*00") is None                     # wrong field count
    assert fo.parse_egovel("$RDALT,1,2*00") is None


def test_logged_imu_interpolation_offset_bias():
    imu = fo.LoggedImu(time_offset_s=-0.1, gyro_bias_dps=1.0)
    for i in range(6):
        imu.feed_line(egovel_v2(0.0, 10.0 * i, i, 0.1 * i), 0.1 * i)       # yaw 0,10,...,50 dps at t = 0..0.5
    s = imu.sample_at(0.35)                                                 # -> queries t = 0.25 -> 25 dps - 1 bias
    assert abs(math.degrees(s["yaw_rate"]) - 24.0) < 1e-6 and s["gyro_ok"] and s["age_s"] < 0.1
    far = imu.sample_at(5.0)
    assert not far["gyro_ok"] and far["age_s"] > 4


# ---------------------------------------------------------------- G. time offset
def test_g_time_offset_recovered(cfg, tmp_path):
    path = synth_log(str(tmp_path / "stepturn.jsonl.gz"), cfg, n_frames=125, v_mps=1.0, yaw_fn=step_turn_dps,
                     lx=2.0, ly=0.0, radar_lag_s=0.120)
    lines = []
    r = calib_time_offset.estimate(path, cfg, lx=2.0, ly=0.0, out=lines.append)
    assert r["offset_grid"] is not None, "\n".join(lines)
    assert abs(r["offset_grid"] - (-0.120)) < 0.020, f"grid search {r['offset_grid']:+.3f} s\n" + "\n".join(lines)
    assert r["offset_xcorr"] is not None and abs(r["offset_xcorr"] - (-0.120)) < cfg["frame_period_s"], "\n".join(lines)
    assert r["passed"], "\n".join(lines)


def test_g_zero_offset_is_zero(cfg, tmp_path):
    path = synth_log(str(tmp_path / "stepturn0.jsonl.gz"), cfg, n_frames=125, yaw_fn=step_turn_dps, lx=2.0, radar_lag_s=0.0, seed=3)
    r = calib_time_offset.estimate(path, cfg, lx=2.0, ly=0.0, out=lambda s: None)
    assert abs(r["offset_grid"]) < 0.020 and r["passed"]


# ---------------------------------------------------------------- M. mount yaw
def test_m_mount_yaw_recovered(cfg, tmp_path):
    fwd = synth_log(str(tmp_path / "fwd.jsonl.gz"), cfg, n_frames=60, v_mps=1.5, mount_yaw_deg=7.0, seed=5)
    back = synth_log(str(tmp_path / "back.jsonl.gz"), cfg, n_frames=60, v_mps=-1.0, mount_yaw_deg=7.0, seed=6)
    r = calib_mount.estimate(fwd, back, cfg, out=lambda s: None)
    assert abs(r["mount_yaw_deg"] - 7.0) < 0.5, r
    assert all(abs(p - 7.0) < 1.0 for p in r["per_log"]), r


# ---------------------------------------------------------------- S. replay smoke
def test_s_replay_smoke(cfg, tmp_path):
    path = synth_log(str(tmp_path / "short.jsonl.gz"), cfg, n_frames=20, v_mps=1.0, yaw_fn=lambda tau: 8.0, lx=2.0)
    lines = []
    s = fo.replay_log(path, cfg=cfg, time_offset_s=-0.05, gyro_bias_dps=0.2, segments=[(0.0, 0.7, "a"), (0.7, 2.0, "b")],
                      static_mask=(-5.0, 2.0, 5.0, 15.0), print_every=1, out=lines.append)
    assert s["n_frames"] == 20 and s["n_ego_lines"] > 50 and s["n_bad_ego_lines"] == 0
    assert sum(s["state_hist"].values()) == 20 and set(s["segments"]) == {"a", "b"}
    assert s["static_mask"]["n_points"] > 0 and 0.0 <= s["static_mask"]["rate"] <= 1.0
    assert s["imu_engine"] in ("reader", "local")
    assert sum(1 for l in lines if l.startswith("t=")) == 20                # one compact ego_info line per frame
    # the estimator sees the injected motion once it has settled (median window + rate limit)
    v_last = float(lines[-1].split("v=")[1].split()[0]) if "v=" in lines[-1] else s["segments"]["b"]["median"]
    assert abs(s["segments"]["b"]["median"] - 1.0) < 0.5, s["segments"]


def test_s_replay_local_engine_matches_keys(cfg, tmp_path):
    path = synth_log(str(tmp_path / "short2.jsonl.gz"), cfg, n_frames=5, v_mps=0.0)
    s = fo.replay_log(path, cfg=cfg, imu_engine="local", print_every=0, out=lambda s: None)
    assert s["imu_engine"] == "local" and s["n_frames"] == 5
