"""radar_filter.py — pre-fusion filtering of radar tracks: EKF time-sync to the camera's exact
timestamp, and rejection of suspicious/outlier tracks. Sits between iwr1642_live.Pipeline's
per-frame track output and fusion's Fusion.on_radar():

    out = pipe.process(fr)
    tracks = sync_and_filter(out["tracks"], dt)     # dt = camera_t - radar_frame_t
    fused, dets, matched = fus.on_radar(fr["t"], fr["frame"], tracks)

Smoothing is achieved by the EKF extrapolation itself (motion-model prediction from an
already-Kalman-smoothed state is inherently smoother than a raw nearest-frame value) — there is
deliberately no second, independently-tuned smoothing pass here, since that would just fight the
tracker's own tuned process noise.
"""
import copy

import numpy as np

import iwr1642_live as radar

# ---------------------------------------------------------------- SETTINGS
REJECT_MAX_MISSES = 3            # coasting longer than this (frames) isn't fed to fusion
REJECT_SIGMA_XY_M = 1.2          # EKF position uncertainty (post-sync) too large to trust
REJECT_SPEED_MPS = 15.0          # implausible speed for this scene — sanity cap, not a real limit
SYNC_MAX_EXTRAPOLATION_S = 0.25  # default |dt| beyond which the whole frame's tracks are dropped rather than extrapolated
                                 # blindly; fusion passes its own (wider) window once it has measured the detector latency


def sync_and_filter(tracks, dt, max_dt=SYNC_MAX_EXTRAPOLATION_S):
    """Extrapolate every track's EKF state by dt (the gap between the radar frame's own
    timestamp and the camera timestamp it's being matched against), then drop suspicious
    tracks. Returns shallow copies — never mutates the tracker's own Track objects, so the
    live Tracker keeps coasting/confirming them normally regardless of what fusion accepts."""
    if abs(dt) > max_dt:
        return []
    out = []
    for tr in tracks:
        if tr.misses > REJECT_MAX_MISSES:
            continue
        # a NaN/Inf state (e.g. a cluster built from a point with corrupted/missing side-info)
        # compares False against every threshold below, so it must be caught explicitly —
        # it won't trip sigma_xy_m/speed_mps on its own.
        if not (np.all(np.isfinite(tr.x)) and np.all(np.isfinite(tr.P))):
            continue
        synced = copy.copy(tr)
        synced.x, synced.P = radar.Track.step_state(tr.x, tr.P, dt)
        if not (np.all(np.isfinite(synced.x)) and np.all(np.isfinite(synced.P))):
            continue
        if synced.sigma_xy_m > REJECT_SIGMA_XY_M or synced.speed_mps > REJECT_SPEED_MPS:
            continue
        out.append(synced)
    return out
