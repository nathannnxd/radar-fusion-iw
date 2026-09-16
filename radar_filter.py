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

import iwr1642_live as radar

# ---------------------------------------------------------------- SETTINGS
REJECT_MAX_MISSES = 3            # coasting longer than this (frames) isn't fed to fusion
REJECT_SIGMA_XY_M = 1.2          # EKF position uncertainty (post-sync) too large to trust
REJECT_SPEED_MPS = 15.0          # implausible speed for this scene — sanity cap, not a real limit
SYNC_MAX_EXTRAPOLATION_S = 0.25  # |dt| beyond this: drop the whole frame's tracks, don't extrapolate blindly


def sync_and_filter(tracks, dt):
    """Extrapolate every track's EKF state by dt (the gap between the radar frame's own
    timestamp and the camera timestamp it's being matched against), then drop suspicious
    tracks. Returns shallow copies — never mutates the tracker's own Track objects, so the
    live Tracker keeps coasting/confirming them normally regardless of what fusion accepts."""
    if abs(dt) > SYNC_MAX_EXTRAPOLATION_S:
        return []
    out = []
    for tr in tracks:
        if tr.misses > REJECT_MAX_MISSES:
            continue
        synced = copy.copy(tr)
        synced.x, synced.P = radar.Track.step_state(tr.x, tr.P, dt)
        if synced.sigma_xy_m > REJECT_SIGMA_XY_M or synced.speed_mps > REJECT_SPEED_MPS:
            continue
        out.append(synced)
    return out
