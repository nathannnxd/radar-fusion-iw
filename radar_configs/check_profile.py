"""Derive the numbers that matter from an IWR1642 mmWave-SDK cfg: v_max, velocity bin, range, range
resolution, active frame time and the L3 radar-cube footprint — and flag the firmware limits the demo
enforces. Usage: python check_profile.py radar_configs/hangar_v9.cfg [more.cfg ...]

Formulas (TI SWRA553A, SPYY005; TDM-MIMO halving per TI E2E; SDK 3.x objectdetection.c sizing):
  T_c   = idleTime + rampEndTime                      (per chirp)
  v_max = lambda / (4 * T_c * numTx)                  lambda at startFreq
  v_res = lambda / (2 * numLoops * T_c * numTx)        = 2 * v_max / numLoops  (Doppler bin)
  r_max = IF_max * c / (2 * slope),  IF_max = 0.9 * digOutSampleRate (IWR1642 IF <= 5 MHz)
  r_res = c / (2 * slope * numAdcSamples / digOutSampleRate)
  radar cube = numRangeBins * numLoops * numVirtualAntennas * 4 B  (+ detection matrix numRangeBins * numDopplerBins * 2 B)
"""
import math
import sys

C = 299_792_458.0
L3_BYTES = 768 * 1024                      # IWR1642 shared L3 (SWRS212)
IF_MAX_HZ = 5.0e6                          # IWR1642 IF bandwidth limit


def parse(path):
    cfg = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        cfg.setdefault(parts[0], []).append([float(p) if p.replace(".", "", 1).replace("-", "", 1).isdigit() else p
                                             for p in parts[1:]])
    return cfg


def analyse(path):
    cfg = parse(path)
    p = cfg["profileCfg"][0]
    start_ghz, idle, adc_start, ramp_end, slope, samples, rate_ksps = p[1], p[2], p[3], p[4], p[7], int(p[9]), p[10]
    frame = cfg["frameCfg"][0]
    num_tx = int(frame[1]) - int(frame[0]) + 1
    loops = int(frame[2])
    period_ms = frame[4]
    ch = cfg["channelCfg"][0]
    num_rx = bin(int(ch[0])).count("1")
    virt = num_rx * num_tx
    lam = C / (start_ghz * 1e9)
    tc = (idle + ramp_end) * 1e-6
    v_max = lam / (4 * tc * num_tx)
    v_res = 2 * v_max / loops
    rate = rate_ksps * 1e3
    sampling_us = samples / rate * 1e6
    bw_sampled_hz = slope * 1e12 * sampling_us * 1e-6
    bw_total_ghz = slope * ramp_end / 1000.0
    if_max = min(IF_MAX_HZ, 0.9 * rate)              # TI app notes: usable IF up to ~0.9 Fs (complex 1x)
    if_cons = min(IF_MAX_HZ, 0.8 * rate)             # the mmWave Demo Visualizer quotes ~0.8 Fs -- conservative
    r_max = if_max * C / (2 * slope * 1e12)
    r_cons = if_cons * C / (2 * slope * 1e12)
    r_res = C / (2 * bw_sampled_hz)
    range_bins = 1 << (samples - 1).bit_length()
    dopp_bins = 1 << (loops - 1).bit_length()
    cube = range_bins * loops * virt * 4
    det = range_bins * dopp_bins * 2
    active_ms = num_tx * loops * tc * 1e3
    print(f"== {path}")
    print(f"  TX {num_tx} / RX {num_rx} -> {virt} virtual antennas; T_c = {idle:g} + {ramp_end:g} = {tc*1e6:.1f} us")
    print(f"  v_max = +-{v_max:.2f} m/s   Doppler bin = {v_res:.3f} m/s   ({loops} loops -> {dopp_bins} Doppler bins)")
    print(f"  r_max = {r_cons:.1f}-{r_max:.1f} m (IF 0.8-0.9 Fs = {if_cons/1e6:.2f}-{if_max/1e6:.2f} MHz)   r_res = {r_res*100:.1f} cm   sampled BW {bw_sampled_hz/1e6:.0f} MHz, total sweep {bw_total_ghz:.2f} GHz")
    print(f"  radar cube {cube/1024:.0f} KiB + det matrix {det/1024:.0f} KiB = {(cube+det)/1024:.0f} KiB of {L3_BYTES/1024:.0f} KiB L3 ({100*(cube+det)/L3_BYTES:.0f} %)")
    print(f"  frame active {active_ms:.2f} ms of {period_ms:g} ms ({100*active_ms/period_ms:.1f} % duty)")
    problems = []
    if slope > 100: problems.append("slope > 100 MHz/us")
    if samples < 64: problems.append("numAdcSamples < 64")
    if rate_ksps > 6250: problems.append("digOutSampleRate > 6250 ksps (complex 1x)")
    if adc_start + sampling_us > ramp_end: problems.append(f"adcStart {adc_start:g} + sampling {sampling_us:.1f} us exceeds rampEnd {ramp_end:g} us")
    elif ramp_end - (adc_start + sampling_us) < 0.5: print(f"  note: only {ramp_end - (adc_start + sampling_us):.2f} us between end of sampling and rampEnd")
    min_idle = 2 if bw_total_ghz < 1 else 3.5 if bw_total_ghz < 2 else 5 if bw_total_ghz < 3 else 6.5
    if idle < min_idle: problems.append(f"idleTime {idle:g} < {min_idle} us required for a {bw_total_ghz:.2f} GHz sweep")
    if start_ghz + bw_total_ghz > 81.0: problems.append("sweep leaves the 77-81 GHz band")
    if loops % 4 or dopp_bins < 16: problems.append("numLoops must be a multiple of 4 with >= 16 Doppler bins")
    if 4 * num_rx * num_tx * loops > 16384: problems.append("4*numRx*numTx*numLoops > 16384 (L1/L2)")
    if cube + det > L3_BYTES: problems.append("radar cube + detection matrix exceed L3")
    if active_ms > 0.5 * period_ms: problems.append("frame duty > 50 %")
    dop = [f for f in cfg.get("cfarFovCfg", []) if int(f[1]) == 1]
    if dop and abs(abs(dop[0][3]) - v_max) > 0.05: problems.append(f"cfarFovCfg Doppler limits {dop[0][2]:g}..{dop[0][3]:g} do not match v_max {v_max:.2f}")
    rng = [f for f in cfg.get("cfarFovCfg", []) if int(f[1]) == 0]
    if rng and rng[0][3] > r_max + 0.05: problems.append(f"cfarFovCfg range max {rng[0][3]:g} > r_max {r_max:.2f}")
    for k in ("clutterRemoval", "extendedMaxVelocity"):
        if k in cfg and int(cfg[k][0][1]) != 0: problems.append(f"{k} is enabled (ego-motion pipeline expects 0)")
    print("  limits: " + ("OK" if not problems else "; ".join(problems)))
    return not problems


if __name__ == "__main__":
    ok = all(analyse(a) for a in sys.argv[1:])
    sys.exit(0 if ok else 1)
