"""dump_radar.py — record the raw IWR1642 data stream to .bin for debugging the parser.

Run on the laptop the radar is connected to (not in Colab).

    python dump_radar.py                        # find ports, record 10 s
    python dump_radar.py --sec 15 --cfg my.cfg  # send config before recording
    python dump_radar.py --data COM7 --cli COM6 # ports manually

What it does:
  1) prints all COM ports (looking for two XDS110: User UART = commands, Auxiliary = data);
  2) if --cfg is given, sends it to the command port (115200);
  3) reads the data port (921600) for N seconds, writes the bytes as-is to radar_dump_<time>.bin;
  4) at the end counts how many frames (magic word TI) ended up in the dump.
"""
import argparse
import sys
import time
from datetime import datetime

import serial
import serial.tools.list_ports

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    print("Found COM ports:")
    for p in ports:
        print(f"  {p.device:8s} {p.description}")
    cli = data = None
    for p in ports:
        d = p.description.lower()
        if "xds110" in d and ("application" in d or "user" in d):
            cli = p.device
        elif "xds110" in d and ("auxiliary" in d or "data" in d):
            data = p.device
    return cli, data


def send_config(cli_port, cfg_path):
    with serial.Serial(cli_port, 115200, timeout=1) as ser, open(cfg_path, encoding="utf-8") as f:
        print(f"Sending {cfg_path} to {cli_port} ...")
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            ser.write((line + "\n").encode())
            time.sleep(0.05)
            reply = ser.read_all().decode(errors="ignore").strip()
            if reply:
                print(f"  > {line}\n    < {reply.splitlines()[-1]}")
    print("Config sent.")


def dump(data_port, seconds, out_path):
    total = 0
    t_end = time.time() + seconds
    with serial.Serial(data_port, 921600, timeout=0.1) as ser, open(out_path, "wb") as f:
        print(f"Writing {data_port} -> {out_path}, {seconds} s ...")
        while time.time() < t_end:
            chunk = ser.read(4096)
            if chunk:
                f.write(chunk)
                total += len(chunk)
    print(f"Wrote {total} bytes.")
    with open(out_path, "rb") as f:
        n_frames = f.read().count(MAGIC)
    print(f"Frames in dump: {n_frames} (~{n_frames / max(seconds, 1):.1f} frames/s)")
    if n_frames == 0:
        print("No frames: check that the config was sent (--cfg) and the data port is correct.")
    return n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", help="command port (115200)")
    ap.add_argument("--data", help="data port (921600)")
    ap.add_argument("--cfg", help=".cfg file to send before recording")
    ap.add_argument("--sec", type=int, default=10, help="how many seconds to record")
    args = ap.parse_args()

    cli, data = list_ports()
    cli = args.cli or cli
    data = args.data or data
    if not data:
        sys.exit("Data port not found. Specify --data COMx (see list above).")
    print(f"CLI: {cli or '-'}   DATA: {data}")

    if args.cfg:
        if not cli:
            sys.exit("--cfg requires a command port: --cli COMx")
        send_config(cli, args.cfg)
        time.sleep(0.5)

    out = f"radar_dump_{datetime.now():%Y%m%d_%H%M%S}.bin"
    dump(data, args.sec, out)
    print(f"\nDone. Send the file {out} (and .cfg, if you sent one).")


if __name__ == "__main__":
    main()
