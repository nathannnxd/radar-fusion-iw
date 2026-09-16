"""dump_radar.py — записать сырой поток данных IWR1642 в .bin для отладки парсера.

Запускать на ноутбуке, к которому подключён радар (не в Colab).

    python dump_radar.py                        # найти порты, записать 10 с
    python dump_radar.py --sec 15 --cfg my.cfg  # с отправкой конфига перед записью
    python dump_radar.py --data COM7 --cli COM6 # порты вручную

Что делает:
  1) печатает все COM-порты (ищем два XDS110: User UART = команды, Auxiliary = данные);
  2) если указан --cfg, отправляет его в командный порт (115200);
  3) читает порт данных (921600) N секунд, пишет байты как есть в radar_dump_<время>.bin;
  4) в конце считает, сколько кадров (magic word TI) попало в дамп.
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
    print("Найденные COM-порты:")
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
        print(f"Отправляю {cfg_path} в {cli_port} ...")
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            ser.write((line + "\n").encode())
            time.sleep(0.05)
            reply = ser.read_all().decode(errors="ignore").strip()
            if reply:
                print(f"  > {line}\n    < {reply.splitlines()[-1]}")
    print("Конфиг отправлен.")


def dump(data_port, seconds, out_path):
    total = 0
    t_end = time.time() + seconds
    with serial.Serial(data_port, 921600, timeout=0.1) as ser, open(out_path, "wb") as f:
        print(f"Пишу {data_port} -> {out_path}, {seconds} с ...")
        while time.time() < t_end:
            chunk = ser.read(4096)
            if chunk:
                f.write(chunk)
                total += len(chunk)
    print(f"Записано {total} байт.")
    with open(out_path, "rb") as f:
        n_frames = f.read().count(MAGIC)
    print(f"Кадров в дампе: {n_frames} (~{n_frames / max(seconds, 1):.1f} кадр/с)")
    if n_frames == 0:
        print("Кадров нет: проверьте, что конфиг отправлен (--cfg) и порт данных выбран верно.")
    return n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", help="командный порт (115200)")
    ap.add_argument("--data", help="порт данных (921600)")
    ap.add_argument("--cfg", help="файл .cfg для отправки перед записью")
    ap.add_argument("--sec", type=int, default=10, help="сколько секунд писать")
    args = ap.parse_args()

    cli, data = list_ports()
    cli = args.cli or cli
    data = args.data or data
    if not data:
        sys.exit("Порт данных не найден. Укажите --data COMx (см. список выше).")
    print(f"CLI: {cli or '-'}   DATA: {data}")

    if args.cfg:
        if not cli:
            sys.exit("Для --cfg нужен командный порт: --cli COMx")
        send_config(cli, args.cfg)
        time.sleep(0.5)

    out = f"radar_dump_{datetime.now():%Y%m%d_%H%M%S}.bin"
    dump(data, args.sec, out)
    print(f"\nГотово. Пришлите файл {out} (и .cfg, если отправляли).")


if __name__ == "__main__":
    main()
