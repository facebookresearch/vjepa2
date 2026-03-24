#!/usr/bin/env python3
# src/siml/scripts/power_wrap.py
import argparse, subprocess, sys, time, json
from threading import Thread, Event
try:
    import pynvml
except ImportError:
    print("Please: pip install nvidia-ml-py3", file=sys.stderr)
    sys.exit(2)

def sample_power(stop_evt, interval_s=1.0, device_index=0):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    readings = []
    t0 = time.time()
    while not stop_evt.is_set():
        t = time.time() - t0
        p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # W
        readings.append((t, p))
        stop_evt.wait(interval_s)
    pynvml.nvmlShutdown()
    return readings

def main():
    ap = argparse.ArgumentParser(description="Wrap a command and record GPU energy")
    ap.add_argument("--out", required=True, help="write JSON with energy/avg power")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="use `--` then your command")
    args = ap.parse_args()

    if not args.cmd or args.cmd[0] != "--":
        print("Usage: power_wrap.py --out OUT.json -- <cmd> ...", file=sys.stderr)
        sys.exit(1)
    cmd = args.cmd[1:]

    stop = Event()
    readings = []
    def run_sampler():
        nonlocal readings
        readings = sample_power(stop, interval_s=1.0, device_index=args.device)

    thr = Thread(target=run_sampler)
    thr.start()
    t0 = time.time()
    rc = 0
    try:
        rc = subprocess.call(cmd)
    finally:
        stop.set()
        thr.join()
    t1 = time.time()

    if len(readings) < 2:
        data = {"gpu_energy_Wh": 0.0, "avg_power_W": 0.0, "duration_s": t1 - t0}
    else:
        ts = [t for t,_ in readings]
        ps = [p for _,p in readings]
        ws = 0.0
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i-1]
            ws += 0.5 * (ps[i] + ps[i-1]) * dt
        data = {"gpu_energy_Wh": ws/3600.0, "avg_power_W": sum(ps)/len(ps), "duration_s": ts[-1]}

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    sys.exit(rc)

if __name__ == "__main__":
    main()
