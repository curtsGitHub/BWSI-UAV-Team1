#!/usr/bin/env python3
"""Download a PX4 .ulg flight log over MAVLink and print a hardware-health summary
(vibration, accel clipping, battery sag, altitude-estimate drift) to stdout.

Usage:
  python3 pull_px4_log.py udp:0.0.0.0:14550          # via mavros GCS forward
  python3 pull_px4_log.py /dev/ttyACM0 --baud 921600 # direct serial (STOP mavros first,
                                                     # two readers on one port corrupts both)
  python3 pull_px4_log.py --analyze flight.ulg       # skip download, just analyze a file
  python3 pull_px4_log.py udp:0.0.0.0:14550 --list   # list logs, download nothing

Downloads the LATEST log unless --log-id N is given.
Dependencies: pip3 install pymavlink pyulog
"""
import argparse
import os
import sys
import time


# ── MAVLink log transfer ─────────────────────────────────────────────────────────

def _connect(url, baud):
    from pymavlink import mavutil
    print(f"[LOG] connecting to {url} ...", flush=True)
    if url.startswith(("udp:", "udpin:", "udpout:", "tcp:")):
        m = mavutil.mavlink_connection(url)
    else:
        m = mavutil.mavlink_connection(url, baud=baud)
    m.wait_heartbeat(timeout=15)
    print(f"[LOG] heartbeat from system {m.target_system}", flush=True)
    return m


def _list_logs(m, timeout=10):
    """LOG_REQUEST_LIST -> [(id, size_bytes, utc)] sorted by id."""
    m.mav.log_request_list_send(m.target_system, m.target_component, 0, 0xFFFF)
    logs, expected, deadline = {}, None, time.time() + timeout
    while time.time() < deadline:
        msg = m.recv_match(type="LOG_ENTRY", blocking=True, timeout=2)
        if msg is None:
            if logs:
                break
            continue
        expected = msg.num_logs
        logs[msg.id] = (msg.size, msg.time_utc)
        if expected and len(logs) >= expected:
            break
    return sorted((i, s, t) for i, (s, t) in logs.items())


def _download(m, log_id, size, out_path, timeout=15):
    """LOG_REQUEST_DATA loop; PX4 answers with LOG_DATA chunks of <=90 bytes."""
    CHUNK = 90 * 500                      # request half-megabit windows
    with open(out_path, "wb") as f:
        ofs, last_data = 0, time.time()
        while ofs < size:
            m.mav.log_request_data_send(m.target_system, m.target_component,
                                        log_id, ofs, min(CHUNK, size - ofs))
            window_end = min(ofs + CHUNK, size)
            while ofs < window_end:
                msg = m.recv_match(type="LOG_DATA", blocking=True, timeout=3)
                if msg is None:
                    if time.time() - last_data > timeout:
                        raise TimeoutError(f"log download stalled at {ofs}/{size} bytes")
                    break                 # re-request the window
                if msg.id != log_id or msg.ofs != ofs:
                    continue              # stale/out-of-order chunk
                data = bytes(msg.data[: msg.count])
                f.write(data)
                ofs += len(data)
                last_data = time.time()
                if ofs % (90 * 5000) < 90:
                    print(f"[LOG] {ofs}/{size} bytes ({100*ofs//size}%)", flush=True)
    m.mav.log_request_end_send(m.target_system, m.target_component)
    print(f"[LOG] saved {out_path} ({size} bytes)", flush=True)


# ── ULog analysis ────────────────────────────────────────────────────────────────

def _stats(values):
    import numpy as np
    v = np.asarray(values, dtype=float)
    return f"mean={v.mean():+.3f} max={v.max():+.3f} min={v.min():+.3f}"

def analyze(path):
    """Print the fields that decide hardware-vs-software to stdout. Field names vary
    across PX4 versions, so match by substring and print whatever exists."""
    import numpy as np
    from pyulog import ULog

    ulog = ULog(path)
    dur = (ulog.last_timestamp - ulog.start_timestamp) / 1e6
    print(f"\n=== {os.path.basename(path)} | duration {dur:.1f}s | "
          f"{len(ulog.dropouts)} logger dropouts ===")

    WANT = {   # dataset-name substring -> field substrings worth printing
        "vehicle_imu_status": ["vibration_metric", "clipping"],
        "estimator_status":   ["vibe"],
        "battery_status":     ["voltage_filtered_v", "current_filtered_a"],
        "vehicle_local_position": ["dist_bottom"],
        "vehicle_attitude":   [],          # handled specially below (roll/pitch range)
        "actuator_controls_0": [],         # handled specially below (hover thrust)
    }
    for d in ulog.data_list:
        for want_name, fields in WANT.items():
            if want_name not in d.name:
                continue
            for key in d.data:
                if any(f in key for f in fields):
                    print(f"  {d.name}[{d.multi_id}].{key}: {_stats(d.data[key])}")
            if "vehicle_attitude" in d.name and "q[0]" in d.data:
                q0, q1 = np.array(d.data["q[0]"]), np.array(d.data["q[1]"])
                q2, q3 = np.array(d.data["q[2]"]), np.array(d.data["q[3]"])
                roll = np.degrees(np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2)))
                pitch = np.degrees(np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1, 1)))
                print(f"  attitude roll(deg): {_stats(roll)}")
                print(f"  attitude pitch(deg): {_stats(pitch)}")
            if "actuator_controls" in d.name and "control[3]" in d.data:
                thr = np.array(d.data["control[3]"])
                air = thr > 0.15          # crude airborne mask
                if air.any():
                    print(f"  thrust while airborne: {_stats(thr[air])}"
                          f"  <- hover value; compare before/after the new legs")

    print("\nHow to read this:")
    print("  accel vibration_metric > ~5 m/s^2, or ANY clipping counts rising -> ")
    print("    vibration reaching the IMU (leg resonance) corrupting the EKF")
    print("  voltage sagging + hover thrust creeping up -> weight/battery margin")
    print("  dist_bottom vs local z disagreeing -> rangefinder obstructed (leg in view?)")


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", nargs="?", help="MAVLink url (udp:0.0.0.0:14550, /dev/ttyACM0)")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--log-id", type=int, default=None, help="download this id, not latest")
    p.add_argument("--list", action="store_true", help="list logs and exit")
    p.add_argument("--analyze", metavar="ULG", help="analyze an existing .ulg, no download")
    p.add_argument("--out", default=None, help="output filename for the download")
    a = p.parse_args()

    if a.analyze:
        analyze(a.analyze)
        return
    if not a.url:
        p.error("give a MAVLink url to download, or --analyze <file.ulg>")

    m = _connect(a.url, a.baud)
    logs = _list_logs(m)
    if not logs:
        sys.exit("[LOG] flight controller reported no logs")
    print(f"[LOG] {len(logs)} logs on the SD card:")
    for i, size, utc in logs[-10:]:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.gmtime(utc)) if utc else "unknown-time"
        print(f"  id={i}  {size/1e6:7.2f} MB  {stamp}")
    if a.list:
        return

    log_id, size, utc = logs[-1] if a.log_id is None else next(
        l for l in logs if l[0] == a.log_id)
    out = a.out or f"px4_log_{log_id}.ulg"
    _download(m, log_id, size, out)
    analyze(out)


if __name__ == "__main__":
    main()
