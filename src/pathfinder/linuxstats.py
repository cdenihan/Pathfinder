#!/usr/bin/env python3
"""
================================================================================
 linux_stats_viewer.py — a btop-style full-screen TUI for regular Linux systems
================================================================================

WHAT IT IS
----------
This is a Jetson-free rewrite of the original tegrastats viewer. It does not use
NVIDIA tegrastats, JetPack, CUDA, or any vendor-specific command. Instead it
samples standard Linux kernel interfaces directly:

    * CPU load       /proc/stat
    * CPU frequency  /sys/devices/system/cpu/.../cpufreq and /proc/cpuinfo
    * Memory/swap    /proc/meminfo
    * Temperatures   /sys/class/thermal and /sys/class/hwmon
    * Power          /sys/class/powercap and /sys/class/power_supply when present
    * Disk I/O       /proc/diskstats
    * Network I/O    /proc/net/dev

The dashboard expands to fill the whole terminal and reflows when resized.
No third-party Python dependencies are required.

USAGE
-----
Run the live TUI:

    python3 linux_stats_viewer.py

Set sampling interval:

    python3 linux_stats_viewer.py --interval 0.5

Print one plain-text sample instead of opening the TUI:

    python3 linux_stats_viewer.py --once

Log every sample to CSV while the dashboard runs:

    python3 linux_stats_viewer.py --log stats.csv

Keys:  q / Esc  quit        Ctrl+C also works
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import curses
import glob
import os
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from shutil import disk_usage
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# SMALL FILE HELPERS
# ---------------------------------------------------------------------------


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def read_int(path: str) -> Optional[int]:
    text = read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# RAW LINUX READERS
# ---------------------------------------------------------------------------
CPU_FIELDS = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
    "guest",
    "guest_nice",
)


def read_proc_stat_cpu() -> Dict[str, Tuple[int, ...]]:
    """Return {'cpu': totals, 'cpu0': totals, ...} from /proc/stat."""
    cpus: Dict[str, Tuple[int, ...]] = {}
    text = read_text("/proc/stat")
    if not text:
        return cpus
    for line in text.splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        if not re.fullmatch(r"cpu\d*", parts[0]):
            continue
        try:
            nums = tuple(int(x) for x in parts[1 : 1 + len(CPU_FIELDS)])
        except ValueError:
            continue
        cpus[parts[0]] = nums
    return cpus


def cpu_percent(prev: Tuple[int, ...], cur: Tuple[int, ...]) -> float:
    """CPU busy percent between two /proc/stat samples."""
    prev_total = sum(prev)
    cur_total = sum(cur)
    total_delta = cur_total - prev_total
    if total_delta <= 0:
        return 0.0
    # idle + iowait are idle time; everything else is busy enough for a dashboard.
    prev_idle = prev[3] + (prev[4] if len(prev) > 4 else 0)
    cur_idle = cur[3] + (cur[4] if len(cur) > 4 else 0)
    busy_delta = total_delta - (cur_idle - prev_idle)
    return max(0.0, min(100.0, busy_delta / total_delta * 100.0))


def read_cpuinfo_freqs() -> Dict[int, float]:
    """Fallback current-ish CPU MHz values parsed from /proc/cpuinfo."""
    text = read_text("/proc/cpuinfo") or ""
    freqs: Dict[int, float] = {}
    cur_cpu: Optional[int] = None
    for line in text.splitlines():
        if line.startswith("processor"):
            m = re.search(r":\s*(\d+)", line)
            cur_cpu = int(m.group(1)) if m else None
        elif line.lower().startswith("cpu mhz") and cur_cpu is not None:
            m = re.search(r":\s*([\d.]+)", line)
            if m:
                freqs[cur_cpu] = float(m.group(1))
    return freqs


def read_cpu_freqs() -> Dict[int, Optional[float]]:
    """Return per-core MHz from cpufreq sysfs where available, else /proc/cpuinfo."""
    freqs: Dict[int, Optional[float]] = {}
    cpu_dirs = sorted(
        glob.glob("/sys/devices/system/cpu/cpu[0-9]*"),
        key=lambda p: int(re.search(r"cpu(\d+)$", p).group(1)),
    )
    fallback = read_cpuinfo_freqs()
    for path in cpu_dirs:
        m = re.search(r"cpu(\d+)$", path)
        if not m:
            continue
        idx = int(m.group(1))
        khz = None
        for name in ("scaling_cur_freq", "cpuinfo_cur_freq"):
            khz = read_int(os.path.join(path, "cpufreq", name))
            if khz is not None:
                break
        if khz is not None:
            freqs[idx] = khz / 1000.0
        elif idx in fallback:
            freqs[idx] = fallback[idx]
        else:
            freqs[idx] = None
    if not freqs:
        freqs = {i: mhz for i, mhz in fallback.items()}
    return freqs


def read_meminfo() -> Dict[str, int]:
    out: Dict[str, int] = {}
    text = read_text("/proc/meminfo") or ""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].rstrip(":")
            try:
                out[key] = int(parts[1])  # kB
            except ValueError:
                pass
    return out


def collect_memory() -> Dict[str, float]:
    mi = read_meminfo()
    total = mi.get("MemTotal", 0) / 1024.0
    available = mi.get("MemAvailable", 0) / 1024.0
    free = mi.get("MemFree", 0) / 1024.0
    cached = (
        mi.get("Cached", 0) + mi.get("SReclaimable", 0) - mi.get("Shmem", 0)
    ) / 1024.0
    buffers = mi.get("Buffers", 0) / 1024.0
    used = max(0.0, total - available) if total else 0.0
    swap_total = mi.get("SwapTotal", 0) / 1024.0
    swap_free = mi.get("SwapFree", 0) / 1024.0
    swap_used = max(0.0, swap_total - swap_free)
    return {
        "mem_used_mb": used,
        "mem_total_mb": total,
        "mem_available_mb": available,
        "mem_free_mb": free,
        "mem_cached_mb": max(0.0, cached),
        "mem_buffers_mb": buffers,
        "swap_used_mb": swap_used,
        "swap_total_mb": swap_total,
    }


def clean_sensor_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name[:40] or "sensor"


def collect_temps() -> Dict[str, float]:
    """Read temperatures from thermal zones and hwmon chips."""
    temps: Dict[str, float] = {}

    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        label = read_text(os.path.join(zone, "type")) or os.path.basename(zone)
        raw = read_int(os.path.join(zone, "temp"))
        if raw is None:
            continue
        c = raw / 1000.0 if abs(raw) > 1000 else float(raw)
        if -30 <= c <= 130:
            base = clean_sensor_name(label)
            key = base
            suffix = 2
            while key in temps:
                key = f"{base}_{suffix}"
                suffix += 1
            temps[key] = c

    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        chip = clean_sensor_name(
            read_text(os.path.join(hw, "name")) or os.path.basename(hw)
        )
        for inp in sorted(glob.glob(os.path.join(hw, "temp*_input"))):
            m = re.search(r"temp(\d+)_input$", inp)
            if not m:
                continue
            idx = m.group(1)
            raw = read_int(inp)
            if raw is None:
                continue
            c = raw / 1000.0 if abs(raw) > 1000 else float(raw)
            if not (-30 <= c <= 130):
                continue
            label = read_text(os.path.join(hw, f"temp{idx}_label")) or f"temp{idx}"
            base = clean_sensor_name(f"{chip}_{label}")
            key = base
            suffix = 2
            while key in temps:
                key = f"{base}_{suffix}"
                suffix += 1
            temps[key] = c
    return temps


@dataclass
class PowercapSnapshot:
    energy_uj: int
    timestamp: float


def read_powercap_energy() -> Dict[str, PowercapSnapshot]:
    out: Dict[str, PowercapSnapshot] = {}
    t = time.monotonic()
    for energy_path in glob.glob("/sys/class/powercap/*/energy_uj") + glob.glob(
        "/sys/class/powercap/*/*/energy_uj"
    ):
        zone = os.path.dirname(energy_path)
        name = clean_sensor_name(
            read_text(os.path.join(zone, "name")) or os.path.basename(zone)
        )
        val = read_int(energy_path)
        if val is None:
            continue
        out[name] = PowercapSnapshot(val, t)
    return out


def collect_power(
    prev: Optional[Dict[str, PowercapSnapshot]],
) -> Tuple[Dict[str, float], Dict[str, PowercapSnapshot], Dict[str, Dict[str, object]]]:
    """Return (watts_by_zone, current_powercap_snapshot, power_supply_info)."""
    cur_caps = read_powercap_energy()
    watts: Dict[str, float] = {}
    if prev:
        for name, cur in cur_caps.items():
            old = prev.get(name)
            if not old:
                continue
            dt = cur.timestamp - old.timestamp
            de = cur.energy_uj - old.energy_uj
            if de < 0:
                # Counter wrapped or reset. Leave this sample blank rather than lying.
                continue
            if dt > 0:
                watts[name] = de / 1_000_000.0 / dt

    supplies: Dict[str, Dict[str, object]] = {}
    for ps in sorted(glob.glob("/sys/class/power_supply/*")):
        name = os.path.basename(ps)
        ptype = read_text(os.path.join(ps, "type")) or ""
        status = read_text(os.path.join(ps, "status")) or ""
        capacity = read_int(os.path.join(ps, "capacity"))
        power_now = read_int(os.path.join(ps, "power_now"))
        if power_now is None:
            cur_ua = read_int(os.path.join(ps, "current_now"))
            volt_uv = read_int(os.path.join(ps, "voltage_now"))
            if cur_ua is not None and volt_uv is not None:
                power_now = int(cur_ua * volt_uv / 1_000_000.0)  # microwatts
        item: Dict[str, object] = {"type": ptype, "status": status}
        if capacity is not None:
            item["capacity_pct"] = capacity
        if power_now is not None:
            item["watts"] = power_now / 1_000_000.0
        supplies[name] = item
    return watts, cur_caps, supplies


def read_diskstats() -> Dict[str, Tuple[int, int]]:
    """Return {device: (read_bytes, write_bytes)} from /proc/diskstats."""
    out: Dict[str, Tuple[int, int]] = {}
    text = read_text("/proc/diskstats") or ""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 14:
            continue
        dev = parts[2]
        # Skip partitions and noisy virtual devices in the aggregate.
        if re.match(r"^(loop|ram|zram|fd)\d+", dev):
            continue
        if re.match(r"^dm-\d+", dev):
            continue
        if re.match(r"^md\d+", dev):
            pass
        elif re.search(r"\d+$", dev) and not re.match(r"^nvme\d+n\d+$", dev):
            continue
        try:
            sectors_read = int(parts[5])
            sectors_written = int(parts[9])
        except ValueError:
            continue
        # Linux diskstats sectors are 512-byte units.
        out[dev] = (sectors_read * 512, sectors_written * 512)
    return out


def diff_rate(
    prev: Optional[Dict[str, Tuple[int, int]]],
    cur: Dict[str, Tuple[int, int]],
    dt: float,
) -> Dict[str, Tuple[float, float]]:
    rates: Dict[str, Tuple[float, float]] = {}
    if not prev or dt <= 0:
        return rates
    for name, (a, b) in cur.items():
        if name not in prev:
            continue
        pa, pb = prev[name]
        rates[name] = (max(0.0, (a - pa) / dt), max(0.0, (b - pb) / dt))
    return rates


def read_netdev() -> Dict[str, Tuple[int, int]]:
    """Return {interface: (rx_bytes, tx_bytes)} from /proc/net/dev."""
    out: Dict[str, Tuple[int, int]] = {}
    text = read_text("/proc/net/dev") or ""
    for line in text.splitlines()[2:]:
        if ":" not in line:
            continue
        iface, rest = line.split(":", 1)
        iface = iface.strip()
        if iface == "lo":
            continue
        parts = rest.split()
        if len(parts) < 16:
            continue
        try:
            out[iface] = (int(parts[0]), int(parts[8]))
        except ValueError:
            pass
    return out


def human_rate(bytes_per_sec: float) -> str:
    units = ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"]
    value = float(bytes_per_sec)
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            if unit == "B/s":
                return f"{value:.0f} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB/s"


def get_root_disk_usage() -> Tuple[int, int, int]:
    try:
        usage = disk_usage("/")
        return usage.used, usage.total, usage.free
    except OSError:
        return (0, 0, 0)


# ---------------------------------------------------------------------------
# SAMPLER
# ---------------------------------------------------------------------------
class LinuxSampler:
    """Polls Linux /proc and /sys and converts counter deltas into rates."""

    def __init__(self):
        self.prev_cpu: Optional[Dict[str, Tuple[int, ...]]] = None
        self.prev_disk: Optional[Dict[str, Tuple[int, int]]] = None
        self.prev_net: Optional[Dict[str, Tuple[int, int]]] = None
        self.prev_powercap: Optional[Dict[str, PowercapSnapshot]] = None
        self.prev_time: Optional[float] = None

    def sample(self) -> Dict[str, object]:
        t = time.monotonic()
        dt = max(0.001, t - self.prev_time) if self.prev_time is not None else 0.0

        cpu_now = read_proc_stat_cpu()
        freqs = read_cpu_freqs()
        cores: List[Tuple[float, Optional[float]]] = []
        cpu_avg = 0.0
        if self.prev_cpu:
            if "cpu" in cpu_now and "cpu" in self.prev_cpu:
                cpu_avg = cpu_percent(self.prev_cpu["cpu"], cpu_now["cpu"])
            core_keys = sorted(
                (k for k in cpu_now if re.fullmatch(r"cpu\d+", k)),
                key=lambda k: int(k[3:]),
            )
            for key in core_keys:
                idx = int(key[3:])
                pct = cpu_percent(self.prev_cpu.get(key, cpu_now[key]), cpu_now[key])
                cores.append((pct, freqs.get(idx)))
        else:
            core_keys = sorted(
                (k for k in cpu_now if re.fullmatch(r"cpu\d+", k)),
                key=lambda k: int(k[3:]),
            )
            for key in core_keys:
                idx = int(key[3:])
                cores.append((0.0, freqs.get(idx)))

        disk_now = read_diskstats()
        disk_rates = diff_rate(self.prev_disk, disk_now, dt)
        net_now = read_netdev()
        net_rates = diff_rate(self.prev_net, net_now, dt)
        power_watts, powercap_now, supplies = collect_power(self.prev_powercap)

        load1, load5, load15 = (
            os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        )
        root_used, root_total, root_free = get_root_disk_usage()

        self.prev_cpu = cpu_now
        self.prev_disk = disk_now
        self.prev_net = net_now
        self.prev_powercap = powercap_now
        self.prev_time = t

        return {
            "time": now_string(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else "linux",
            "uptime_s": safe_float((read_text("/proc/uptime") or "0").split()[0]),
            "cpu_avg_pct": cpu_avg,
            "cores": cores,
            "loadavg": (load1, load5, load15),
            "memory": collect_memory(),
            "temps": collect_temps(),
            "power_watts": power_watts,
            "power_supplies": supplies,
            "disk_rates": disk_rates,
            "net_rates": net_rates,
            "root_disk": {"used": root_used, "total": root_total, "free": root_free},
        }


class State:
    def __init__(self, hist_len: int = 2048):
        self.lock = threading.Lock()
        self.latest: Optional[Dict[str, object]] = None
        self.cpu_hist = deque(maxlen=hist_len)
        self.mem_hist = deque(maxlen=hist_len)
        self.stop = threading.Event()
        self.error: Optional[str] = None


class CsvLogger:
    def __init__(self, path: str):
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer: Optional[csv.DictWriter] = None
        self.fieldnames: List[str] = []

    @staticmethod
    def flatten(s: Dict[str, object]) -> Dict[str, object]:
        row: Dict[str, object] = {
            "timestamp": s.get("time", ""),
            "hostname": s.get("hostname", ""),
            "uptime_s": round(safe_float(s.get("uptime_s")), 1),
            "cpu_avg_pct": round(safe_float(s.get("cpu_avg_pct")), 2),
        }
        loadavg = s.get("loadavg", (0.0, 0.0, 0.0))
        if isinstance(loadavg, tuple):
            row["load1"], row["load5"], row["load15"] = [
                round(safe_float(x), 3) for x in loadavg
            ]

        for i, core in enumerate(s.get("cores", [])):  # type: ignore[union-attr]
            pct, mhz = core
            row[f"cpu{i}_pct"] = round(safe_float(pct), 2)
            row[f"cpu{i}_mhz"] = "" if mhz is None else round(safe_float(mhz), 1)

        mem = s.get("memory", {})
        if isinstance(mem, dict):
            for key, val in mem.items():
                row[key] = round(safe_float(val), 2)

        temps = s.get("temps", {})
        if isinstance(temps, dict):
            for key, val in temps.items():
                row[f"temp_{key}_c"] = round(safe_float(val), 2)

        power = s.get("power_watts", {})
        if isinstance(power, dict):
            for key, val in power.items():
                row[f"power_{key}_w"] = round(safe_float(val), 3)

        supplies = s.get("power_supplies", {})
        if isinstance(supplies, dict):
            for name, data in supplies.items():
                if isinstance(data, dict):
                    if "capacity_pct" in data:
                        row[f"battery_{name}_pct"] = data["capacity_pct"]
                    if "watts" in data:
                        row[f"battery_{name}_w"] = round(safe_float(data["watts"]), 3)
                    if "status" in data:
                        row[f"battery_{name}_status"] = data["status"]

        for prefix, rates in (
            ("disk", s.get("disk_rates", {})),
            ("net", s.get("net_rates", {})),
        ):
            if isinstance(rates, dict):
                for name, pair in rates.items():
                    rx, tx = pair
                    if prefix == "disk":
                        row[f"disk_{name}_read_Bps"] = round(safe_float(rx), 1)
                        row[f"disk_{name}_write_Bps"] = round(safe_float(tx), 1)
                    else:
                        row[f"net_{name}_rx_Bps"] = round(safe_float(rx), 1)
                        row[f"net_{name}_tx_Bps"] = round(safe_float(tx), 1)
        return row

    def write(self, s: Dict[str, object]) -> None:
        row = self.flatten(s)
        if self.writer is None:
            self.fieldnames = list(row)
            self.writer = csv.DictWriter(
                self.file, fieldnames=self.fieldnames, restval="", extrasaction="ignore"
            )
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        except OSError:
            pass


def sampler_thread(
    interval: float,
    state: State,
    logger: Optional[CsvLogger] = None,
    samples: Optional[int] = None,
) -> None:
    sampler = LinuxSampler()
    count = 0
    try:
        # Prime counters so the first visible sample has meaningful deltas.
        sampler.sample()
        time.sleep(min(max(interval, 0.05), 1.0))
        while not state.stop.is_set():
            s = sampler.sample()
            mem = s.get("memory", {})
            mem_pct = 0.0
            if isinstance(mem, dict):
                total = safe_float(mem.get("mem_total_mb"))
                used = safe_float(mem.get("mem_used_mb"))
                mem_pct = used / total * 100.0 if total else 0.0
            with state.lock:
                state.latest = s
                state.cpu_hist.append(safe_float(s.get("cpu_avg_pct")))
                state.mem_hist.append(mem_pct)
            if logger:
                logger.write(s)
            count += 1
            if samples is not None and count >= samples:
                state.stop.set()
                break
            state.stop.wait(max(0.05, interval))
    except Exception as exc:  # keep curses cleanup friendly
        state.error = f"sampler error: {exc}"
        state.stop.set()


# ---------------------------------------------------------------------------
# CURSES DRAWING
# ---------------------------------------------------------------------------
GREEN, YELLOW, RED, TITLE, DIMP, BLUE = 1, 2, 3, 4, 5, 6
BLOCKS = " ▁▂▃▄▅▆▇█"


def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(GREEN, curses.COLOR_GREEN, -1)
    curses.init_pair(YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(RED, curses.COLOR_RED, -1)
    curses.init_pair(TITLE, curses.COLOR_CYAN, -1)
    curses.init_pair(DIMP, curses.COLOR_WHITE, -1)
    curses.init_pair(BLUE, curses.COLOR_BLUE, -1)


def load_attr(pct: float) -> int:
    if pct < 50:
        return curses.color_pair(GREEN)
    if pct < 80:
        return curses.color_pair(YELLOW)
    return curses.color_pair(RED)


def temp_attr(c: float) -> int:
    if c < 60:
        return curses.color_pair(GREEN)
    if c < 80:
        return curses.color_pair(YELLOW)
    return curses.color_pair(RED)


def put(
    scr, y: int, x: int, text: str, attr: int = 0, xmax: Optional[int] = None
) -> None:
    h, w = scr.getmaxyx()
    limit = w if xmax is None else min(w, xmax)
    if not (0 <= y < h) or x >= limit:
        return
    if x < 0:
        text, x = text[-x:], 0
    try:
        scr.addstr(y, x, text[: max(0, limit - x)], attr)
    except curses.error:
        pass


def draw_box(scr, y: int, x: int, h: int, w: int, title: str = "") -> None:
    if h < 2 or w < 2:
        return
    put(scr, y, x, "┌" + "─" * (w - 2) + "┐")
    for r in range(1, h - 1):
        put(scr, y + r, x, "│")
        put(scr, y + r, x + w - 1, "│")
    put(scr, y + h - 1, x, "└" + "─" * (w - 2) + "┘")
    if title:
        put(scr, y, x + 2, f" {title} ", curses.color_pair(TITLE) | curses.A_BOLD)


def draw_gauge(
    scr, y: int, x: int, width: int, pct: float, attr: Optional[int] = None
) -> None:
    pct = max(0.0, min(100.0, pct))
    width = max(0, width)
    filled = int(round(pct / 100.0 * width))
    a = attr if attr is not None else load_attr(pct)
    put(scr, y, x, "█" * filled, a)
    put(scr, y, x + filled, "░" * (width - filled), curses.A_DIM)


def draw_graph(scr, y: int, x: int, h: int, w: int, hist: Iterable[float]) -> None:
    vals = list(hist)[-max(0, w) :]
    if h < 1 or w < 1:
        return
    for col, v in enumerate(vals):
        cx = x + w - len(vals) + col
        pct = max(0.0, min(100.0, safe_float(v)))
        eighths = pct / 100.0 * h * 8
        full, part = int(eighths // 8), int(eighths % 8)
        a = load_attr(pct)
        for r in range(min(full, h)):
            put(scr, y + h - 1 - r, cx, "█", a)
        if part and full < h:
            put(scr, y + h - 1 - full, cx, BLOCKS[part], a)


def panel_cpu(
    scr, y: int, x: int, h: int, w: int, s: Dict[str, object], hist: Iterable[float]
) -> None:
    draw_box(scr, y, x, h, w, "CPU")
    cores = s.get("cores", [])
    if not isinstance(cores, list):
        cores = []
    iy, ix, iw = y + 1, x + 2, w - 4
    rows_avail = max(1, h - 2)
    core_rows_target = max(2, rows_avail - 5)
    ncols = 1 if len(cores) <= core_rows_target else 2
    percol = max(1, -(-len(cores) // ncols))
    col_w = max(24, iw // ncols)
    bw = max(5, col_w - 24)

    for i, core in enumerate(cores):
        ry = iy + i % percol
        if ry >= y + h - 5:
            continue
        rx = ix + (i // percol) * col_w
        pct, mhz = core
        pct = safe_float(pct)
        put(scr, ry, rx, f"cpu{i:<3}", curses.A_BOLD)
        draw_gauge(scr, ry, rx + 7, bw, pct)
        put(scr, ry, rx + 7 + bw, f" {pct:5.1f}%", xmax=x + w - 1)
        if mhz is not None:
            put(
                scr,
                ry,
                rx + 7 + bw + 8,
                f" {safe_float(mhz):.0f}MHz",
                curses.A_DIM,
                xmax=x + w - 1,
            )

    avg = safe_float(s.get("cpu_avg_pct"))
    ay = min(y + h - 5, iy + percol)
    put(scr, ay, ix, "avg   ", curses.A_BOLD)
    draw_gauge(scr, ay, ix + 7, max(5, iw - 20), avg)
    put(
        scr,
        ay,
        ix + 7 + max(5, iw - 20),
        f" {avg:5.1f}%",
        curses.A_BOLD,
        xmax=x + w - 1,
    )

    loadavg = s.get("loadavg", (0.0, 0.0, 0.0))
    if isinstance(loadavg, tuple):
        put(
            scr,
            ay + 1,
            ix,
            f"load avg {loadavg[0]:.2f}  {loadavg[1]:.2f}  {loadavg[2]:.2f}",
            curses.A_DIM,
            xmax=x + w - 1,
        )

    gy = ay + 3
    gh = y + h - 1 - gy
    if gh >= 2:
        put(scr, gy - 1, ix, "history", curses.color_pair(DIMP) | curses.A_DIM)
        draw_graph(scr, gy, ix, gh, iw, hist)


def panel_memory(
    scr, y: int, x: int, h: int, w: int, s: Dict[str, object], hist: Iterable[float]
) -> None:
    draw_box(scr, y, x, h, w, "Memory")
    mem = s.get("memory", {})
    if not isinstance(mem, dict):
        return
    iy, ix, iw = y + 1, x + 2, w - 4
    bw = max(5, iw - 8)
    used = safe_float(mem.get("mem_used_mb"))
    total = safe_float(mem.get("mem_total_mb"))
    avail = safe_float(mem.get("mem_available_mb"))
    cached = safe_float(mem.get("mem_cached_mb"))
    pct = used / total * 100 if total else 0.0
    put(scr, iy, ix, "RAM  ", curses.A_BOLD)
    draw_gauge(scr, iy, ix + 6, bw, pct)
    put(
        scr,
        iy + 1,
        ix + 6,
        f"{used:.0f}/{total:.0f} MB ({pct:.0f}%)  avail {avail:.0f} MB  cache {cached:.0f} MB",
        curses.A_DIM,
        xmax=x + w - 1,
    )

    sw_used = safe_float(mem.get("swap_used_mb"))
    sw_total = safe_float(mem.get("swap_total_mb"))
    sw_pct = sw_used / sw_total * 100 if sw_total else 0.0
    row = iy + 3
    if row < y + h - 1:
        put(scr, row, ix, "SWAP ", curses.A_BOLD)
        draw_gauge(scr, row, ix + 6, bw, sw_pct)
        put(
            scr,
            row + 1,
            ix + 6,
            f"{sw_used:.0f}/{sw_total:.0f} MB ({sw_pct:.0f}%)",
            curses.A_DIM,
            xmax=x + w - 1,
        )

    gy = row + 3
    gh = y + h - 1 - gy
    if gh >= 1:
        draw_graph(scr, gy, ix, gh, iw, hist)


def panel_temps(scr, y: int, x: int, h: int, w: int, s: Dict[str, object]) -> None:
    draw_box(scr, y, x, h, w, "Temperatures")
    temps = s.get("temps", {})
    if not isinstance(temps, dict) or not temps:
        put(
            scr,
            y + 1,
            x + 2,
            "no temperature sensors exposed in /sys",
            curses.A_DIM,
            xmax=x + w - 1,
        )
        return
    iy, ix, iw = y + 1, x + 2, w - 4
    bw = max(5, iw - 24)
    for i, (name, c_obj) in enumerate(
        sorted(temps.items(), key=lambda kv: kv[0].lower())
    ):
        if iy + i >= y + h - 1:
            break
        c = safe_float(c_obj)
        label = name[:12]
        put(scr, iy + i, ix, f"{label:<12}", curses.A_BOLD)
        draw_gauge(scr, iy + i, ix + 13, bw, c, attr=temp_attr(c))
        put(scr, iy + i, ix + 13 + bw, f" {c:5.1f}°C", temp_attr(c), xmax=x + w - 1)


def panel_power(scr, y: int, x: int, h: int, w: int, s: Dict[str, object]) -> None:
    draw_box(scr, y, x, h, w, "Power / Battery")
    iy, ix, iw = y + 1, x + 2, w - 4
    row = iy
    power = s.get("power_watts", {})
    supplies = s.get("power_supplies", {})
    any_rows = False
    if isinstance(power, dict) and power:
        peak = max((safe_float(v) for v in power.values()), default=1.0)
        bw = max(5, iw - 28)
        for name, val in sorted(power.items(), key=lambda kv: kv[0].lower()):
            if row >= y + h - 1:
                break
            watts = safe_float(val)
            put(scr, row, ix, f"{name[:12]:<12}", curses.A_BOLD)
            draw_gauge(
                scr,
                row,
                ix + 13,
                bw,
                watts / max(peak, 0.001) * 100.0,
                attr=curses.color_pair(GREEN),
            )
            put(scr, row, ix + 13 + bw, f" {watts:6.2f} W", xmax=x + w - 1)
            row += 1
            any_rows = True
    if isinstance(supplies, dict) and supplies:
        for name, data in sorted(supplies.items(), key=lambda kv: kv[0].lower()):
            if row >= y + h - 1:
                break
            if not isinstance(data, dict):
                continue
            cap = data.get("capacity_pct")
            watts = data.get("watts")
            status = data.get("status", "")
            bits = [str(status)] if status else []
            if cap is not None:
                bits.append(f"{cap}%")
            if watts is not None:
                bits.append(f"{safe_float(watts):.2f} W")
            put(scr, row, ix, f"{name[:12]:<12}", curses.A_BOLD)
            put(
                scr,
                row,
                ix + 13,
                "  ".join(bits) or "present",
                curses.A_DIM,
                xmax=x + w - 1,
            )
            row += 1
            any_rows = True
    if not any_rows:
        put(
            scr,
            row,
            ix,
            "no powercap or battery data exposed",
            curses.A_DIM,
            xmax=x + w - 1,
        )


def panel_io(scr, y: int, x: int, h: int, w: int, s: Dict[str, object]) -> None:
    draw_box(scr, y, x, h, w, "Disk / Network")
    iy, ix = y + 1, x + 2
    row = iy
    root = s.get("root_disk", {})
    if isinstance(root, dict):
        used = safe_float(root.get("used"))
        total = safe_float(root.get("total"))
        pct = used / total * 100.0 if total else 0.0
        bw = max(5, w - 28)
        put(scr, row, ix, "/     ", curses.A_BOLD)
        draw_gauge(scr, row, ix + 7, bw, pct)
        put(scr, row, ix + 7 + bw, f" {pct:4.0f}%", xmax=x + w - 1)
        row += 1
    disk_rates = s.get("disk_rates", {})
    if isinstance(disk_rates, dict):
        for name, pair in sorted(disk_rates.items())[: max(0, (y + h - 1 - row) // 2)]:
            if row >= y + h - 1:
                break
            rb, wb = pair
            put(scr, row, ix, f"{name[:10]:<10}", curses.A_BOLD)
            put(
                scr,
                row,
                ix + 11,
                f"read {human_rate(safe_float(rb)):<10} write {human_rate(safe_float(wb)):<10}",
                curses.A_DIM,
                xmax=x + w - 1,
            )
            row += 1
    net_rates = s.get("net_rates", {})
    if isinstance(net_rates, dict):
        for name, pair in sorted(net_rates.items())[: max(0, y + h - 1 - row)]:
            if row >= y + h - 1:
                break
            rx, tx = pair
            put(scr, row, ix, f"{name[:10]:<10}", curses.A_BOLD)
            put(
                scr,
                row,
                ix + 11,
                f"rx {human_rate(safe_float(rx)):<10} tx {human_rate(safe_float(tx)):<10}",
                curses.A_DIM,
                xmax=x + w - 1,
            )
            row += 1


def draw_frame(scr, state: State) -> None:
    scr.erase()
    H, W = scr.getmaxyx()
    with state.lock:
        s = state.latest
        cpu_hist = list(state.cpu_hist)
        mem_hist = list(state.mem_hist)
    if state.error:
        put(scr, 0, 0, state.error, curses.color_pair(RED))
        scr.refresh()
        return
    if s is None:
        put(scr, H // 2, max(0, W // 2 - 14), "collecting Linux stats...", curses.A_DIM)
        scr.refresh()
        return
    if H < 18 or W < 80:
        put(
            scr,
            0,
            0,
            f"terminal too small ({W}x{H}), need at least 80x18",
            curses.color_pair(RED),
        )
        scr.refresh()
        return

    body_h = H - 1
    left_w = max(48, int(W * 0.58))
    right_w = W - left_w
    rx = left_w
    mem_h = min(10, max(7, body_h // 3))
    cpu_h = body_h - mem_h
    panel_cpu(scr, 0, 0, cpu_h, left_w, s, cpu_hist)
    panel_memory(scr, cpu_h, 0, mem_h, left_w, s, mem_hist)

    temps_count = len(
        s.get("temps", {}) if isinstance(s.get("temps", {}), dict) else {}
    )
    top_h = min(max(5, temps_count + 2), max(5, body_h // 3))
    power_h = max(5, body_h // 4)
    io_h = body_h - top_h - power_h
    if io_h < 5:
        io_h = 5
        power_h = max(4, body_h - top_h - io_h)
    panel_temps(scr, 0, rx, top_h, right_w, s)
    panel_power(scr, top_h, rx, power_h, right_w, s)
    panel_io(scr, top_h + power_h, rx, io_h, right_w, s)

    put(scr, H - 1, 1, " q quit ", curses.A_REVERSE)
    tail = f"{s.get('hostname', 'linux')}  {s.get('time', '')} "
    put(scr, H - 1, W - len(tail) - 1, tail, curses.A_DIM)
    scr.refresh()


def tui(scr, state: State) -> None:
    curses.curs_set(0)
    init_colors()
    scr.timeout(200)
    while not state.stop.is_set():
        ch = scr.getch()
        if ch in (ord("q"), ord("Q"), 27):
            state.stop.set()
            break
        draw_frame(scr, state)
    draw_frame(scr, state)


# ---------------------------------------------------------------------------
# PLAIN OUTPUT + ENTRY POINT
# ---------------------------------------------------------------------------
def snapshot_text(s: Dict[str, object]) -> str:
    lines = [f"Linux stats snapshot  {s.get('time', '')}  {s.get('hostname', '')}"]
    lines.append(f"  uptime: {safe_float(s.get('uptime_s')) / 3600:.1f} hours")
    loadavg = s.get("loadavg", (0.0, 0.0, 0.0))
    if isinstance(loadavg, tuple):
        lines.append(
            f"  load average: {loadavg[0]:.2f}, {loadavg[1]:.2f}, {loadavg[2]:.2f}"
        )
    lines.append(f"  cpu avg: {safe_float(s.get('cpu_avg_pct')):.1f}%")
    for i, core in enumerate(s.get("cores", [])):  # type: ignore[union-attr]
        pct, mhz = core
        freq = "" if mhz is None else f" @ {safe_float(mhz):.0f} MHz"
        lines.append(f"    cpu{i}: {safe_float(pct):5.1f}%{freq}")
    mem = s.get("memory", {})
    if isinstance(mem, dict):
        used = safe_float(mem.get("mem_used_mb"))
        total = safe_float(mem.get("mem_total_mb"))
        pct = used / total * 100 if total else 0.0
        lines.append(f"  ram: {used:.0f}/{total:.0f} MB ({pct:.0f}%)")
        sw_used = safe_float(mem.get("swap_used_mb"))
        sw_total = safe_float(mem.get("swap_total_mb"))
        lines.append(f"  swap: {sw_used:.0f}/{sw_total:.0f} MB")
    temps = s.get("temps", {})
    if isinstance(temps, dict) and temps:
        for name, c in sorted(temps.items(), key=lambda kv: kv[0].lower())[:20]:
            lines.append(f"  temp {name}: {safe_float(c):.1f}°C")
    power = s.get("power_watts", {})
    if isinstance(power, dict) and power:
        for name, watts in sorted(power.items(), key=lambda kv: kv[0].lower()):
            lines.append(f"  power {name}: {safe_float(watts):.2f} W")
    supplies = s.get("power_supplies", {})
    if isinstance(supplies, dict) and supplies:
        for name, data in sorted(supplies.items(), key=lambda kv: kv[0].lower()):
            if isinstance(data, dict):
                bits = []
                if data.get("status"):
                    bits.append(str(data["status"]))
                if "capacity_pct" in data:
                    bits.append(f"{data['capacity_pct']}%")
                if "watts" in data:
                    bits.append(f"{safe_float(data['watts']):.2f} W")
                lines.append(
                    f"  power_supply {name}: " + (", ".join(bits) or "present")
                )
    disk_rates = s.get("disk_rates", {})
    if isinstance(disk_rates, dict) and disk_rates:
        for name, pair in sorted(disk_rates.items())[:10]:
            rb, wb = pair
            lines.append(
                f"  disk {name}: read {human_rate(safe_float(rb))}, write {human_rate(safe_float(wb))}"
            )
    net_rates = s.get("net_rates", {})
    if isinstance(net_rates, dict) and net_rates:
        for name, pair in sorted(net_rates.items())[:10]:
            rx, tx = pair
            lines.append(
                f"  net {name}: rx {human_rate(safe_float(rx))}, tx {human_rate(safe_float(tx))}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-screen Linux /proc + /sys resource viewer with optional CSV logging."
    )
    p.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="sampling interval in seconds (default: 1.0)",
    )
    p.add_argument(
        "--log",
        metavar="FILE.csv",
        dest="csv_path",
        help="write every sample to a CSV file",
    )
    p.add_argument(
        "--once", action="store_true", help="print one plain-text snapshot and exit"
    )
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help="collect N samples, useful with --log or redirected output",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    interval = max(0.05, args.interval)
    logger = CsvLogger(args.csv_path) if args.csv_path else None

    if args.once or not sys.stdout.isatty():
        sampler = LinuxSampler()
        sampler.sample()
        time.sleep(interval)
        s = sampler.sample()
        try:
            if logger:
                logger.write(s)
            print(snapshot_text(s))
        finally:
            if logger:
                logger.close()
        return

    state = State()
    thread = threading.Thread(
        target=sampler_thread, args=(interval, state, logger, args.samples), daemon=True
    )
    thread.start()

    def stop_handler(signum, frame):  # type: ignore[no-untyped-def]
        state.stop.set()

    old_int = signal.signal(signal.SIGINT, stop_handler)
    try:
        curses.wrapper(tui, state)
    finally:
        state.stop.set()
        thread.join(timeout=2)
        if logger:
            logger.close()
        signal.signal(signal.SIGINT, old_int)


if __name__ == "__main__":
    main()
