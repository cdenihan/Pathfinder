#!/usr/bin/env python3
"""
================================================================================
 powermetrics_viewer.py — a btop-style full-screen TUI for Apple `powermetrics`
================================================================================

WHAT IT IS
----------
Apple's `powermetrics` prints a large multi-line report for each sample. This
script turns that stream into a full-screen curses dashboard with panels for:

    * CPU       — one gauge per CPU core using active residency %, plus MHz
    * GPU       — GPU active residency %, MHz, power, and a scrolling history
    * Power     — CPU, GPU, ANE, and combined power rails in mW
    * I/O       — battery %, network throughput, disk throughput, sample period
    * Tasks     — top sampled tasks by CPU ms/s

The script intentionally uses only Python's standard library. On macOS, curses
ships with the system Python, but some python.org/homebrew builds may require
`pip install windows-curses` only on Windows. This tool is meant for macOS.

USAGE
-----
Run it directly. It will authenticate sudo first, then spawn powermetrics:

    python3 powermetrics_viewer.py

Make it executable:

    chmod +x powermetrics_viewer.py
    ./powermetrics_viewer.py

Run with extra powermetrics arguments:

    python3 powermetrics_viewer.py --powermetrics-args "-i 1000"

Pipe powermetrics in yourself. Keyboard input still works when /dev/tty exists:

    sudo powermetrics | python3 powermetrics_viewer.py

Replay a saved powermetrics log. This prints a plain-text snapshot of the last
sample instead of opening the TUI:

    python3 powermetrics_viewer.py powermetrics.log

Export parsed samples to CSV while the dashboard runs:

    python3 powermetrics_viewer.py --log stats.csv
    python3 powermetrics_viewer.py powermetrics.log --log stats.csv

Keys:  q / Esc  quit        Ctrl+C also works
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import curses
import os
import re
import shlex
import subprocess
import sys
import threading
from collections import deque
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, cast

# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------
SAMPLE_RE = re.compile(
    r"^\*\*\* Sampled system activity \((?P<time>.*?)\) "
    r"\((?P<elapsed>[\d.]+)ms elapsed\) \*\*\*"
)
TASK_RE = re.compile(
    r"^(?P<name>.+?)\s+"
    r"(?P<pid>-?\d+)\s+"
    r"(?P<cpu_ms>[\d.]+)\s+"
    r"(?P<user_pct>[\d.]+)\s+"
    r"(?P<deadline_lt2>[\d.]+)\s+"
    r"(?P<deadline_2_5>[\d.]+)\s+"
    r"(?P<wake_intr>[\d.]+)\s+"
    r"(?P<wake_pkg>[\d.]+)\s*$"
)
CPU_FREQ_RE = re.compile(r"^CPU (?P<id>\d+) frequency:\s+(?P<mhz>[\d.]+) MHz")
CPU_ACTIVE_RE = re.compile(r"^CPU (?P<id>\d+) active residency:\s+(?P<pct>[\d.]+)%")
CPU_IDLE_RE = re.compile(r"^CPU (?P<id>\d+) idle residency:\s+(?P<pct>[\d.]+)%")
CPU_DOWN_RE = re.compile(r"^CPU (?P<id>\d+) down residency:\s+(?P<pct>[\d.]+)%")
CLUSTER_ONLINE_RE = re.compile(r"^(?P<name>.+?-Cluster) Online:\s+(?P<pct>[\d.]+)%")
CLUSTER_FREQ_RE = re.compile(
    r"^(?P<name>.+?-Cluster) HW active frequency:\s+(?P<mhz>[\d.]+) MHz"
)
CLUSTER_ACTIVE_RE = re.compile(
    r"^(?P<name>.+?-Cluster) HW active residency:\s+(?P<pct>[\d.]+)%"
)
CLUSTER_IDLE_RE = re.compile(
    r"^(?P<name>.+?-Cluster) idle residency:\s+(?P<pct>[\d.]+)%"
)
CLUSTER_DOWN_RE = re.compile(
    r"^(?P<name>.+?-Cluster) down residency:\s+(?P<pct>[\d.]+)%"
)
GPU_FREQ_RE = re.compile(r"^GPU HW active frequency:\s+(?P<mhz>[\d.]+) MHz")
GPU_ACTIVE_RE = re.compile(r"^GPU HW active residency:\s+(?P<pct>[\d.]+)%")
GPU_IDLE_RE = re.compile(r"^GPU idle residency:\s+(?P<pct>[\d.]+)%")
POWER_RE = re.compile(r"^(?P<name>CPU|GPU|ANE) Power:\s+(?P<mw>[\d.]+) mW")
COMBINED_POWER_RE = re.compile(r"^Combined Power.*?:\s+(?P<mw>[\d.]+) mW")
BATTERY_RE = re.compile(r"^Battery: percent_charge:\s+(?P<pct>[\d.]+)")
NET_RE = re.compile(
    r"^(?P<dir>in|out):\s+(?P<packets>[\d.]+) packets/s,\s+(?P<bytes>[\d.]+) bytes/s"
)
DISK_RE = re.compile(
    r"^(?P<dir>read|write):\s+(?P<ops>[\d.]+) ops/s\s+(?P<kb>[\d.]+) KBytes/s"
)
IRQ_CPU_RE = re.compile(r"^CPU (?P<id>\d+):")
IRQ_TOTAL_RE = re.compile(r"^\s*Total IRQ:\s+(?P<irq>[\d.]+) interrupts/sec")


def _f(value: str) -> float:
    """Convert powermetrics numbers to float, accepting values like '.45'."""
    value = value.strip()
    if value.startswith("."):
        value = "0" + value
    return float(value)


def _section_name(line: str) -> Optional[str]:
    clean = line.strip()
    if clean == "*** Running tasks ***":
        return "tasks"
    if clean == "**** Battery and backlight usage ****":
        return "battery"
    if clean == "**** Network activity ****":
        return "network"
    if clean == "**** Disk activity ****":
        return "disk"
    if clean == "****  Interrupt distribution ****":
        return "interrupts"
    if clean == "**** Processor usage ****":
        return "processor"
    if clean == "**** GPU usage ****":
        return "gpu"
    if clean.startswith("***") or clean.startswith("****"):
        return None
    return "__same__"


def iter_sample_blocks(stream: Iterable[str]) -> Iterable[List[str]]:
    """Yield one complete powermetrics sample block at a time."""
    block: List[str] = []
    for line in stream:
        if SAMPLE_RE.match(line):
            if block:
                yield block
            block = [line]
        elif block:
            block.append(line)
    if block:
        yield block


def parse_sample(lines: List[str]) -> Optional[dict]:
    """Parse one multi-line powermetrics sample block into a stats dict."""
    if not lines:
        return None
    m = SAMPLE_RE.match(lines[0])
    if not m:
        return None

    stats = {
        "time": m.group("time"),
        "elapsed_ms": _f(m.group("elapsed")),
        "cores": {},  # cpu_id -> {mhz, pct, idle_pct, down_pct}
        "clusters": {},  # cluster name -> {mhz, pct, idle_pct, down_pct, online_pct}
        "gpu": {},  # {mhz, pct, idle_pct, power_mw}
        "power": {},  # CPU/GPU/ANE/Combined -> mW
        "io": {},  # battery/network/disk fields
        "tasks": [],  # sampled task table rows
        "irqs": {},  # cpu_id -> total interrupts/sec
    }

    section: Optional[str] = None
    irq_cpu: Optional[int] = None

    for raw in lines[1:]:
        line = raw.rstrip("\n")
        new_section = _section_name(line)
        if new_section != "__same__":
            section = new_section
            irq_cpu = None
            continue

        if not line.strip():
            continue

        if section == "tasks":
            tm = TASK_RE.match(line)
            if tm:
                name = tm.group("name").strip()
                # ALL_TASKS is useful as a summary; DEAD_TASKS is mostly noise
                # in the Top Tasks panel but remains available in CSV if needed.
                stats["tasks"].append(
                    {
                        "name": name,
                        "pid": int(tm.group("pid")),
                        "cpu_ms_s": _f(tm.group("cpu_ms")),
                        "user_pct": _f(tm.group("user_pct")),
                        "deadline_lt2_ms_s": _f(tm.group("deadline_lt2")),
                        "deadline_2_5_ms_s": _f(tm.group("deadline_2_5")),
                        "wakeups_intr_s": _f(tm.group("wake_intr")),
                        "wakeups_pkg_idle_s": _f(tm.group("wake_pkg")),
                    }
                )
            continue

        if section == "battery":
            bm = BATTERY_RE.match(line)
            if bm:
                stats["io"]["battery_percent"] = _f(bm.group("pct"))
            continue

        if section == "network":
            nm = NET_RE.match(line)
            if nm:
                prefix = f"network_{nm.group('dir')}"
                stats["io"][f"{prefix}_packets_s"] = _f(nm.group("packets"))
                stats["io"][f"{prefix}_bytes_s"] = _f(nm.group("bytes"))
            continue

        if section == "disk":
            dm = DISK_RE.match(line)
            if dm:
                prefix = f"disk_{dm.group('dir')}"
                stats["io"][f"{prefix}_ops_s"] = _f(dm.group("ops"))
                stats["io"][f"{prefix}_kbytes_s"] = _f(dm.group("kb"))
            continue

        if section == "interrupts":
            cm = IRQ_CPU_RE.match(line)
            if cm:
                irq_cpu = int(cm.group("id"))
                continue
            im = IRQ_TOTAL_RE.match(line)
            if im and irq_cpu is not None:
                stats["irqs"][irq_cpu] = _f(im.group("irq"))
            continue

        # Processor/GPU/power lines are parsed globally because powermetrics
        # repeats GPU Power in both processor and GPU sections.
        for regex, field in (
            (CPU_FREQ_RE, "mhz"),
            (CPU_ACTIVE_RE, "pct"),
            (CPU_IDLE_RE, "idle_pct"),
            (CPU_DOWN_RE, "down_pct"),
        ):
            cm = regex.match(line)
            if cm:
                cpu_id = int(cm.group("id"))
                stats["cores"].setdefault(cpu_id, {})[field] = _f(cm.group(2))
                break
        else:
            matched_cluster = False
            for regex, field in (
                (CLUSTER_ONLINE_RE, "online_pct"),
                (CLUSTER_FREQ_RE, "mhz"),
                (CLUSTER_ACTIVE_RE, "pct"),
                (CLUSTER_IDLE_RE, "idle_pct"),
                (CLUSTER_DOWN_RE, "down_pct"),
            ):
                cm = regex.match(line)
                if cm:
                    stats["clusters"].setdefault(cm.group("name"), {})[field] = _f(
                        cm.group(2)
                    )
                    matched_cluster = True
                    break
            if matched_cluster:
                continue

            gm = GPU_FREQ_RE.match(line)
            if gm:
                stats["gpu"]["mhz"] = _f(gm.group("mhz"))
                continue
            gm = GPU_ACTIVE_RE.match(line)
            if gm:
                stats["gpu"]["pct"] = _f(gm.group("pct"))
                continue
            gm = GPU_IDLE_RE.match(line)
            if gm:
                stats["gpu"]["idle_pct"] = _f(gm.group("pct"))
                continue
            pm = POWER_RE.match(line)
            if pm:
                name = pm.group("name")
                mw = _f(pm.group("mw"))
                stats["power"][name] = mw
                if name == "GPU":
                    stats["gpu"]["power_mw"] = mw
                continue
            pm = COMBINED_POWER_RE.match(line)
            if pm:
                stats["power"]["Combined"] = _f(pm.group("mw"))
                continue

    # Convert core dict to a dense list, preserving CPU IDs.
    if stats["cores"]:
        stats["cores"] = [
            {"id": cpu_id, **values}
            for cpu_id, values in sorted(stats["cores"].items())
        ]
    else:
        stats["cores"] = []

    # Drop DEAD_TASKS from the rendered top-task list, but keep ALL_TASKS so the
    # snapshot/CSV can report total CPU if it appears in the top slice.
    stats["tasks"] = sorted(
        stats["tasks"], key=lambda t: t.get("cpu_ms_s", 0.0), reverse=True
    )

    return stats


def cpu_average(stats: dict) -> float:
    vals = [c.get("pct", 0.0) for c in stats.get("cores", [])]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# CSV LOGGING (--log)
# ---------------------------------------------------------------------------
class CsvLogger:
    """Append one CSV row per parsed powermetrics sample."""

    def __init__(self, path: str):
        self.file = open(path, "w", newline="")
        self.writer: Optional[csv.DictWriter] = None

    @staticmethod
    def flatten(s: dict) -> Dict[str, object]:
        row: Dict[str, object] = {
            "timestamp": s.get("time", ""),
            "elapsed_ms": s.get("elapsed_ms", ""),
            "cpu_avg_pct": round(cpu_average(s), 2),
        }

        for core in s.get("cores", []):
            cid = core["id"]
            row[f"cpu{cid}_pct"] = core.get("pct", "")
            row[f"cpu{cid}_mhz"] = core.get("mhz", "")
            row[f"cpu{cid}_idle_pct"] = core.get("idle_pct", "")
            row[f"cpu{cid}_down_pct"] = core.get("down_pct", "")

        for name, cluster in s.get("clusters", {}).items():
            safe = re.sub(r"\W+", "_", name).strip("_").lower()
            for key in ("online_pct", "pct", "mhz", "idle_pct", "down_pct"):
                row[f"cluster_{safe}_{key}"] = cluster.get(key, "")

        gpu = s.get("gpu", {})
        row["gpu_pct"] = gpu.get("pct", "")
        row["gpu_mhz"] = gpu.get("mhz", "")
        row["gpu_idle_pct"] = gpu.get("idle_pct", "")
        row["gpu_power_mw"] = gpu.get("power_mw", "")

        for name, mw in s.get("power", {}).items():
            safe = re.sub(r"\W+", "_", name).strip("_").lower()
            row[f"power_{safe}_mw"] = mw

        for key, value in sorted(s.get("io", {}).items()):
            row[key] = value
        for cpu_id, irq in sorted(s.get("irqs", {}).items()):
            row[f"cpu{cpu_id}_irq_s"] = irq

        # Keep task logging rectangular by using task rank rather than process
        # names as columns. The top 25 is plenty for spotting load sources.
        for i, task in enumerate(s.get("tasks", [])[:25]):
            prefix = f"task{i}"
            row[f"{prefix}_name"] = task.get("name", "")
            row[f"{prefix}_pid"] = task.get("pid", "")
            row[f"{prefix}_cpu_ms_s"] = task.get("cpu_ms_s", "")
            row[f"{prefix}_user_pct"] = task.get("user_pct", "")
            row[f"{prefix}_wakeups_intr_s"] = task.get("wakeups_intr_s", "")
            row[f"{prefix}_wakeups_pkg_idle_s"] = task.get("wakeups_pkg_idle_s", "")

        return row

    def write(self, s: dict) -> None:
        row = self.flatten(s)
        if self.writer is None:
            self.writer = csv.DictWriter(
                self.file,
                fieldnames=list(row),
                restval="",
                extrasaction="ignore",
            )
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SHARED STATE + READER THREAD
# ---------------------------------------------------------------------------
class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: Optional[dict] = None
        self.cpu_hist = deque(maxlen=1024)
        self.gpu_hist = deque(maxlen=1024)
        self.power_hist = deque(maxlen=1024)
        self.eof = False


def reader_thread(
    stream: TextIO, state: State, logger: Optional[CsvLogger] = None
) -> None:
    try:
        for block in iter_sample_blocks(stream):
            s = parse_sample(block)
            if not s:
                continue
            if logger:
                logger.write(s)
            with state.lock:
                state.latest = s
                state.cpu_hist.append(cpu_average(s))
                state.gpu_hist.append(s.get("gpu", {}).get("pct", 0.0))
                state.power_hist.append(s.get("power", {}).get("Combined", 0.0))
    except (OSError, ValueError):
        pass
    finally:
        state.eof = True


# ---------------------------------------------------------------------------
# CURSES DRAWING PRIMITIVES
# ---------------------------------------------------------------------------
GREEN, YELLOW, RED, TITLE, DIMP = 1, 2, 3, 4, 5
BLOCKS = " ▁▂▃▄▅▆▇█"


def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(GREEN, curses.COLOR_GREEN, -1)
    curses.init_pair(YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(RED, curses.COLOR_RED, -1)
    curses.init_pair(TITLE, curses.COLOR_CYAN, -1)
    curses.init_pair(DIMP, curses.COLOR_WHITE, -1)


def load_attr(pct: float):
    if pct < 50:
        return curses.color_pair(GREEN)
    if pct < 80:
        return curses.color_pair(YELLOW)
    return curses.color_pair(RED)


def put(scr, y: int, x: int, text: str, attr=0, xmax: Optional[int] = None) -> None:
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


def draw_gauge(scr, y: int, x: int, width: int, pct: float, attr=None) -> None:
    width = max(0, width)
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100 * width))
    a = attr if attr is not None else load_attr(pct)
    put(scr, y, x, "█" * filled, a)
    put(scr, y, x + filled, "░" * (width - filled), curses.A_DIM)


def draw_graph(
    scr, y: int, x: int, h: int, w: int, hist: Iterable[float], scale: float = 100.0
) -> None:
    if h < 1 or w < 1:
        return
    vals = list(hist)[-w:]
    scale = max(1.0, scale)
    for col, v in enumerate(vals):
        cx = x + w - len(vals) + col
        pct = max(0.0, min(100.0, v / scale * 100.0))
        eighths = pct / 100.0 * h * 8
        full, part = int(eighths // 8), int(eighths % 8)
        a = load_attr(pct)
        for r in range(full):
            put(scr, y + h - 1 - r, cx, "█", a)
        if part and full < h:
            put(scr, y + h - 1 - full, cx, BLOCKS[part], a)


def fmt_bytes_per_s(bytes_s: float) -> str:
    units = ["B/s", "KB/s", "MB/s", "GB/s"]
    value = float(bytes_s)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    return f"{value:.1f} {units[unit]}"


# ---------------------------------------------------------------------------
# PANELS
# ---------------------------------------------------------------------------
def panel_cpu(scr, y: int, x: int, h: int, w: int, s: dict, hist) -> None:
    draw_box(scr, y, x, h, w, "CPU")
    cores = s.get("cores", [])
    iy, ix, iw = y + 1, x + 2, w - 4
    rows_avail = h - 2
    if not cores:
        put(scr, iy, ix, "No CPU residency data found yet", curses.A_DIM)
        return

    ncols = 1 if len(cores) + 2 <= rows_avail - 3 else 2
    percol = max(1, -(-len(cores) // ncols))
    col_w = max(10, iw // ncols)
    bw = max(5, col_w - 23)

    for i, core in enumerate(cores):
        ry = iy + i % percol
        rx = ix + (i // percol) * col_w
        if ry >= y + h - 4:
            continue
        pct = core.get("pct", 0.0)
        mhz = core.get("mhz")
        put(scr, ry, rx, f"cpu{core['id']:<2}", curses.A_BOLD)
        draw_gauge(scr, ry, rx + 7, bw, pct)
        put(scr, ry, rx + 7 + bw, f" {pct:5.1f}%", xmax=x + w - 1)
        if mhz is not None:
            put(
                scr,
                ry,
                rx + 7 + bw + 8,
                f" @{mhz:.0f}MHz",
                curses.A_DIM,
                xmax=x + w - 1,
            )

    avg = cpu_average(s)
    ay = min(iy + percol, y + h - 3)
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

    gy = ay + 2
    gh = y + h - 1 - gy
    if gh >= 2:
        put(
            scr,
            gy - 1,
            ix,
            "active residency history",
            curses.color_pair(DIMP) | curses.A_DIM,
        )
        draw_graph(scr, gy, ix, gh, iw, hist)


def panel_gpu(scr, y: int, x: int, h: int, w: int, s: dict, hist) -> None:
    draw_box(scr, y, x, h, w, "GPU")
    iy, ix, iw = y + 1, x + 2, w - 4
    gpu = s.get("gpu", {})
    pct = gpu.get("pct", 0.0)
    mhz = gpu.get("mhz")
    power = gpu.get("power_mw")
    bw = max(5, iw - 28)
    put(scr, iy, ix, "active", curses.A_BOLD)
    draw_gauge(scr, iy, ix + 8, bw, pct)
    put(scr, iy, ix + 8 + bw, f" {pct:5.1f}%", xmax=x + w - 1)
    pieces = []
    if mhz is not None:
        pieces.append(f"{mhz:.0f} MHz")
    if power is not None:
        pieces.append(f"{power:.0f} mW")
    if pieces:
        put(scr, iy + 1, ix + 8, " · ".join(pieces), curses.A_DIM, xmax=x + w - 1)
    gh = y + h - 1 - (iy + 3)
    if gh >= 1:
        draw_graph(scr, iy + 3, ix, gh, iw, hist)


def panel_power(scr, y: int, x: int, h: int, w: int, s: dict) -> None:
    draw_box(scr, y, x, h, w, "Power")
    iy, ix, iw = y + 1, x + 2, w - 4
    order = ["CPU", "GPU", "ANE", "Combined"]
    power = s.get("power", {})
    values = [(name, power[name]) for name in order if name in power]
    if not values:
        put(scr, iy, ix, "No power data found yet", curses.A_DIM)
        return
    peak = max((mw for _, mw in values), default=1.0)
    bw = max(5, iw - 22)
    for i, (name, mw) in enumerate(values):
        if iy + i >= y + h - 1:
            break
        put(scr, iy + i, ix, f"{name:<8}", curses.A_BOLD)
        pct = mw / peak * 100 if peak else 0
        draw_gauge(scr, iy + i, ix + 9, bw, pct, attr=load_attr(pct))
        put(scr, iy + i, ix + 9 + bw, f" {mw:7.0f} mW", xmax=x + w - 1)


def panel_io(scr, y: int, x: int, h: int, w: int, s: dict) -> None:
    draw_box(scr, y, x, h, w, "I/O + Status")
    iy, ix, iw = y + 1, x + 2, w - 4
    io = s.get("io", {})
    row = iy
    if "battery_percent" in io:
        pct = io["battery_percent"]
        put(scr, row, ix, "Battery", curses.A_BOLD)
        draw_gauge(scr, row, ix + 9, max(5, iw - 20), pct)
        put(scr, row, ix + iw - 8, f"{pct:.0f}%", xmax=x + w - 1)
        row += 1
    put(
        scr,
        row,
        ix,
        f"Sample  {s.get('elapsed_ms', 0):.0f} ms",
        curses.A_DIM,
        xmax=x + w - 1,
    )
    row += 1
    if row < y + h - 1 and "network_in_bytes_s" in io:
        put(scr, row, ix, "Net in ", curses.A_BOLD)
        put(
            scr,
            row,
            ix + 8,
            fmt_bytes_per_s(io.get("network_in_bytes_s", 0)),
            xmax=x + w - 1,
        )
        row += 1
    if row < y + h - 1 and "network_out_bytes_s" in io:
        put(scr, row, ix, "Net out", curses.A_BOLD)
        put(
            scr,
            row,
            ix + 8,
            fmt_bytes_per_s(io.get("network_out_bytes_s", 0)),
            xmax=x + w - 1,
        )
        row += 1
    if row < y + h - 1 and "disk_read_kbytes_s" in io:
        put(scr, row, ix, "Disk rd", curses.A_BOLD)
        put(
            scr,
            row,
            ix + 8,
            fmt_bytes_per_s(io.get("disk_read_kbytes_s", 0) * 1024),
            xmax=x + w - 1,
        )
        row += 1
    if row < y + h - 1 and "disk_write_kbytes_s" in io:
        put(scr, row, ix, "Disk wr", curses.A_BOLD)
        put(
            scr,
            row,
            ix + 8,
            fmt_bytes_per_s(io.get("disk_write_kbytes_s", 0) * 1024),
            xmax=x + w - 1,
        )


def panel_tasks(scr, y: int, x: int, h: int, w: int, s: dict) -> None:
    draw_box(scr, y, x, h, w, "Top Tasks")
    iy, ix, iw = y + 1, x + 2, w - 4
    tasks = [
        t
        for t in s.get("tasks", [])
        if t.get("name") not in {"DEAD_TASKS", "ALL_TASKS"}
    ]
    if not tasks:
        put(scr, iy, ix, "No task table found yet", curses.A_DIM)
        return
    put(scr, iy, ix, "Name", curses.A_DIM)
    put(scr, iy, max(ix, x + w - 26), "CPU ms/s  Wake/s", curses.A_DIM, xmax=x + w - 1)
    for i, task in enumerate(tasks[: max(0, h - 4)]):
        row = iy + 1 + i
        name_w = max(8, iw - 22)
        put(
            scr,
            row,
            ix,
            task.get("name", "")[:name_w],
            curses.A_BOLD if i == 0 else 0,
            xmax=x + w - 24,
        )
        put(
            scr,
            row,
            x + w - 23,
            f"{task.get('cpu_ms_s', 0):8.1f}",
            load_attr(min(100, task.get("cpu_ms_s", 0) / 10)),
            xmax=x + w - 1,
        )
        put(
            scr,
            row,
            x + w - 13,
            f"{task.get('wakeups_intr_s', 0):7.1f}",
            curses.A_DIM,
            xmax=x + w - 1,
        )


def panel_clusters(scr, y: int, x: int, h: int, w: int, s: dict) -> None:
    draw_box(scr, y, x, h, w, "Clusters")
    iy, ix, iw = y + 1, x + 2, w - 4
    clusters = s.get("clusters", {})
    if not clusters:
        put(scr, iy, ix, "No cluster data found yet", curses.A_DIM)
        return
    bw = max(5, iw - 24)
    for i, (name, cluster) in enumerate(clusters.items()):
        if iy + i >= y + h - 1:
            break
        pct = cluster.get("pct", 0.0)
        mhz = cluster.get("mhz")
        label = name.replace("-Cluster", "")[:6]
        put(scr, iy + i, ix, f"{label:<6}", curses.A_BOLD)
        draw_gauge(scr, iy + i, ix + 7, bw, pct)
        suffix = f" {pct:4.0f}%"
        if mhz is not None:
            suffix += f" @{mhz:.0f}"
        put(scr, iy + i, ix + 7 + bw, suffix, curses.A_DIM, xmax=x + w - 1)


# ---------------------------------------------------------------------------
# FRAME + TUI LOOP
# ---------------------------------------------------------------------------
def draw_frame(scr, state: State, source_label: str) -> None:
    scr.erase()
    H, W = scr.getmaxyx()
    with state.lock:
        s = state.latest
        cpu_hist = list(state.cpu_hist)
        gpu_hist = list(state.gpu_hist)

    if s is None:
        put(
            scr,
            H // 2,
            max(0, W // 2 - 14),
            "waiting for powermetrics...",
            curses.A_DIM,
        )
        scr.refresh()
        return

    if H < 16 or W < 78:
        put(
            scr,
            0,
            0,
            f"terminal too small ({W}x{H}), need at least 78x16",
            curses.color_pair(RED),
        )
        scr.refresh()
        return

    body_h = H - 1
    left_w = min(max(52, int(W * 0.58)), W - 34)
    right_w = W - left_w
    rx = left_w

    gpu_h = max(6, body_h // 4)
    cpu_h = body_h - gpu_h
    panel_cpu(scr, 0, 0, cpu_h, left_w, s, cpu_hist)
    panel_gpu(scr, cpu_h, 0, gpu_h, left_w, s, gpu_hist)

    power_h = min(7, max(5, body_h // 5))
    io_h = min(8, max(6, body_h // 4))
    cluster_h = min(len(s.get("clusters", {})) + 2, max(3, body_h // 5))
    tasks_h = body_h - power_h - io_h - cluster_h
    if tasks_h < 5:
        tasks_h = 5
        cluster_h = max(3, body_h - power_h - io_h - tasks_h)

    panel_power(scr, 0, rx, power_h, right_w, s)
    panel_io(scr, power_h, rx, io_h, right_w, s)
    panel_clusters(scr, power_h + io_h, rx, cluster_h, right_w, s)
    panel_tasks(scr, power_h + io_h + cluster_h, rx, tasks_h, right_w, s)

    put(scr, H - 1, 1, " q quit ", curses.A_REVERSE)
    tail = f"{source_label}  {s.get('time', '')} "
    put(scr, H - 1, max(10, W - len(tail) - 1), tail, curses.A_DIM)
    scr.refresh()


def tui(scr, state: State, source_label: str) -> None:
    curses.curs_set(0)
    init_colors()
    scr.timeout(200)
    while True:
        ch = scr.getch()
        if ch in (ord("q"), ord("Q"), 27):
            break
        draw_frame(scr, state, source_label)
        if state.eof:
            break


# ---------------------------------------------------------------------------
# INPUT ACQUISITION + ENTRY POINT
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="btop-style TUI and CSV exporter for Apple's powermetrics."
    )
    p.add_argument(
        "logfile",
        nargs="?",
        help="replay a saved powermetrics text log instead of live data",
    )
    p.add_argument(
        "--log",
        metavar="FILE.csv",
        dest="csv_path",
        help="export every parsed sample to this CSV file",
    )
    p.add_argument(
        "--powermetrics-args",
        default="",
        help="extra arguments passed to powermetrics, e.g. '-i 1000'",
    )
    p.add_argument(
        "--no-sudo",
        action="store_true",
        help="run powermetrics without sudo; useful only if already root or using a wrapper",
    )
    return p.parse_args()


def sudo_prefix(args: argparse.Namespace) -> List[str]:
    if args.no_sudo or getattr(os, "geteuid", lambda: 0)() == 0:
        return []
    if sys.stdin.isatty() and sys.stdout.isatty():
        print("powermetrics requires sudo. Authenticating before opening the TUI...")
        try:
            subprocess.run(["sudo", "-v"], check=True)
        except subprocess.CalledProcessError:
            sys.exit("sudo authentication failed; cannot run powermetrics")
        return ["sudo", "-n"]
    return ["sudo"]


def acquire_stream(
    args: argparse.Namespace,
) -> Tuple[TextIO, str, Optional[subprocess.Popen[str]]]:
    if args.logfile:
        return open(args.logfile, errors="replace"), f"file:{args.logfile}", None

    if not sys.stdin.isatty():
        data = os.fdopen(os.dup(0), errors="replace")
        try:
            tty = open("/dev/tty")
            os.dup2(tty.fileno(), 0)
        except OSError:
            pass
        return data, "stdin", None

    extra = shlex.split(args.powermetrics_args)
    cmd = sudo_prefix(args) + ["powermetrics"] + extra
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError:
        sys.exit("powermetrics not found. This script needs to run on macOS.")
    if proc.stdout is None:
        sys.exit("failed to capture powermetrics output")
    return (
        cast(TextIO, proc.stdout),
        "sudo powermetrics" if cmd[:1] == ["sudo"] else "powermetrics",
        proc,
    )


def snapshot(stream: TextIO, logger: Optional[CsvLogger] = None) -> None:
    last = None
    for block in iter_sample_blocks(stream):
        s = parse_sample(block)
        if s:
            if logger:
                logger.write(s)
            last = s
    if not last:
        sys.exit("no powermetrics sample data found in input")

    print(f"powermetrics snapshot  {last.get('time', '')}")
    print(f"  sample: {last.get('elapsed_ms', 0):.0f} ms")
    print(f"  cpu avg active: {cpu_average(last):.1f}%")
    for core in last.get("cores", []):
        pct = core.get("pct", 0.0)
        mhz = core.get("mhz")
        mhz_text = f" @ {mhz:.0f} MHz" if mhz is not None else ""
        print(f"  cpu{core['id']}: {pct:5.1f}%{mhz_text}")
    gpu = last.get("gpu", {})
    if gpu:
        print(
            "  gpu: "
            f"{gpu.get('pct', 0):.1f}%"
            + (f" @ {gpu['mhz']:.0f} MHz" if "mhz" in gpu else "")
            + (f", {gpu['power_mw']:.0f} mW" if "power_mw" in gpu else "")
        )
    if last.get("power"):
        print(
            "  power: " + ", ".join(f"{k} {v:.0f} mW" for k, v in last["power"].items())
        )
    io = last.get("io", {})
    if "battery_percent" in io:
        print(f"  battery: {io['battery_percent']:.0f}%")
    if "network_in_bytes_s" in io or "network_out_bytes_s" in io:
        print(
            "  network: "
            f"in {fmt_bytes_per_s(io.get('network_in_bytes_s', 0))}, "
            f"out {fmt_bytes_per_s(io.get('network_out_bytes_s', 0))}"
        )
    if "disk_read_kbytes_s" in io or "disk_write_kbytes_s" in io:
        print(
            "  disk: "
            f"read {fmt_bytes_per_s(io.get('disk_read_kbytes_s', 0) * 1024)}, "
            f"write {fmt_bytes_per_s(io.get('disk_write_kbytes_s', 0) * 1024)}"
        )
    top_tasks = [
        t
        for t in last.get("tasks", [])
        if t.get("name") not in {"DEAD_TASKS", "ALL_TASKS"}
    ]
    if top_tasks:
        print("  top tasks:")
        for task in top_tasks[:8]:
            print(
                f"    {task['name']}: {task['cpu_ms_s']:.1f} CPU ms/s, {task['wakeups_intr_s']:.1f} wakeups/s"
            )


def main() -> None:
    args = parse_args()
    stream, label, proc = acquire_stream(args)
    logger = CsvLogger(args.csv_path) if args.csv_path else None

    if label.startswith("file:") or not sys.stdout.isatty():
        try:
            snapshot(stream, logger)
        finally:
            if logger:
                logger.close()
            if stream is not sys.stdin:
                stream.close()
        return

    state = State()
    t = threading.Thread(
        target=reader_thread, args=(stream, state, logger), daemon=True
    )
    t.start()

    try:
        curses.wrapper(tui, state, label)
    except KeyboardInterrupt:
        pass
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            stream.close()
        except OSError:
            pass
        if logger:
            t.join(timeout=1)
            logger.close()


if __name__ == "__main__":
    main()
