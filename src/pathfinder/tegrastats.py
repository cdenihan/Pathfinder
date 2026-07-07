#!/usr/bin/env python3
"""
================================================================================
 tegrastats_viewer.py — a btop-style full-screen TUI for NVIDIA Jetson
                        `tegrastats`
================================================================================

WHAT IT IS
----------
`tegrastats` (shipped with NVIDIA JetPack on Jetson boards) prints one long,
hard-to-read line per sample, e.g.:

    07-07-2026 15:20:47 RAM 1812/7621MB (lfb 42x4MB) SWAP 0/3810MB (cached 0MB)
    CPU [21%@729,28%@729,...] GR3D_FREQ 0% cpu@45.25C ... VDD_IN 3560mW/3806mW ...

This script turns that stream into a full-screen curses dashboard that expands
to fill the whole terminal (and re-flows when you resize it), with bordered
panels for:

    * CPU    — one gauge per core (load % + clock MHz, auto 1/2 columns),
               an average row, and a scrolling load-history graph
    * GPU    — GR3D load gauge + scrolling history graph
    * Memory — RAM and SWAP gauges (largest-free-block / swap-cache info)
    * Temps  — every temperature sensor found on the line, with mini gauges
    * Power  — every VDD_* rail: current draw, tegrastats' running average

USAGE
-----
Run it directly (it spawns `tegrastats` itself — easiest):

    python3 tegrastats_viewer.py                # or sudo if your board needs it

Or pipe tegrastats in (keyboard still works — keys are read from /dev/tty):

    sudo tegrastats | python3 tegrastats_viewer.py
    sudo tegrastats --interval 500 | ./tegrastats_viewer.py

Replay a saved log (no TUI, prints a plain-text snapshot of the last sample):

    python3 tegrastats_viewer.py logfile.txt

Export every sample to CSV while the dashboard runs (works in every mode,
including file replay — see CSV LOGGING below):

    python3 tegrastats_viewer.py --log stats.csv
    sudo tegrastats | python3 tegrastats_viewer.py --log stats.csv
    python3 tegrastats_viewer.py logfile.txt --log stats.csv   # convert log->CSV

Keys:  q / Esc  quit        (Ctrl+C also works)

CSV LOGGING (--log)
-------------------
With `--log FILE.csv`, EVERY field parsed from EVERY tegrastats sample is
appended as one CSV row — nothing is dropped. The columns are derived from
the first sample seen (tegrastats emits the same fields every line within a
run, so the set is stable) and are, in order:

    timestamp                       "07-07-2026 15:20:47" as printed
    cpu0_pct,  cpu0_mhz, ...        load % and clock per core
                                    (both empty when the core is offline)
    cpu_avg_pct                     average load over the online cores
    ram_used_mb, ram_total_mb,      RAM usage
    ram_lfb                         largest free block, e.g. "42x4MB"
    swap_used_mb, swap_total_mb,    SWAP usage
    swap_cached_mb                  swap cache
    gpu_pct, gpu_mhz                GR3D load (+ frequency when reported)
    temp_<sensor>_c                 one column per sensor (cpu, gpu, tj, ...)
    <rail>_mw, <rail>_avg_mw        current + average draw per VDD_* rail

Rows are flushed to disk after every sample, so the file is complete even if
the run ends with Ctrl+C or tegrastats dies. The file is opened in write
mode (existing contents are replaced). Logging is handled by CsvLogger and
happens on the reader thread, in parallel with the TUI.

No third-party dependencies — Python 3 stdlib only (curses ships with it).

HOW IT WORKS (high-level flow)
------------------------------
1. INPUT   — acquire_stream() picks the data source: a file argument, the
             stdin pipe, or a `tegrastats` subprocess it spawns itself when
             run interactively. When stdin is the data pipe, the real
             terminal (/dev/tty) is dup2()'d onto fd 0 so curses can still
             read the keyboard while the pipe is drained separately.
2. PARSE   — a background reader thread matches each incoming line against
             a set of regexes (LINE_RE / CORE_RE), one per tegrastats field
             (RAM, SWAP, CPU, GR3D_FREQ, temps, power rails), producing a
             plain dict. It also appends CPU-average / GPU load to bounded
             history deques that feed the graphs, and — when --log is given —
             hands each sample to CsvLogger. Non-matching lines are silently
             skipped.
3. DRAW    — the main thread runs the curses loop: every 200 ms it polls
             the keyboard and redraws all panels from the latest sample.
             Layout is computed fresh from the current terminal size each
             frame, so resizing (KEY_RESIZE) "just works". Gauges are
             Unicode block bars colored green/yellow/red (<50 / <80 / >=80 %
             load; <60 / <80 / >=80 °C for temperatures); graphs use the
             ▁▂▃▄▅▆▇█ partial-block ramp, one column per sample.

FIELD NOTES / EDGE CASES HANDLED
--------------------------------
* Offline cores: tegrastats prints "off" for parked cores — shown as a
  dimmed "offline" row and excluded from the CPU average.
* GPU frequency: some Jetsons report "GR3D_FREQ 15%@612", others just
  "GR3D_FREQ 15%" — both forms parse; the MHz is shown only if present.
* Sensor names aren't hardcoded: any "name@NN.NC" temperature and any
  "VDD_xxx cur/avg mW" rail on the line is picked up automatically, so the
  script works across Jetson models (Nano, Orin, Xavier, ...).
* Tiny terminals: below a minimum usable size the TUI shows a "resize me"
  notice instead of crashing; every draw call is clipped to the screen.
* If the input stream ends (tegrastats dies, pipe closes), the TUI exits
  cleanly after the final frame instead of hanging.
================================================================================
"""

import argparse
import csv
import curses
import os
import re
import subprocess
import sys
import threading
from collections import deque

# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------
# One regex per tegrastats field. Each is search()ed independently against the
# whole line, so field order doesn't matter and missing fields (which vary by
# Jetson model / L4T version) simply leave that key out of the stats dict.
LINE_RE = {
    # "07-07-2026 15:20:47" — timestamp at the start of the line
    "time":  re.compile(r"^(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})"),
    # "RAM 1812/7621MB (lfb 42x4MB)" -> used MB, total MB, largest free block
    "ram":   re.compile(r"RAM (\d+)/(\d+)MB \(lfb (\S+)\)"),
    # "SWAP 0/3810MB (cached 0MB)" -> used MB, total MB, cached MB
    "swap":  re.compile(r"SWAP (\d+)/(\d+)MB \(cached (\d+)MB\)"),
    # "CPU [21%@729,28%@729,off,...]" -> raw comma-separated core list,
    # individual cores are split out with CORE_RE below
    "cpu":   re.compile(r"CPU \[([^\]]+)\]"),
    # "GR3D_FREQ 0%" or "GR3D_FREQ 15%@612" -> load %, optional MHz
    "gpu":   re.compile(r"GR3D_FREQ (\d+)%(?:@(\d+))?"),
    # "cpu@45.25C", "tj@45.625C", ... -> findall() grabs every sensor
    "temp":  re.compile(r"(\w+)@([\d.]+)C"),
    # "VDD_IN 3560mW/3806mW" -> rail name, current mW, average mW
    "power": re.compile(r"(VDD_\w+) (\d+)mW/(\d+)mW"),
}
# One entry inside the CPU [...] list: either "21%@729" (load%@MHz)
# or the literal "off" for a parked/offline core.
CORE_RE = re.compile(r"(\d+)%@(\d+)|off")


def parse_line(line):
    """Parse one tegrastats line into a stats dict, or None if it doesn't match.

    Returned dict shape (keys absent if that field wasn't on the line):
        time  : "07-07-2026 15:20:47"
        cores : [(load_pct, freq_mhz) or None, ...]   None = core offline
        ram   : (used_mb, total_mb, lfb_str)
        swap  : (used_mb, total_mb, cached_mb)
        gpu   : (load_pct, freq_mhz_or_None)
        temps : {sensor_name: degrees_c, ...}
        power : {rail_name: (current_mw, average_mw), ...}
    """
    # The CPU block is mandatory in tegrastats output, so use it as the
    # "is this actually a tegrastats line?" gate — anything without it
    # (blank lines, shell noise, etc.) is rejected here.
    m = LINE_RE["cpu"].search(line)
    if not m:
        return None

    stats = {"cores": []}
    for part in m.group(1).split(","):
        cm = CORE_RE.match(part.strip())
        if cm and cm.group(1) is not None:
            stats["cores"].append((int(cm.group(1)), int(cm.group(2))))
        else:
            stats["cores"].append(None)  # core reported as "off"

    m = LINE_RE["time"].search(line)
    stats["time"] = m.group(1) if m else ""

    m = LINE_RE["ram"].search(line)
    if m:
        stats["ram"] = (int(m.group(1)), int(m.group(2)), m.group(3))

    m = LINE_RE["swap"].search(line)
    if m:
        stats["swap"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = LINE_RE["gpu"].search(line)
    if m:
        stats["gpu"] = (int(m.group(1)), int(m.group(2)) if m.group(2) else None)

    # findall() sweeps the whole line so every sensor/rail present is captured,
    # regardless of which Jetson model produced the output.
    stats["temps"] = {name: float(v) for name, v in LINE_RE["temp"].findall(line)}
    stats["power"] = {name: (int(cur), int(avg))
                      for name, cur, avg in LINE_RE["power"].findall(line)}
    return stats


# ---------------------------------------------------------------------------
# CSV LOGGING (--log)
# ---------------------------------------------------------------------------
class CsvLogger:
    """Appends one CSV row per tegrastats sample; see CSV LOGGING in the README.

    Columns are locked in from the FIRST sample (flatten() of it defines the
    header), because tegrastats emits an identical field set on every line of
    a run. Should a later sample somehow carry an extra field it is ignored
    (extrasaction="ignore") and a missing one becomes an empty cell
    (restval="") — the file stays rectangular either way. Every row is
    flushed immediately so the CSV is intact even after Ctrl+C or a crash.
    """

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = None                    # created on the first sample

    @staticmethod
    def flatten(s):
        """Turn one parsed stats dict into a flat {column: value} row.

        Mirrors parse_line()'s output one-to-one so no tegrastats data is
        lost: every core, both RAM/SWAP triplets, GPU, every temperature
        sensor and every power rail get their own columns.
        """
        row = {"timestamp": s.get("time", "")}
        online = []
        for i, core in enumerate(s["cores"]):
            if core is None:                  # offline core -> empty cells
                row[f"cpu{i}_pct"] = row[f"cpu{i}_mhz"] = ""
            else:
                row[f"cpu{i}_pct"], row[f"cpu{i}_mhz"] = core
                online.append(core[0])
        row["cpu_avg_pct"] = round(sum(online) / max(1, len(online)), 2)
        if "ram" in s:
            row["ram_used_mb"], row["ram_total_mb"], row["ram_lfb"] = s["ram"]
        if "swap" in s:
            (row["swap_used_mb"], row["swap_total_mb"],
             row["swap_cached_mb"]) = s["swap"]
        if "gpu" in s:
            pct, freq = s["gpu"]
            row["gpu_pct"] = pct
            row["gpu_mhz"] = freq if freq is not None else ""
        for name, c in s.get("temps", {}).items():
            row[f"temp_{name}_c"] = c
        for name, (cur, avg) in s.get("power", {}).items():
            row[f"{name}_mw"] = cur
            row[f"{name}_avg_mw"] = avg
        return row

    def write(self, s):
        row = self.flatten(s)
        if self.writer is None:               # first sample defines the header
            self.writer = csv.DictWriter(self.file, fieldnames=list(row),
                                         restval="", extrasaction="ignore")
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()                     # survive Ctrl+C / crashes

    def close(self):
        try:
            self.file.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SHARED STATE + READER THREAD
# ---------------------------------------------------------------------------
class State:
    """Latest sample + bounded history, shared between reader and UI thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None                    # most recent parsed stats dict
        self.cpu_hist = deque(maxlen=1024)    # CPU average %, one per sample
        self.gpu_hist = deque(maxlen=1024)    # GPU load %, one per sample
        self.eof = False                      # input stream finished/died


def reader_thread(stream, state, logger=None):
    """Drain the tegrastats stream, updating shared state per parsed line.

    CSV logging lives here (not in the UI loop) so every sample is recorded
    even if the UI is busy or drawing slower than data arrives.
    """
    try:
        for line in stream:
            s = parse_line(line)
            if not s:
                continue
            if logger:
                logger.write(s)
            online = [c[0] for c in s["cores"] if c]
            with state.lock:
                state.latest = s
                state.cpu_hist.append(sum(online) / max(1, len(online)))
                if "gpu" in s:
                    state.gpu_hist.append(s["gpu"][0])
    except (OSError, ValueError):
        pass  # stream closed under us during shutdown — that's fine
    state.eof = True


# ---------------------------------------------------------------------------
# CURSES DRAWING PRIMITIVES
# ---------------------------------------------------------------------------
# Color pair ids, assigned in init_colors(). Load/temperature levels reuse the
# same green/yellow/red scheme as before; TITLE is for panel headers.
GREEN, YELLOW, RED, TITLE, DIMP = 1, 2, 3, 4, 5

# Partial-block ramp used by the history graphs (index = eighths filled).
BLOCKS = " ▁▂▃▄▅▆▇█"


def init_colors():
    curses.start_color()
    curses.use_default_colors()               # -1 = terminal default background
    curses.init_pair(GREEN, curses.COLOR_GREEN, -1)
    curses.init_pair(YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(RED, curses.COLOR_RED, -1)
    curses.init_pair(TITLE, curses.COLOR_CYAN, -1)
    curses.init_pair(DIMP, curses.COLOR_WHITE, -1)


def load_attr(pct):
    """Load color thresholds: green under 50%, yellow under 80%, else red."""
    if pct < 50:
        return curses.color_pair(GREEN)
    if pct < 80:
        return curses.color_pair(YELLOW)
    return curses.color_pair(RED)


def temp_attr(c):
    """Temperature thresholds: green under 60°C, yellow under 80°C, else red.

    80°C is comfortably below Jetson throttle points (~90-97°C depending on
    model), so red here means "hot" not "already throttling".
    """
    if c < 60:
        return curses.color_pair(GREEN)
    if c < 80:
        return curses.color_pair(YELLOW)
    return curses.color_pair(RED)


def put(scr, y, x, text, attr=0, xmax=None):
    """addstr clipped to the screen; silently drops anything out of bounds.

    curses raises an error when writing outside the window (including the
    quirky bottom-right cell), so every draw goes through this guard.
    xmax (exclusive) additionally clips to a panel's right border so long
    text can never overwrite the box edge in narrow terminals.
    """
    h, w = scr.getmaxyx()
    limit = w if xmax is None else min(w, xmax)
    if not (0 <= y < h) or x >= limit:
        return
    if x < 0:
        text, x = text[-x:], 0
    try:
        scr.addstr(y, x, text[: limit - x], attr)
    except curses.error:
        pass


def draw_box(scr, y, x, h, w, title=""):
    """Draw a bordered panel with an optional title in the top edge."""
    if h < 2 or w < 2:
        return
    put(scr, y, x, "┌" + "─" * (w - 2) + "┐")
    for r in range(1, h - 1):
        put(scr, y + r, x, "│")
        put(scr, y + r, x + w - 1, "│")
    put(scr, y + h - 1, x, "└" + "─" * (w - 2) + "┘")
    if title:
        put(scr, y, x + 2, f" {title} ", curses.color_pair(TITLE) | curses.A_BOLD)


def draw_gauge(scr, y, x, width, pct, attr=None):
    """One horizontal bar: '████░░░░' colored by load (or a caller attr)."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100 * width))
    a = attr if attr is not None else load_attr(pct)
    put(scr, y, x, "█" * filled, a)
    put(scr, y, x + filled, "░" * (width - filled), curses.A_DIM)


def draw_graph(scr, y, x, h, w, hist):
    """Scrolling history graph, newest sample at the right edge.

    Each column is one sample; a value fills its column bottom-up with full
    blocks plus one partial block (from BLOCKS) for the fractional remainder,
    giving 8x vertical resolution per character cell.
    """
    if h < 1 or w < 1:
        return
    vals = list(hist)[-w:]
    for col, v in enumerate(vals):
        cx = x + w - len(vals) + col          # right-align: newest at right
        eighths = max(0, min(100, v)) / 100 * h * 8
        full, part = int(eighths // 8), int(eighths % 8)
        a = load_attr(v)
        for r in range(full):
            put(scr, y + h - 1 - r, cx, "█", a)
        if part and full < h:
            put(scr, y + h - 1 - full, cx, BLOCKS[part], a)


# ---------------------------------------------------------------------------
# PANELS
# ---------------------------------------------------------------------------
# Each panel gets a rectangle (y, x, h, w) and draws its box + contents inside
# it. The layout in draw_frame() decides the rectangles from the screen size.

def panel_cpu(scr, y, x, h, w, s, hist):
    draw_box(scr, y, x, h, w, "CPU")
    cores = s["cores"]
    iy, ix, iw = y + 1, x + 2, w - 4          # interior with 1-col padding
    rows_avail = h - 2

    # Fit the core gauges: 1 column normally, 2 columns when there are more
    # cores than rows (leaving at least 3 rows for the history graph).
    ncols = 1 if len(cores) + 1 <= rows_avail - 3 else 2
    percol = -(-len(cores) // ncols)          # ceil division
    col_w = iw // ncols
    # label "coreN " + " 100%" + " @9999MHz" ≈ 22 chars of non-bar text
    bw = max(5, col_w - 22)

    for i, core in enumerate(cores):
        ry = iy + i % percol
        rx = ix + (i // percol) * col_w
        put(scr, ry, rx, f"core{i:<2}", curses.A_BOLD)
        if core is None:
            put(scr, ry, rx + 7, "─" * bw + "  offline", curses.A_DIM)
        else:
            pct, freq = core
            draw_gauge(scr, ry, rx + 7, bw, pct)
            put(scr, ry, rx + 7 + bw, f" {pct:3d}%", xmax=x + w - 1)
            put(scr, ry, rx + 7 + bw + 5, f" @{freq}MHz", curses.A_DIM,
                xmax=x + w - 1)

    online = [c[0] for c in cores if c]
    avg = sum(online) / max(1, len(online))
    ay = iy + percol
    put(scr, ay, ix, "avg   ", curses.A_BOLD)
    draw_gauge(scr, ay, ix + 7, bw, avg)
    put(scr, ay, ix + 7 + bw, f" {avg:3.0f}%", curses.A_BOLD)

    # Whatever vertical space is left under the gauges becomes the graph.
    gy = ay + 2
    gh = y + h - 1 - gy
    if gh >= 2:
        put(scr, gy - 1, ix, "history", curses.color_pair(DIMP) | curses.A_DIM)
        draw_graph(scr, gy, ix, gh, iw, hist)


def panel_gpu(scr, y, x, h, w, s, hist):
    draw_box(scr, y, x, h, w, "GPU")
    iy, ix, iw = y + 1, x + 2, w - 4
    pct, freq = s.get("gpu", (0, None))
    bw = max(5, iw - 22)
    put(scr, iy, ix, "GR3D  ", curses.A_BOLD)
    draw_gauge(scr, iy, ix + 7, bw, pct)
    put(scr, iy, ix + 7 + bw, f" {pct:3d}%", xmax=x + w - 1)
    if freq:
        put(scr, iy, ix + 7 + bw + 5, f" @{freq}MHz", curses.A_DIM,
            xmax=x + w - 1)
    gh = y + h - 1 - (iy + 1)
    if gh >= 1:
        draw_graph(scr, iy + 1, ix, gh, iw, hist)


def panel_mem(scr, y, x, h, w, s):
    draw_box(scr, y, x, h, w, "Memory")
    iy, ix, iw = y + 1, x + 2, w - 4
    bw = max(5, iw - 7)
    row = iy
    if "ram" in s:
        used, total, lfb = s["ram"]
        pct = used / total * 100
        put(scr, row, ix, "RAM   ", curses.A_BOLD)
        draw_gauge(scr, row, ix + 6, bw, pct)
        put(scr, row + 1, ix + 6,
            f"{used}/{total} MB ({pct:.0f}%)   lfb {lfb}", curses.A_DIM,
            xmax=x + w - 1)
        row += 2
    if "swap" in s:
        used, total, cached = s["swap"]
        pct = used / total * 100 if total else 0   # guard: swap may be 0 MB
        put(scr, row, ix, "SWAP  ", curses.A_BOLD)
        draw_gauge(scr, row, ix + 6, bw, pct)
        put(scr, row + 1, ix + 6,
            f"{used}/{total} MB ({pct:.0f}%)   cached {cached} MB",
            curses.A_DIM, xmax=x + w - 1)


def panel_temps(scr, y, x, h, w, s):
    draw_box(scr, y, x, h, w, "Temperatures")
    iy, ix, iw = y + 1, x + 2, w - 4
    # Mini-gauges scaled 0-100°C, one sensor per row (sorted = stable order).
    bw = max(5, iw - 18)
    for i, (name, c) in enumerate(sorted(s.get("temps", {}).items())):
        if iy + i >= y + h - 1:
            break                              # more sensors than rows: clip
        put(scr, iy + i, ix, f"{name:<6}", curses.A_BOLD)
        draw_gauge(scr, iy + i, ix + 7, bw, c, attr=temp_attr(c))
        put(scr, iy + i, ix + 7 + bw, f" {c:5.1f}°C", temp_attr(c),
            xmax=x + w - 1)


def panel_power(scr, y, x, h, w, s):
    draw_box(scr, y, x, h, w, "Power")
    iy, ix, iw = y + 1, x + 2, w - 4
    rails = s.get("power", {})
    # Scale bars against the busiest rail so they're comparable to each other.
    peak = max((max(cur, avg) for cur, avg in rails.values()), default=1)
    bw = max(5, iw - 34)
    for i, (name, (cur, pavg)) in enumerate(rails.items()):
        if iy + i >= y + h - 1:
            break
        put(scr, iy + i, ix, f"{name:<15}", curses.A_BOLD)
        draw_gauge(scr, iy + i, ix + 16, bw, cur / peak * 100,
                   attr=curses.color_pair(GREEN))
        put(scr, iy + i, ix + 16 + bw, f" {cur:5d} mW", xmax=x + w - 1)
        put(scr, iy + i, ix + 16 + bw + 9, f" avg {pavg} mW", curses.A_DIM,
            xmax=x + w - 1)


# ---------------------------------------------------------------------------
# LAYOUT + MAIN LOOP
# ---------------------------------------------------------------------------
def draw_frame(scr, state, source_label):
    """Compute the panel layout from the current screen size and draw once.

    Layout (proportions recomputed every frame, so resize just works):

        ┌ CPU (58% width) ──────┐┌ Memory ────────────┐
        │ core gauges + graph   ││ RAM / SWAP gauges  │
        │                       │├ Temperatures ──────┤
        ├ GPU ──────────────────┤│ sensor gauges      │
        │ gauge + graph         │├ Power ─────────────┤
        │                       ││ VDD_* rails        │
        └───────────────────────┘└────────────────────┘
         q quit                        <timestamp>
    """
    scr.erase()
    H, W = scr.getmaxyx()

    with state.lock:
        s = state.latest
        cpu_hist = list(state.cpu_hist)
        gpu_hist = list(state.gpu_hist)

    if s is None:
        put(scr, H // 2, max(0, W // 2 - 12), "waiting for tegrastats...",
            curses.A_DIM)
        scr.refresh()
        return

    if H < 16 or W < 70:
        put(scr, 0, 0, f"terminal too small ({W}x{H}), need at least 70x16",
            curses.color_pair(RED))
        scr.refresh()
        return

    body_h = H - 1                             # last row is the footer
    left_w = max(46, int(W * 0.58))
    right_w = W - left_w
    rx = left_w

    # Left column: CPU on top, GPU below. GPU gets a fixed slice; CPU the rest.
    gpu_h = max(6, body_h // 4)
    cpu_h = body_h - gpu_h
    panel_cpu(scr, 0, 0, cpu_h, left_w, s, cpu_hist)
    panel_gpu(scr, cpu_h, 0, gpu_h, left_w, s, gpu_hist)

    # Right column: Memory (fixed) on top, Temps and Power split the rest.
    mem_h = 6
    ntemps = len(s.get("temps", {}))
    npower = len(s.get("power", {}))
    temp_h = min(ntemps + 2, max(3, (body_h - mem_h) * ntemps
                                 // max(1, ntemps + npower)))
    power_h = body_h - mem_h - temp_h
    panel_mem(scr, 0, rx, mem_h, right_w, s)
    panel_temps(scr, mem_h, rx, temp_h, right_w, s)
    panel_power(scr, mem_h + temp_h, rx, power_h, right_w, s)

    # Footer: keybind hint left, data source + timestamp right.
    put(scr, H - 1, 1, " q quit ", curses.A_REVERSE)
    tail = f"{source_label}  {s.get('time', '')} "
    put(scr, H - 1, W - len(tail) - 1, tail, curses.A_DIM)
    scr.refresh()


def tui(scr, state, source_label):
    """Curses main loop: poll keyboard, redraw ~5x/s, exit on q/Esc/EOF."""
    curses.curs_set(0)
    init_colors()
    scr.timeout(200)                           # getch() doubles as frame timer
    while True:
        ch = scr.getch()
        if ch in (ord("q"), ord("Q"), 27):     # 27 = Esc
            break
        draw_frame(scr, state, source_label)
        if state.eof:                          # stream died: last frame, leave
            break


# ---------------------------------------------------------------------------
# INPUT ACQUISITION + ENTRY POINT
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="btop-style TUI (and CSV exporter) for NVIDIA Jetson "
                    "tegrastats. See the module docstring for full docs.")
    p.add_argument("logfile", nargs="?",
                   help="replay a saved tegrastats text log instead of live "
                        "data (prints a plain snapshot, no TUI)")
    p.add_argument("--log", metavar="FILE.csv", dest="csv_path",
                   help="export every parsed sample (all fields) to this CSV "
                        "file during the run; rows are flushed as they arrive")
    return p.parse_args()


def acquire_stream(args):
    """Work out where the data comes from. Returns (stream, label, proc).

    Three cases:
      * file argument      -> no TUI, caller prints a plain snapshot
      * stdin is a pipe    -> data from stdin; steal /dev/tty onto fd 0 so
                              curses can still read the keyboard
      * stdin is the tty   -> we spawn `tegrastats` ourselves (proc returned
                              so it can be killed on exit)
    """
    if args.logfile:
        return open(args.logfile), f"file:{args.logfile}", None

    if not sys.stdin.isatty():
        # Keep a private duplicate of the pipe for the reader thread, then
        # point fd 0 at the real terminal so curses gets key presses.
        data = os.fdopen(os.dup(0), errors="replace")
        try:
            tty = open("/dev/tty")
            os.dup2(tty.fileno(), 0)
        except OSError:
            pass  # headless (no controlling tty): TUI runs without keyboard
        return data, "stdin", None

    # Run interactively with no pipe: launch tegrastats ourselves.
    try:
        proc = subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
    except FileNotFoundError:
        sys.exit("tegrastats not found — run on a Jetson, or pipe input:\n"
                 "  sudo tegrastats | python3 tegrastats_viewer.py")
    return proc.stdout, "tegrastats (spawned)", proc


def snapshot(stream, logger=None):
    """Non-TUI fallback (file replay / redirected output): print last sample.

    With --log this doubles as a converter: every line of the replayed file
    is written to the CSV, then the final sample is printed as a summary.
    """
    last = None
    for line in stream:
        s = parse_line(line)
        if s:
            if logger:
                logger.write(s)
            last = s
    if not last:
        sys.exit("no tegrastats data found in input")
    print(f"tegrastats snapshot  {last.get('time', '')}")
    online = [c[0] for c in last["cores"] if c]
    avg = sum(online) / max(1, len(online))
    for i, core in enumerate(last["cores"]):
        print(f"  core{i}: " + ("offline" if core is None
                                else f"{core[0]:3d}% @ {core[1]} MHz"))
    print(f"  cpu avg: {avg:.0f}%")
    if "gpu" in last:
        pct, freq = last["gpu"]
        print(f"  gpu: {pct}%" + (f" @ {freq} MHz" if freq else ""))
    if "ram" in last:
        u, t, lfb = last["ram"]
        print(f"  ram: {u}/{t} MB ({u / t * 100:.0f}%)  lfb {lfb}")
    if "swap" in last:
        u, t, c = last["swap"]
        print(f"  swap: {u}/{t} MB  cached {c} MB")
    for name, c in sorted(last.get("temps", {}).items()):
        print(f"  temp {name}: {c:.1f}°C")
    for name, (cur, pavg) in last.get("power", {}).items():
        print(f"  {name}: {cur} mW (avg {pavg} mW)")


def main():
    args = parse_args()
    stream, label, proc = acquire_stream(args)
    logger = CsvLogger(args.csv_path) if args.csv_path else None

    # No terminal to draw on, or replaying a file: plain snapshot instead.
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
    t = threading.Thread(target=reader_thread, args=(stream, state, logger),
                         daemon=True)
    t.start()

    try:
        curses.wrapper(tui, state, label)      # wrapper restores the terminal
    except KeyboardInterrupt:
        pass                                   # Ctrl+C is a normal way to quit
    finally:
        if proc:                               # stop a tegrastats we spawned
            proc.terminate()
        try:
            stream.close()
        except OSError:
            pass
        if logger:
            # The reader thread owns the logger; give it a moment to finish
            # its last row before closing the file underneath it.
            t.join(timeout=1)
            logger.close()


if __name__ == "__main__":
    main()
