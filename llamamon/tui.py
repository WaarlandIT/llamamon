#!/usr/bin/env python3
"""llamamon - live TUI monitor for a local llama-server.

Shows GPU usage, memory (GTT/VARM when the driver exposes it), tokens/s,
query latency in llama-server, temperatures, fan speed and clock frequencies.
Runs locally on the machine where llama-server runs.

Keys: q quit | p pause | r reset history | space refresh
"""

import asyncio
import collections
import json
import socket
import os
import re
import subprocess
import time
from pathlib import Path

import httpx
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from . import __version__

DEFAULT_URL = "http://127.0.0.1:8090"   # llama-server's own default port


def default_url() -> str:
    """LLAMA_URL, else whatever llama-server listens on, else llama-server's default port."""
    env = os.environ.get("LLAMA_URL")
    if env:
        return env.rstrip("/")
    for port in (8080, 8090):
        if _port_open(port):
            return f"http://127.0.0.1:{port}"
    return DEFAULT_URL


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.15):
            return True
    except OSError:
        return False


LLAMA = default_url()
INTERVAL = float(os.environ.get("LLAMAMON_INTERVAL", "1.0"))
HIST = 120  # samples kept for the charts
MAX_HISTORY_LINES = 20000  # queries kept in the JSONL log
RAM_FALLBACK = 16 * 1024**3  # used when /proc/meminfo is unreadable
HISTORY = os.environ.get("LLAMAMON_HISTORY", "~/.llamamon-history.jsonl")
# Board specific sensor labels; see `sensors -j` on your own machine. Defaults
# are left empty on purpose: the tool then takes the best match it can find
# (the first TSI/Tctl it sees, the fastest fan) instead of showing nothing.
SENS_FAN_CHIP = os.environ.get("LLAMAMON_SENS_CHIP", "")
SENS_TSI = os.environ.get("LLAMAMON_SENS_TSI", "AMD TSI")
SENS_TCTL_CHIP = os.environ.get("LLAMAMON_SENS_TCTL_CHIP", "")

SPARK = "▁▂▃▄▅▆▇█"
TEMP_WARN, TEMP_CRIT = 85.0, 95.0
MEM_WARN, MEM_CRIT = 90.0, 97.0


# ---------------------------------------------------------------- verzamelen
def read_sys(path: str, cast=float):
    try:
        return cast(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def find_hwmon() -> str:
    for d in sorted(Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*")):
        if read_sys(str(d / "name"), str) == "amdgpu":
            return str(d)
    return ""


def find_freq_mhz() -> float:
    for p in Path("/sys/class/drm").glob("card*/device/pp_dpm_sclk"):
        try:
            for line in p.read_text().splitlines():
                if line.rstrip().endswith("*"):
                    m = re.search(r"(\d+)Mhz", line, re.I)
                    if m:
                        return float(m.group(1))
        except OSError:
            pass
    return 0.0


def read_sensors() -> dict:
    """sensors -j for temperatures amdgpu hwmon does not expose (TSI, tctl, fans)."""
    try:
        out = subprocess.run(
            ["sensors", "-j"], capture_output=True, text=True, timeout=3
        ).stdout
        return json.loads(out or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}


def pick(sensors: dict, chip_prefix: str, feature: str, label_re: str):
    """First reading in `sensors -j` whose chip and label match.

    An empty `chip_prefix` means "any chip", so a machine whose labels we do not
    know still shows its temperatures.
    """
    for chip, feats in sensors.items():
        if chip_prefix and not chip.startswith(chip_prefix):
            continue
        for feat, vals in feats.items():
            if not feat.startswith(feature):
                continue
            lab = feat.split(":")[-1].strip()
            m = re.search(label_re, feat + " " + lab, re.I)
            if not m:
                continue
            for k, v in vals.items():
                if k.endswith("_input") and v is not None:
                    return float(v)
    return None


def pick_first_fan(sensors: dict):
    fans = []
    for chip, feats in sensors.items():
        if SENS_FAN_CHIP and not chip.startswith(SENS_FAN_CHIP):
            continue
        for feat, vals in feats.items():
            if "fan" in feat.lower() and not feat.lower().startswith("pwm"):
                v = vals.get("fan_input")  # sensors -j: fan2_input
                for k, val in vals.items():
                    if k.endswith("_input"):
                        v = val
                        break
                if v:
                    fans.append(float(v))
    return max(fans) if fans else None


def read_stat() -> list:
    """Per-CPU lines from /proc/stat (first line = total, then cpu0..cpuN)."""
    out = []
    try:
        for line in open("/proc/stat"):
            if not line.startswith("cpu"):
                break
            out.append([int(x) for x in line.split()[1:]])
    except (OSError, ValueError):
        pass
    return out


def compute_busy(prev, cur, idx: int) -> float:
    """CPU load of one cpu line between two samples."""
    if not prev or len(prev) <= idx or len(cur) <= idx:
        return 0.0
    a, b = prev[idx], cur[idx]
    idle_a, idle_b = a[3] + (a[4] if len(a) > 4 else 0), b[3] + (b[4] if len(b) > 4 else 0)
    dt = sum(b) - sum(a)
    di = idle_b - idle_a
    if dt <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (dt - di) / dt))


def gpu_busy(hwmon: str) -> float:
    """Busy percentage from the driver when it exists, otherwise the busiest CPU
    core as a proxy -- on an APU the GPU backend runs on the same cores."""
    for p in ("gpu_busy_percent", "mem_busy_percent"):
        v = read_sys(str(Path(hwmon).parent.parent / p))
        if v is not None:
            return v
    return gpu_busy.proxy


gpu_busy.proxy = 0.0


def cpu_busy() -> float:
    try:
        s = open("/proc/stat").readline().split()[1:]
        vals = [int(x) for x in s]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
    except (OSError, ValueError):
        return 0.0
    prev = cpu_busy.prev
    cpu_busy.prev = (total, idle)
    if not prev or total == prev[0]:
        return 0.0
    dt, di = total - prev[0], idle - prev[1]
    return max(0.0, min(100.0, 100.0 * (dt - di) / dt))


cpu_busy.prev = None


def find_pid_by_prompt(now: float):
    """PID and start time (epoch) of the llama-server listening on our port.

    /proc/<pid>/fd van andere gebruikers is niet leesbaar (ptrace_scope=1), dus:
    koppel de luisterende socket-inode aan de UID in /proc/net/tcp, loop /proc voor
    die UID af en neem het proces met dezelfde inode (via /proc/<pid>/net is niet
    bruikbaar, dus we vergelijken starttijd + sockettelling-alternatief: het enige
    proces van die UID met 'llama-server' in cmdline)."""
    if now - find_pid_by_prompt.last < 20 and find_pid_by_prompt.pid:
        return find_pid_by_prompt.pid, find_pid_by_prompt.started
    find_pid_by_prompt.last = now
    owners = set()
    want = f"{find_pid_by_prompt.port:04X}"
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in open(path).read().splitlines()[1:]:
                p = line.split()
                if len(p) > 9 and p[1].split(":")[1] == want and p[3] == "0A":
                    owners.add(p[7])
        except OSError:
            continue
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            st = d.stat()
            if str(st.st_uid) not in owners:
                continue
            cmd = (d / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
            if "llama-server" not in cmd:
                continue
        except OSError:
            continue
        find_pid_by_prompt.pid = int(d.name)
        find_pid_by_prompt.started = st.st_ctime
        return find_pid_by_prompt.pid, find_pid_by_prompt.started
    find_pid_by_prompt.pid = None
    return None, None


find_pid_by_prompt.last = 0.0
find_pid_by_prompt.pid = None
find_pid_by_prompt.started = None
find_pid_by_prompt.port = int(default_url().rstrip("/").rsplit(":", 1)[-1])


def proc_ppid(pid: int) -> int:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 1


def clients_on_port(port: int) -> int:
    """Number of TCP connections from other processes to the llama-server port."""
    inodes = set()
    n = 0
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = open(f).read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            p = line.split()
            if len(p) < 10:
                continue
            if p[1].split(":")[1].upper() == f"{port:04X}" and p[3] == "01":  # ESTABLISHED
                inodes.add(p[9])
                n += 1
    if not inodes:
        return 0
    return n


class Collector:
    """One cycle: all hardware readings plus the derived values from /slots."""

    def __init__(self, url: str = None, interval: float = INTERVAL):
        self.hwmon = find_hwmon()
        self.url = (url or default_url()).rstrip("/")
        self.interval = interval
        self.port = int(self.url.rsplit(":", 1)[1])
        self.client = httpx.AsyncClient(base_url=self.url, timeout=2.0)
        self.prev_task = None
        self.prev_at = None
        self.prev_stat = None
        self.prev_processed = 0
        self.prev_pre_n = 0
        self.prev_pre_at = None
        self.prev_decoded = 0
        self.requests = 0
        self.started = time.time()
        self.last_req = None
        self.cur = None

    async def _get(self, path: str):
        try:
            r = await self.client.get(path)
            if r.status_code == 200:
                return r.json()
        except (httpx.HTTPError, json.JSONDecodeError, RuntimeError, OSError):
            return None

    def now(self) -> float:
        """Single source of 'current time', so tests and screenshots can pin it."""
        return time.time()

    async def sample(self, now: float = None) -> dict:
        now = self.now() if now is None else now
        m = {"ts": now, "url": self.url}

        # --- systeem
        try:
            mem = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":")[0], line.split()[1]
                mem[k] = int(v) * 1024
            m["mem_total"] = mem.get("MemTotal", RAM_FALLBACK)
            m["mem_avail"] = mem.get("MemAvailable", 0)
            m["swap_total"] = mem.get("SwapTotal", 0)
            m["swap_used"] = mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)
        except OSError:
            m["mem_total"] = RAM_FALLBACK
            m["mem_avail"] = 0
            m["swap_total"] = m["swap_used"] = 0
        m["mem_used"] = m["mem_total"] - m["mem_avail"]
        m["cpu"] = cpu_busy()
        m["load"] = os.getloadavg()[0]
        stat_now = read_stat()
        if self.prev_stat:
            busiest = 0.0
            for i in range(1, min(len(stat_now), len(self.prev_stat))):
                b = compute_busy(self.prev_stat, stat_now, i)
                m.setdefault("cores", []).append(b)
                busiest = max(busiest, b)
            gpu_busy.proxy = busiest
            m["core_max"] = busiest
        else:
            m["core_max"] = 0.0
        self.prev_stat = stat_now

        # --- GPU (amdgpu hwmon + drm)
        dev = Path(self.hwmon).parent.parent if self.hwmon else Path("/nonexistent")
        m["gpu_busy"] = gpu_busy(self.hwmon) if self.hwmon else 0.0
        m["gtt_total"] = read_sys(str(dev / "mem_info_gtt_total")) or 0.0
        m["gtt_used"] = read_sys(str(dev / "mem_info_gtt_used")) or 0.0
        m["vram_total"] = read_sys(str(dev / "mem_info_vram_total")) or 0.0
        m["vram_used"] = read_sys(str(dev / "mem_info_vram_used")) or 0.0
        m["sclk"] = find_freq_mhz()
        m["vddgfx"] = read_sys(f"{self.hwmon}/in0_input")
        m["power"] = (read_sys(f"{self.hwmon}/power1_input") or 0.0) / 1e6
        m["temp_edge"] = (read_sys(f"{self.hwmon}/temp1_input") or 0.0) / 1000

        sens = read_sensors()
        m["temp_tsi"] = pick(sens, SENS_FAN_CHIP, SENS_TSI, r"TSI")
        m["temp_tctl"] = pick(sens, SENS_TCTL_CHIP, "Tctl", r"Tctl")
        m["therm14"] = pick(sens, SENS_FAN_CHIP, "Thermistor 14", r"Thermistor 14")
        m["therm15"] = pick(sens, SENS_FAN_CHIP, "Thermistor 15", r"Thermistor 15")
        m["fan"] = pick_first_fan(sens)

        # --- llama-server
        props = await self._get("/props")
        slots = await self._get("/slots")
        m["server"] = props is not None
        if props:
            m["model"] = props.get("model_alias") or "?"
            m["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx", 0)
            m["slots_total"] = props.get("total_slots", 1)
        else:
            m["model"] = m.get("model", "?")
            m["n_ctx"] = m.get("n_ctx", 0)
            m["slots_total"] = m.get("slots_total", 1)

        m["clients"] = clients_on_port(self.port)

        # server-starttijd en uptime van de service
        find_pid_by_prompt.port = self.port
        pid, started = find_pid_by_prompt(now)
        if pid:
            m["server_since"] = started
            m["server_uptime"] = now - started
            try:
                roll = Path(f"/proc/{pid}/smaps_rollup").read_text()
                m["server_rss"] = int(roll.split("Rss:")[1].split()[0]) * 1024
                m["server_swap"] = int(roll.split("Swap:")[1].split()[0]) * 1024
                m["server_locked"] = int(roll.split("Locked:")[1].split()[0]) * 1024
            except (OSError, ValueError, IndexError):
                pass
            if m.get("server_rss") is None:  # smaps_rollup unreadable: fall back to status
                try:
                    st = Path(f"/proc/{pid}/status").read_text()
                    m["server_rss"] = int(st.split("VmRSS:")[1].split()[0]) * 1024
                except (OSError, ValueError, IndexError):
                    pass
        for k in ("server_uptime", "server_rss", "server_swap", "server_locked"):
            m.setdefault(k, None)

        # llama-server zonder --metrics: geen eigen request-teller, dus zelf afgeleid
        m["requests"] = self.requests
        m["uptime"] = now - self.started

        self.derive_slots(slots or [], now, m)
        return m

    def derive_slots(self, slots, now: float, m: dict = None) -> dict:
        """Turn the raw /slots answer into per-request state: which task is
        running, how fast it generates and prefill runs, and when it started.

        Kept free of I/O so it can be unit-tested: pass the slots list and a
        timestamp, get the derived values back.
        """
        m = {} if m is None else m
        processing = False
        task = None
        d_tokens = 0
        for sl in slots:
            if sl.get("is_processing"):
                processing = True
                tid = sl.get("id_task")
                if tid is not None:
                    task = dict(
                        id=tid,
                        prompt=sl.get("n_prompt_tokens", 0),
                        prompt_done=sl.get("n_prompt_tokens_processed", 0),
                        cached=sl.get("n_prompt_tokens_cache", 0),
                        ctx=sl.get("n_ctx", m.get("n_ctx", 0)),
                        max_tok=(sl.get("params") or {}).get("max_tokens", 0),
                    )
                    nt = sl.get("next_token") or [{}]
                    d_tokens = nt[0].get("n_decoded", 0) or 0
                break

        inst_gen = inst_pre = 0.0
        same = (processing and task and self.prev_task
                and self.prev_task["id"] == task["id"] and self.prev_at)
        if same:
            dt = now - self.prev_at
            if dt > 0:
                inst_gen = max(0.0, (d_tokens - self.prev_decoded) / dt)
                inst_pre = max(0.0, (task["prompt_done"] - self.prev_processed) / dt)
        elif processing and task:
            inst_gen = inst_pre = 0.0
        m["tok_s"] = inst_gen
        m["prefill_s"] = inst_pre
        m["processing"] = processing

        if processing and task:
            if not self.prev_task or self.prev_task["id"] != task["id"]:
                self.cur = dict(
                    id=task["id"], t0=now, prompt=task["prompt"],
                    cached=task["cached"], max_tok=task["max_tok"],
                    ctx=task["ctx"], prefill_s=0.0, gen=[],
                    decode_tokens=0, has_output=False,
                )
            c = self.cur
            if c and not c["gen"]:
                c["prefill_s"] = c["prefill_s"] or 0.0
            if c:
                c["prompt"] = task["prompt"]
                c["cached"] = task["cached"]
                # prefill is only measured during the prefill phase; generation
                # starts once there is output (the first jump is not measurable)
                if task["prompt_done"] and task["cached"] is not None and not c.get("has_output"):
                    done = max(0, task["prompt_done"] - (task["cached"] or 0))
                    pref_done = task["prompt_done"] >= task["prompt"]
                    if pref_done and done and self.prev_pre_at:
                        dt = max(now - self.prev_pre_at, 0.05)
                        c["prefill_s"] = ((done - self.prev_pre_n) / dt
                                          if done > self.prev_pre_n else done / dt)
                    elif done and now > c["t0"]:
                        c["prefill_s"] = done / (now - c["t0"])
                if d_tokens and c.get("has_output") and inst_gen >= 0:
                    c["gen"].append(inst_gen)
                if d_tokens:
                    c["has_output"] = True
                    c["decode_tokens"] = d_tokens
                c["elapsed"] = now - c["t0"]
                self.last_req = c
            m["req"] = c
        else:
            m["req"] = None
            self.prev_task, self.prev_at = None, None

        if m.get("req"):
            self.prev_task = task
            self.prev_at = now
            self.prev_processed = task["prompt_done"]
            self.prev_pre_n, self.prev_pre_at = task["prompt_done"], now
            self.prev_decoded = d_tokens
        return m

    def reset(self):
        self.requests = 0
        self.started = time.time()


# ---------------------------------------------------------------- widgets
def spark(series, width: int = 40, lo=None, hi=None) -> str:
    vals = list(series)[-width:]
    if not vals:
        return "·" * min(width, 8)
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    span = max(hi - lo, 1e-9)
    out = []
    for v in vals:
        if v is None:
            out.append("·")
            continue
        idx = int((v - lo) / span * (len(SPARK) - 1))
        out.append(SPARK[max(0, min(len(SPARK) - 1, idx))])
    return "".join(out)


def fmt_cores(cores) -> str:
    out = []
    for i, c in enumerate(cores):
        out.append(("█" if c > 85 else "▓" if c > 50 else "▒" if c > 15 else "░") + f"{c:3.0f}")
    return " ".join(out)


def bar(frac: float, width: int, color: str = "green") -> Text:
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * width))
    t = Text()
    t.append("█" * fill, style=color)
    t.append("░" * (width - fill), style="grey30")
    return t


def fmt_dur(s) -> str:
    if s is None:
        return "—"
    s = float(s)
    if s < 60:
        return f"{s:4.1f}s"
    return f"{int(s // 60)}m{s % 60:04.1f}s"


def fmt_gb(n) -> str:
    return f"{n / 1024**3:5.2f}G"


def mem_line(label: str, used, total, width: int) -> Text:
    """One memory row, sized to the panel so it never wraps."""
    frac = (used / total) if total else 0.0
    col = "red" if frac > .97 else "yellow" if frac > .9 else "green"
    if not total:
        return Text.assemble((label, "bold"), ("      n/a\n", "grey50"))
    txt = f" {fmt_gb(used)} / {fmt_gb(total)}\n"
    barw = max(8, width - len(label) - len(txt) - 2)
    return Text.assemble((label, "bold"), bar(frac, barw, col), (txt, ""))


def temp_color(t) -> str:
    if t is None:
        return "grey50"
    if t >= TEMP_CRIT:
        return "bold red"
    if t >= TEMP_WARN:
        return "yellow"
    return "green"


class HistoryFile:
    """One JSON line per finished query, so tok/s and latency stay available
    outside a running session (llamamon --stats)."""

    def __init__(self, path: str, max_lines: int = MAX_HISTORY_LINES):
        self.path = Path(os.path.expanduser(path))
        self.max_lines = max_lines
        self.lines_written = 0

    def trim(self) -> None:
        """Keep the log bounded: an always-on monitor writes a line per query."""
        try:
            lines = self.path.read_text(errors="replace").splitlines()
            if len(lines) > self.max_lines:
                self.path.write_text("\n".join(lines[-self.max_lines:]) + "\n")
        except OSError:
            pass

    def append(self, r: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({
                    "ts": round(r["t0"], 2),
                    "duration": round(r.get("elapsed") or 0, 2),
                    "prompt": r["prompt"],
                    "cache": r["cached"] or 0,
                    "out": r["decode_tokens"],
                    "prefill_s": round(r.get("prefill_s") or 0, 1),
                    "gen_s": round(sum(r["gen"]) / len(r["gen"]), 2) if r["gen"] else 0.0,
                }) + "\n")
            self.lines_written += 1
            if self.lines_written % 200 == 0:
                self.trim()
        except OSError:
            pass

    def read(self):
        rows = []
        try:
            for line in self.path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
        return rows


def normalise(rows):
    """Accept logs written by older builds, where the elapsed field was "duur"."""
    for r in rows:
        if "duration" not in r and "duur" in r:
            r["duration"] = r["duur"]
    return rows


def print_stats(path: str, since_hours=None) -> None:
    """Summary of the query log: p50/p90/max latency and tok/s, per hour."""
    rows = normalise(HistoryFile(path).read())
    if since_hours is not None:
        rows = [r for r in rows if r["ts"] > time.time() - since_hours * 3600]
    if not rows:
        print(f"no queries in {path}")
        return

    def pct(vals, p):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    durs = [r.get("duration", 0) or 0 for r in rows]
    gens = [r["gen_s"] for r in rows if r["gen_s"]]
    pres = [r["prefill_s"] for r in rows if r["prefill_s"]]
    outs = [r["out"] for r in rows]
    print(f"{len(rows)} queries between {time.strftime('%Y-%m-%d %H:%M', time.localtime(rows[0]['ts']))}"
          f" and {time.strftime('%Y-%m-%d %H:%M', time.localtime(rows[-1]['ts']))}")
    print(f"  duration  p50 {pct(durs, .5):6.1f}s   p90 {pct(durs, .9):6.1f}s   max {max(durs):6.1f}s")
    if gens:
        print(f"  tok/s     p50 {pct(gens, .5):6.1f}    p90 {pct(gens, .9):6.1f}    min {min(gens):6.1f}")
    if pres:
        print(f"  prefill   p50 {pct(pres, .5):6.0f}    p90 {pct(pres, .9):6.0f}    min {min(pres):6.0f} tok/s")
    print(f"  output    p50 {pct(outs, .5):5.0f} tok  total {sum(outs):,} tok")
    print("\n  per hour:")
    buckets = {}
    for r in rows:
        buckets.setdefault(time.strftime("%Y-%m-%d %H", time.localtime(r["ts"])), []).append(r)
    for h, rs in sorted(buckets.items()):
        g = [x["gen_s"] for x in rs if x["gen_s"]]
        print(f"    {h}  {len(rs):3d} queries   p50 {pct([x['duration'] for x in rs], .5):6.1f}s"
              f"   tok/s p50 {pct(g, .5) if g else 0:5.1f}   {sum(x['out'] for x in rs):6,} tok")


class Panel(Static):
    """Bordered panel with a title; content is replaced on every tick."""

    def __init__(self, title: str = "", **kw):
        super().__init__(**kw)
        self.border_title = title


class Graphs(Panel):
    def __init__(self, **kw):
        super().__init__("charts (last 2 min)", **kw)
        self.hist = {k: collections.deque(maxlen=HIST) for k in
                     ("gpu", "tok", "cpu", "mem", "temp", "power")}

    def on_mount(self) -> None:
        self.set_interval(self.app.interval, self.tick)

    def tick(self) -> None:
        d = self.app.data
        if not d or self.app.demo:  # a frozen screen keeps its fixtures
            return
            return
        h = self.hist
        h["gpu"].append(d.get("gpu_busy", 0.0))
        h["tok"].append(d.get("tok_s") or 0.0)
        h["cpu"].append(d.get("cpu", 0.0))
        total = d.get("mem_total") or 1
        h["mem"].append(100.0 * (d.get("mem_used", 0) / total))
        h["temp"].append(d.get("temp_edge") or 0.0)
        h["power"].append(d.get("power") or 0.0)
        self.refresh()

    def render(self) -> Text:
        d = self.app.data or {}
        w = max(16, self.size.width - 26)  # leave room for the value column
        t = Text()
        rows = [
            ("GPU  %", "gpu", 0, 100, f"{d.get('gpu_busy', 0):5.1f}%"),
            ("tok/s", "tok", 0, None, f"{d.get('tok_s') or 0:5.1f}"),
        ]
        for label, key, lo, hi, val in rows:
            s = spark(self.hist[key], w, lo, hi if hi else None)
            t.append(f"  {label:<14}", style="bold")
            t.append(s, style="cyan")
            t.append(f" {val}\n")
        mem_pct = 100.0 * (d.get("mem_used", 0) / (d.get("mem_total") or 1))
        t.append(f"  {'RAM %':<14}", style="bold")
        t.append(spark(self.hist["mem"], w, 0, 100), style="magenta")
        t.append(f" {mem_pct:5.1f}%\n")
        t.append(f"  {'CPU %':<14}", style="bold")
        t.append(spark(self.hist["cpu"], w, 0, 100), style="blue")
        t.append(f" {d.get('cpu', 0):5.1f}%\n")
        t.append(f"  {'temp °C':<14}", style="bold")
        t.append(spark(self.hist["temp"], w, 40, 100), style=temp_color(d.get("temp_edge")))
        t.append(f" {d.get('temp_edge', 0):5.1f}\n")
        t.append(f"  {'power W':<14}", style="bold")
        t.append(spark(self.hist["power"], w, 0, 160), style="yellow")
        t.append(f" {(d.get('power') or 0):5.1f}\n")
        return t


class GpuPanel(Panel):
    def __init__(self, title: str = "GPU / thermal", **kw):
        super().__init__(title, **kw)

    def render(self) -> Text:
        d = self.app.data or {}
        if not d:
            return Text("waiting for first sample…", style="grey50")
        t = Text()
        gpu_txt = f" {d.get('gpu_busy', 0):5.1f}%\n"
        t.append_text(Text.assemble(("  GPU  ", "bold"),
                                    bar(d.get('gpu_busy', 0) / 100,
                                        max(8, self.size.width - 2 - 28 - len(gpu_txt)), "cyan"),
                                    (gpu_txt, "")))
        t.append_text(Text.assemble(("  cores         ", "bold"),
                                    (fmt_cores(d.get("cores") or []), "grey70"), "\n"))
        w = self.size.width - 2
        t.append_text(mem_line("  GTT (VRAM)    ", d.get('gtt_used', 0), d.get('gtt_total', 0), w))
        t.append_text(mem_line("  system RAM    ", d.get('mem_used', 0), d.get('mem_total', 0), w))
        t.append_text(mem_line("  swap          ", d.get('swap_used', 0), d.get('swap_total', 0), w))
        cpu = f" {d.get('cpu', 0):5.1f}%  load {d.get('load', 0):.2f}\n"
        t.append_text(Text.assemble(("  CPU / load    ", "bold"),
                                    bar(d.get('cpu', 0) / 100, max(8, w - 24 - len(cpu)), "blue"),
                                    (cpu, "")))
        t.append("\n")
        t.append_text(Text.assemble(
            ("  temp edge     ", "bold"), (f"{d.get('temp_edge', 0):5.1f}°C\n", temp_color(d.get('temp_edge', 0)))))
        t.append_text(Text.assemble(
            ("  temp TSI      ", "bold"), (f"{(d.get('temp_tsi', 0) or 0):5.1f}°C\n", temp_color(d.get('temp_tsi', 0)))))
        t.append_text(Text.assemble(
            ("  temp Tctl     ", "bold"), (f"{(d.get('temp_tctl', 0) or 0):5.1f}°C\n", temp_color(d.get('temp_tctl', 0)))))
        t.append_text(Text.assemble(
            ("  thermistor14/15 ", "bold"),
            (f"{(d.get('therm14', 0) or 0):4.1f} / {(d.get('therm15', 0) or 0):4.1f}°C\n", temp_color(max(d.get('therm14', 0) or 0, d.get('therm15', 0) or 0)))))
        t.append_text(Text.assemble(("  fan / sclk    ", "bold"),
                                    (f"{(d.get('fan', 0) or 0):4.0f} rpm  {d.get('sclk', 0):4.0f} MHz\n")))
        t.append_text(Text.assemble(("  PPT / vddgfx  ", "bold"),
                                    (f"{(d.get('power', 0) or 0):5.1f} W  {(d.get('vddgfx', 0) or 0):4.0f} mV\n")))
        return t


class ServerPanel(Panel):
    def __init__(self, title: str = "llama-server", **kw):
        super().__init__(title, **kw)

    def render(self) -> Text:
        d = self.app.data or {}
        if not d:
            return Text("waiting for first sample…", style="grey50")
        t = Text()
        st = ("● online", "green") if d.get("server") else ("✖ offline", "bold red")
        t.append_text(Text.assemble(("  status        ", "bold"), st, (f"  {d.get('url','')}", "grey50"),
                                    (f"   clients {d.get('clients', 0)}   "
                                     f"slots {d.get('slots_total', 1)}\n", "")))
        t.append_text(Text.assemble(("  model         ", "bold"), (f"{d.get('model','?')}\n", "cyan")))
        t.append_text(Text.assemble(("  context       ", "bold"),
                                    (f"{(d.get('n_ctx') or 0):,} tokens\n", "")))
        up = d.get("server_uptime")
        rss = d.get("server_rss")
        t.append_text(Text.assemble(
            ("  server up     ", "bold"),
            (f"{fmt_dur(up) if up is not None else '—'}", ""),
            ("   RSS ", "bold"),
            (f"{fmt_gb(rss) if rss else '—'}\n", "")))
        t.append("\n")
        cur = d.get("req")
        busy = ("● busy", "bold yellow") if d.get("processing") else ("○ idle", "grey50")
        t.append_text(Text.assemble(("  task          ", "bold"), busy, "\n"))
        if cur:
            t.append_text(Text.assemble(
                ("  elapsed       ", "bold"),
                (fmt_dur(cur.get("elapsed")), "bold yellow"),
                ("   now ", "bold"), (f"{(d.get('tok_s') or 0):4.1f} tok/s\n", "green")))
            gen = sum(cur["gen"]) / len(cur["gen"]) if cur["gen"] else 0.0
            t.append_text(Text.assemble(
                ("  speed         ", "bold"), (f"{gen:5.1f} tok/s avg", "green"),
                ("   prefill ", "bold"), (f"{(cur['prefill_s'] or 0):6.1f} tok/s\n", "cyan")))
            ctx = cur.get("ctx") or 1
            # prompt usage as a fraction of the context, in the space there is
            tail = f" {100 * cur['prompt'] / ctx:.1f}% of {ctx // 1024}k"
            ptxt = f"{cur['prompt']:,} tok ({cur['cached']:,} cached)"
            bw = max(6, min(10, self.size.width - 18 - len(ptxt) - len(tail) - 1))
            t.append_text(Text.assemble(
                ("  prompt        ", "bold"), (ptxt + " ", ""),
                bar(cur["prompt"] / ctx, bw, "cyan"), (tail + "\n", "")))
            mx = cur.get("max_tok") or 0
            if mx > 0:
                t.append_text(Text.assemble(
                    ("  output        ", "bold"),
                    (f"{cur['decode_tokens']:,} / {mx:,} tok  ", ""),
                    bar(cur["decode_tokens"] / mx, 12, "green"),
                    f" {100*cur['decode_tokens']/mx:4.1f}%\n"))
        else:
            t.append_text(Text("  no active task\n", style="grey50"))
        return t


class History(Panel):
    def __init__(self, **kw):
        super().__init__("recent queries", **kw)
        self.done = collections.deque(maxlen=8)

    def add(self, req: dict) -> None:
        self.done.append(dict(req))

    def render(self) -> Text:
        t = Text()
        if not self.done:
            return Text("  no finished query in this session yet", style="grey50")
        t.append(f"  {'time':>8} {'elap':>6} {'prompt':>7} {'cache':>6} "
                 f"{'out':>5} {'tok/s':>6} {'pre/out':>8}\n", style="bold")
        for r in reversed(self.done):
            gen = sum(r["gen"]) / len(r["gen"]) if r["gen"] else 0.0
            pre = r.get("prefill_s") or 0
            p_out = f"{pre:.0f}/{gen:.0f}" if pre else f"-/{gen:.0f}"
            p_out = p_out[-8:]
            t.append(f"  {time.strftime('%H:%M:%S', time.localtime(r['t0'])):>8} "
                     f"{fmt_dur(r.get('elapsed')):>6} {r['prompt']:>7,} {r['cached'] or 0:>6,} "
                     f"{r['decode_tokens']:>5,} {gen:>6.1f} {p_out:>8}\n")
        return t


class MonitorApp(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 1fr; }
    #left { width: 1fr; }
    #right { width: 1fr; }
    Panel {
        border: round $panel-lighten-2;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 0 0 0 0;
        height: 1fr;
    }
    Header { background: $panel; color: $text; }
    Footer { background: $panel; }
    """
    BINDINGS = [
        ("q", "quit", "quit"),
        ("p", "toggle_pause", "pause"),
        ("r", "reset", "reset"),
        ("space", "force", "refresh"),
    ]

    def __init__(self, url: str = None, interval: float = INTERVAL):
        super().__init__()
        self.collector = Collector(url, interval)
        self.url, self.interval = self.collector.url, interval
        self.history_log = HistoryFile(HISTORY)
        self.data = None
        self.paused = False
        self.demo = None  # a frozen sample dict, used for screenshots/tests

    def compose(self) -> ComposeResult:
        if self.demo:
            self.data = self.demo
        yield Header(show_clock=True)
        with Vertical(id="top"):
            with Horizontal():
                with Vertical(id="left"):
                    yield GpuPanel("GPU / thermal", id="gpu")
                    yield ServerPanel("llama-server", id="server")
                with Vertical(id="right"):
                    yield Graphs(id="graphs")
                    yield History(id="history")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "llamamon"
        self.sub_title = self.url
        self.set_interval(self.interval, self.pump)

    async def pump(self) -> None:
        if self.paused or self.demo:
            return
        try:
            d = await self.collector.sample()
        except Exception as e:  # a failed sample must never take the tool down
            self.notify(f"sample failed: {e}", severity="error", timeout=3)
            return
        if not d:  # nothing measured yet (e.g. the collector was reset)
            return
        prev = self.data.get("req") if self.data else None
        self.data = d
        cur = d.get("req")
        hist = self.query_one(History)
        if prev and (not cur or cur["id"] != prev["id"]):
            hist.add(prev)
            self.history_log.append(prev)
            self.collector.prev_task = None
            self.collector.prev_at = None
        self.query_one(GpuPanel).refresh()
        self.query_one(ServerPanel).refresh()
        hist.refresh()

    async def worker(self):
        return

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.notify("paused" if self.paused else "resumed")

    def action_force(self) -> None:
        self.paused = False
        self.refresh()

    def action_reset(self) -> None:
        self.collector.reset()
        self.query_one(History).done.clear()
        self.query_one(Graphs).hist = {k: collections.deque(maxlen=HIST) for k in
                                       ("gpu", "tok", "cpu", "mem", "temp", "power")}
        self.notify("history reset")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Live monitor for a local llama-server")
    ap.add_argument("--stats", action="store_true", help="print a query-log summary and exit")
    ap.add_argument("--since", type=float, default=None, help="with --stats: last N hours only")
    ap.add_argument("--url", default=None, help="llama-server URL (default %s)" % LLAMA)
    ap.add_argument("--interval", type=float, default=None, help="sample interval in seconds")
    ap.add_argument("--version", action="version", version="llamamon %s" % __version__)
    a = ap.parse_args()
    if a.stats:
        print_stats(HISTORY, a.since)
    else:
        MonitorApp(a.url or LLAMA, a.interval or INTERVAL).run()


if __name__ == "__main__":
    main()
