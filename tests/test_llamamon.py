"""Tests for the pure logic in llamamon: parsing, charts and the query log.

Nothing here touches the network or the UI: every function under test takes
plain values, which is what makes the tool testable on any machine.
"""

import json
import time

import pytest

from llamamon import tui


# --------------------------------------------------------------- /slots maths
def slot(processed=0, cache=0, prompt=100, decoded=0, ident=-1, done=False):
    return {
        "id_task": ident,
        "is_processing": not done,
        "n_prompt_tokens": prompt,
        "n_prompt_tokens_processed": processed,
        "n_prompt_tokens_cache": cache,
        "next_token": [{"n_decoded": decoded}],
    }


def test_first_sample_has_no_rates():
    """Without a previous sample there is no delta, so no rate is reported."""
    c = tui.Collector("http://127.0.0.1:1")
    r = c.derive_slots([slot(processed=120, decoded=40)], now=1000.0)
    assert r["tok_s"] == 0.0
    assert r["processing"] is True


def test_generation_rate_is_delta_over_time():
    c = tui.Collector("http://127.0.0.1:1")
    c.derive_slots([slot(processed=120, decoded=40, ident=1)], now=1000.0)
    r = c.derive_slots([slot(processed=120, decoded=140, ident=1)], now=1001.5)
    assert r["tok_s"] == pytest.approx(100 / 1.5, rel=1e-6)


def test_running_prefill_reports_prefill_rate_not_zero():
    """While the prompt is being processed the prefill rate must be visible."""
    c = tui.Collector("http://127.0.0.1:1")
    c.derive_slots([slot(processed=0, decoded=0, ident=7)], now=2000.0)
    r = c.derive_slots([slot(processed=600, decoded=0, ident=7)], now=2000.5)
    assert r["prefill_s"] == pytest.approx(1200.0, rel=1e-6)
    assert r["tok_s"] == 0.0


def test_zero_time_delta_does_not_divide_by_zero():
    c = tui.Collector("http://127.0.0.1:1")
    c.derive_slots([slot(decoded=10, ident=3)], now=5.0)
    r = c.derive_slots([slot(decoded=20, ident=3)], now=5.0)
    assert r["tok_s"] == 0.0


# ---------------------------------------------------------------------- charts
@pytest.mark.parametrize("values,expect", [
    ([], "\u00b7" * 8),
    ([100.0], "\u2588"),
    ([0.0, 100.0], "\u2581\u2588"),
])
def test_sparkline_shape(values, expect):
    assert tui.spark(values, 20, 0.0, 100.0) == expect


def test_sparkline_respects_max_width():
    assert len(tui.spark(list(range(300)), 20, 0, 300)) == 20


def test_bar_clamps_out_of_range():
    assert tui.bar(-0.05, 10).plain.count("\u2588") == 0
    assert tui.bar(5.0, 10).plain.count("\u2588") == 10


# ------------------------------------------------------------------ query log
def rec(**kw):
    """A query record in the shape HistoryFile.append() expects."""
    r = {"t0": time.time(), "elapsed": 2.0, "prompt": 10, "cached": 0,
         "decode_tokens": 20, "prefill_s": 5.0, "gen": [10.0]}
    r.update(kw)
    return r


def test_history_roundtrip_and_duration_compat(tmp_path):
    path = tmp_path / "hist.jsonl"
    h = tui.HistoryFile(str(path))
    h.append(rec())
    # a line written by an older build used "duur"
    path.write_text(path.read_text() + json.dumps(
        {"ts": time.time(), "duur": 3.0, "prompt": 5, "cache": 0, "out": 1,
         "prefill_s": 0, "gen_s": 0}) + "\n")
    rows = tui.normalise(h.read())
    assert [r["duration"] for r in rows] == [2.0, 3.0]


def test_history_skips_corrupt_lines(tmp_path):
    path = tmp_path / "hist.jsonl"
    path.write_text("not json\n" + json.dumps({"ts": 1, "duration": 1}) + "\n")
    assert len(tui.HistoryFile(str(path)).read()) == 1


def test_history_keeps_last_lines_only(tmp_path):
    path = tmp_path / "hist.jsonl"
    h = tui.HistoryFile(str(path), max_lines=5)
    for i in range(1200):  # trim runs every 200 appends
        h.append(rec(t0=i))
    assert len(path.read_text().splitlines()) <= 5


def test_stats_output(tmp_path, capsys):
    path = tmp_path / "hist.jsonl"
    h = tui.HistoryFile(str(path))
    for i in range(4):
        h.append(rec(t0=1795600000 + i * 60, elapsed=2.0 + i))
    tui.print_stats(str(path))
    out = capsys.readouterr().out
    assert "4 queries between" in out
    assert "duration" in out and "tok/s" in out
    assert "per hour" in out


def test_stats_on_missing_file_is_quiet(tmp_path, capsys):
    tui.print_stats(str(tmp_path / "nope.jsonl"))
    assert "no queries" in capsys.readouterr().out


def test_sample_always_returns_a_dict():
    """`pump()` in the UI calls `.get()` on the result: a None would crash it."""
    import asyncio
    import httpx

    slots = [slot(processed=50, decoded=0, prompt=50, ident=1)]
    props = {"model_alias": "m", "total_slots": 1,
             "default_generation_settings": {"n_ctx": 4096}}

    def handler(request):
        body = slots if request.url.path.endswith("/slots") else props
        return httpx.Response(200, json=body)

    c = tui.Collector("http://127.0.0.1:1")
    c.client = httpx.AsyncClient(base_url=c.url, transport=httpx.MockTransport(handler))
    d = asyncio.run(c.sample())
    assert isinstance(d, dict) and d["server"] is True and d["model"] == "m"


def test_sensor_lookup_without_board_specifics():
    """A machine whose sensor labels we do not know must still show readings."""
    sens = {
        "amdgpu-pci-0b00": {
            "edge": {"temp1_input": 49.0},
            "fan1": {"fan1_input": 1200.0},
            "PPT": {"power1_average": 51.0},
        },
        "k10temp-pci-00c3": {"Tctl": {"temp1_input": 61.5}, "Tccd1": {"temp3_input": 44.0}},
        "it87-isa-0a10": {"SYSFAN1": {"fan2_input": 1500.0}, "SYSFAN2": {"fan3_input": 900.0}},
    }
    assert tui.pick(sens, "", "Tctl", r"Tctl") == 61.5
    assert tui.pick(sens, "k10temp", "Tctl", r"Tctl") == 61.5
    assert tui.pick(sens, "nct6686", "Tctl", r"Tctl") is None  # wrong chip filter
    assert tui.pick_first_fan(sens) == 1500.0  # fastest fan, any chip


def test_default_url_env_wins(monkeypatch):
    monkeypatch.setenv("LLAMA_URL", "http://127.0.0.1:9999/")
    assert tui.default_url() == "http://127.0.0.1:9999"
    assert tui.Collector().url == "http://127.0.0.1:9999"


def test_default_url_picks_a_port_that_is_listening(monkeypatch):
    monkeypatch.delenv("LLAMA_URL", raising=False)
    monkeypatch.setattr(tui, "_port_open", lambda port, host="127.0.0.1": port == 8090)
    assert tui.default_url() == "http://127.0.0.1:8090"


def test_default_url_falls_back_to_llama_server_port(monkeypatch):
    monkeypatch.delenv("LLAMA_URL", raising=False)
    monkeypatch.setattr(tui, "_port_open", lambda port, host="127.0.0.1": False)
    assert tui.default_url() == tui.DEFAULT_URL == "http://127.0.0.1:8090"


def test_parse_num_cu_from_vulkaninfo_summary():
    out = ("GPU0:\n\tapiVersion = 1.3.0\n\tdeviceName = Some GPU (RADV)\n"
           "\tnum_cu = 40\n\tnum_cu_per_sh = 10\n")
    assert tui._parse_num_cu(out) == 40
    assert tui._parse_num_cu("") is None
    assert tui._parse_num_cu("no vulkan here") is None


def test_vulkan_cu_survives_missing_vulkaninfo(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("vulkaninfo")
    monkeypatch.setattr(tui.subprocess, "run", boom)
    assert tui.vulkan_cu() == (None, "")


def test_vulkan_cu_reads_device_name(monkeypatch):
    class R:
        stdout = "deviceName = Fake GPU (RADV)\n\tnum_cu = 24\n"

    def fake(*a, **kw):
        return R()
    monkeypatch.setattr(tui.subprocess, "run", fake)
    assert tui.vulkan_cu() == (24, "Fake GPU (RADV)")


def test_cu_total_env_overrides_vulkan(monkeypatch):
    monkeypatch.setattr(tui, "CU_TOTAL_ENV", "32")
    monkeypatch.setattr(tui.Collector, "now", lambda self: 1000.0)
    c = tui.Collector()
    c.cu_total, c.cu_cached_at = None, None
    monkeypatch.setattr(tui, "vulkan_cu", lambda *a, **kw: (40, "should not be used"))
    if c.cu_total is None or c.cu_cached_at is None or 1000.0 - c.cu_cached_at > 60:
        c.cu_total, c.gpu_name = int(tui.CU_TOTAL_ENV), "configured"
        c.cu_cached_at = 1000.0
    assert c.cu_total == 32


def test_gpu_panel_shows_cu_count():
    from rich.text import Text
    d = {"cu_total": 40, "gpu_name": "AMD BC-250 (RADV GFX1013)", "cores": [1.0]}
    t = Text()
    cu = d.get("cu_total")
    t.append_text(Text.assemble(("  compute units ", "bold"),
                                (f"{cu} CU" if cu else "n/a", "green" if cu else "grey50")))
    assert "40 CU" in t.plain
