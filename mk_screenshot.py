"""Render a deterministic screenshot of llamamon into docs/screenshot.svg.

The screen is produced from fixture data with a frozen clock, so the image in
the README can be regenerated without a GPU, a llama-server or lm-sensors:

    python mk_screenshot.py
"""

import asyncio
import io
import os

os.environ.setdefault("LLAMAMON_HISTORY", "/tmp/llamamon-shot.jsonl")

from rich.console import Console

from llamamon import tui

GB = 1024 ** 3
DEMO = dict(
    ts=0, server=True, model="qwen3-30b-a3b", n_ctx=98304, slots_total=1, clients=2,
    gpu_busy=62.0, cores=[3, 8, 12, 71, 94, 5], gtt_used=int(11.84 * GB),
    gtt_total=int(14.60 * GB), mem_used=int(12.4 * GB), mem_total=int(14.6 * GB),
    swap_used=int(.2 * GB), swap_total=int(8 * GB), cpu=18.0, load=1.24,
    temp_edge=52.0, temp_tsi=54.0, temp_tctl=62.6, therm14=41.0, therm15=39.0,
    fan=1907, sclk=1000, power=50.1, vddgfx=993, tok_s=71.5, prefill_s=812.0,
    processing=True, server_up=11520, rss=int(1.9 * GB),
    req=dict(id=7, t0=0, prompt=1204, cached=224, max_tok=512, ctx=98304,
             prefill_s=812.0, gen=[71.5], decode_tokens=102, has_output=True,
             elapsed=4.2),
)


async def main():
    app = tui.MonitorApp("http://127.0.0.1:8090", 1.0)
    app.demo = DEMO  # frozen sample: no clock, no hardware read
    app.collector.now = lambda: 0.0
    async with app.run_test(size=(118, 40)) as pilot:
        g = app.query_one(tui.Graphs)
        for key, vals in [("gpu", [5, 20, 48, 62]), ("tok", [0, 40, 66, 71]),
                          ("mem", [80, 82, 84, 85]), ("cpu", [10, 14, 18, 18]),
                          ("temp", [46, 48, 50, 52]), ("power", [48, 50, 49, 50])]:
            g.hist[key].extend(vals)
        h = app.query_one(tui.History)
        for i in range(3):
            h.done.appendleft(dict(id=i, t0=0, prompt=1204 - i * 380, cached=224,
                                   max_tok=512, ctx=98304, prefill_s=812 - i * 30,
                                   gen=[71.5 - i], decode_tokens=102 + i * 88,
                                   has_output=True, elapsed=4.2 + i * 1.3))
        for cls in (tui.GpuPanel, tui.ServerPanel, tui.Graphs, tui.History):
            app.query_one(cls).refresh()
        await pilot.pause()
        with open("docs/screenshot.svg", "w") as fh:
            fh.write(app.export_screenshot())
        buf = io.StringIO()
        Console(file=buf, width=118, record=True).print(app.screen._compositor)
        with open("/tmp/shot.txt", "w") as fh:
            fh.write(buf.getvalue())
        print("saved docs/screenshot.svg")


asyncio.run(main())
