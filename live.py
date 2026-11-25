"""Ask the running llama-server a few questions and watch the collector follow.

Run on the host that runs llama-server:   python live.py [url]
It prints one line per phase (idle / prefill / generate / done) with what the
collector reports, so a change in the math is visible without the TUI.
"""

import asyncio
import json
import sys
import time

import httpx

sys.path.insert(0, ".")
from llamamon import tui  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
ASK = ("In about four paragraphs, explain what a reverse proxy firewall does, "
       "how it decides to block a request, and which signals it uses. Be concrete.")


async def stream_questions():
    async with httpx.AsyncClient(timeout=None) as cl:
        body = {"prompt": ASK, "n_predict": 300, "temperature": 0.2}
        async with cl.stream("POST", URL + "/completion", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    try:
                        j = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if j.get("stop"):
                        return


async def main():
    col = tui.Collector(URL, 0.5)
    task = asyncio.create_task(stream_questions())
    print(f"{'t':>5} {'state':<9} {'proc':<5} {'tok/s':>7} {'prefill':>8} "
          f"{'prompt':>7} {'out':>5} {'gpu%':>5} {'watt':>6} {'temp':>5}")
    t0 = time.time()
    while not task.done() or time.time() - t0 < 3:
        d = await col.sample()
        req = d.get("req") or {}
        print(f"{time.time()-t0:5.1f} {'busy' if d.get('processing') else 'idle':<9} "
              f"{str(bool(req)):<5} {(d.get('tok_s') or 0):7.1f} {(d.get('prefill_s') or 0):8.1f} "
              f"{req.get('prompt', 0):7} {req.get('decode_tokens', 0):5} "
              f"{d.get('gpu_busy', 0):5.1f} {(d.get('power') or 0):6.1f} "
              f"{(d.get('temp_edge') or 0):5.1f}")
        await asyncio.sleep(0.5)
    await task


asyncio.run(main())
