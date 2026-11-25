# Changelog

Versions follow [semantic versioning](https://semver.org/).

## 1.0.0

First release.

- GPU / thermal panel: GPU busy (or busiest core), per-core load, GTT and RAM
  usage, swap, CPU and load average, edge/Tctl/TSI temperatures, fan speed,
  shader clock, package power, vddgfx.
- llama-server panel: online/offline, model, context size, uptime, RSS, client
  count, the query being processed with elapsed time, live tok/s and prefill
  rate, prompt/cache usage, output progress.
- Sparkline charts over the last two minutes: GPU, tok/s, RAM, CPU, temperature,
  power.
- Query log (`~/.llamamon-history.jsonl`) with p50/p90/max per hour via
  `llamamon --stats`.
- Everything configurable through environment variables; no board or GPU model
  is hardcoded.
- `install.sh` for a root-free install into `~/.venvs/llamamon` plus
  `~/.local/bin`.
