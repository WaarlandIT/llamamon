# docs

`screenshot.svg` is rendered from fixture data with a frozen clock, so it can be
regenerated without a GPU, a llama-server or lm-sensors:

    .venv/bin/python mk_screenshot.py      # writes docs/screenshot.svg

It is an SVG (not a PNG) because that stays sharp at any size and GitHub renders
it directly in the README.
