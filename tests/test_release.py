"""Release invariants: the version, the README and the licence must agree.

A release is a number in one place and a claim in several others; these tests
are what keeps the copies from drifting apart after a bump.
"""

import re
from pathlib import Path

import llamamon

ROOT = Path(__file__).resolve().parents[1]
V = llamamon.__version__


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", V)


def test_pyproject_version_matches_package():
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version = "([^"]+)"', text, re.M)
    assert m and m.group(1) == V


def test_readme_states_version_and_changelog_row():
    readme = (ROOT / "README.md").read_text()
    assert f"**Version {V}**" in readme
    assert re.search(r"^\| " + re.escape(V) + r" \|", readme, re.M)


def test_licence_is_agpl():
    first = (ROOT / "LICENSE").read_text().splitlines()[0]
    assert first.strip() == "GNU AFFERO GENERAL PUBLIC LICENSE"
    assert "MIT License" not in (ROOT / "README.md").read_text()


def test_no_private_hosts_in_tree():
    """The public tree must not name LAN addresses or internal hosts."""
    bad = re.compile(r"192\.168\.|gitlab\.waarland|10\.0\.0\.")
    for p in ROOT.rglob("*"):
        if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts \
                and p.suffix in {".py", ".md", ".toml", ".sh", ""}:
            assert not bad.search(p.read_text(errors="replace")), p
