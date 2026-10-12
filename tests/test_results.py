"""The committed results are consistent with each other and with the README."""
import json
import re

from rl_lab.config import REPO_ROOT
from rl_lab.summarize import markdown_table

README = (REPO_ROOT / "README.md").read_text()
SUMMARY = json.loads((REPO_ROOT / "results" / "summary.json").read_text())


def test_readme_table_is_summarize_output():
    assert SUMMARY["experiments"], "results/summary.json has no experiments"
    table = markdown_table(SUMMARY["experiments"])
    assert table in README, "README results table must be the verbatim output of rl_lab.summarize"


def test_readme_figures_exist():
    refs = re.findall(r"\]\((results/[^)]+\.png)\)", README)
    assert refs, "README references no results/*.png"
    for ref in refs:
        assert (REPO_ROOT / ref).is_file(), f"README references missing figure {ref}"
