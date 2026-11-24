'''
test_m9.py: step 5 - the cells that fail checkMesh or have a lower quality, and the AI summary

On a new NACA0012 wing (mshGen's default cells, created and built in a temporary folder):
  - Allrun.pre writes the quality fields (nonOrthoAngle, skewness) and the cell centres (one per cell)
  - core/meshquality.analyse: one group per set written by checkMesh, with the set's size; a set of a failed check
    is "failed"; the lower-quality groups hold cells in their range (non-orthogonality 50-70 deg, skewness 2-4)
  - core/runner.run_quality reproduces the fields and centres of an older build (fields removed)
  - core/aisummary.mesh_facts: serialisable, without the case name, the folder or the user name
  - the Claude Code CLI is found; a CLI that fails gives a readable error (a stand-in binary: nothing is sent)
  - with --live (calls Claude through the Claude Code CLI with your Claude login, sending the facts): a summary with
    the five sections that ends with the closing sentence
Run (Anaconda module loaded; OpenFOAM via Allrun.pre):  python GUI/tests/test_m9.py [--live] [--keep]
'''
import getpass
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("MPLBACKEND", "Agg")
from core import aisummary, geometry, meshquality  # noqa: E402
from core.results import _numbers, measured_yplus  # noqa: E402
from core.runner import checkmesh_summary, run_allrun, run_quality  # noqa: E402
from core.writer import write_case  # noqa: E402

SECTIONS = ["### What is good", "### Suggested modelling approach", "### What can be improved",
            "### Things to consider and assumptions", "### A crazy idea"]


def check_groups(folder):
    groups = meshquality.analyse(folder)
    assert groups is not None
    sets_dir = os.path.join(folder, "constant", "polyMesh", "sets")
    names = sorted(os.listdir(sets_dir)) if os.path.isdir(sets_dir) else []
    by = {g["name"]: g for g in groups}
    for n in names:
        size = len(_numbers(os.path.join(sets_dir, n)))
        if size:
            assert by[n]["count"] == size, (n, by[n]["count"], size)
    s = checkmesh_summary(folder)
    if any("aspect ratio" in w.lower() for w in s["warnings"]):
        assert by["highAspectRatioCells"]["status"] == "failed"
    for g in groups:
        if g["status"] == "lower quality":
            lo, hi = {"nonOrthoAngle": (50, 70), "skewness": (2, 4)}[g["name"]]
            assert np.all((g["value"] >= lo) & (g["value"] < hi)), g["name"]
        assert len(g["x"]) <= meshquality.MAX_POINTS
    return groups


def main():
    live, keep = "--live" in sys.argv, "--keep" in sys.argv
    tmp = tempfile.mkdtemp(prefix="mshgen_m9_")
    try:
        c = geometry.new_wing_case("m9_quality", "NACA0012", cases_dir=tmp)
        folder = os.path.join(tmp, c.name)
        c.environment.fluent = False
        write_case(folder, c)
        t0 = time.time()
        lines = []
        assert run_allrun(folder, on_line=lines.append) == 0, "\n".join(lines[-15:])
        s = checkmesh_summary(folder)
        assert meshquality.available(folder)
        C = _numbers(os.path.join(folder, "constant", "nonOrthoAngle"), "internalField")
        assert len(C) == s["cells"], (len(C), s["cells"])
        groups = check_groups(folder)
        print("OK    build with quality fields ({:.0f} s, {:,} cells): ".format(time.time() - t0, s["cells"])
              + "; ".join("{} {} {:,}".format(g["status"], g["name"], g["count"]) for g in groups))
        # an older build: no fields -> run_quality
        for f in ("C", "nonOrthoAngle", "skewness", "aspectRatio"):
            os.remove(os.path.join(folder, "constant", f))
        assert not meshquality.available(folder)
        assert run_quality(folder, c.environment.openfoam_module, on_line=lambda line: None) == 0
        again = check_groups(folder)
        assert [(g["name"], g["count"]) for g in again] == [(g["name"], g["count"]) for g in groups]
        print("OK    run_quality on an older build: same groups")
        # facts: no names or paths
        facts = aisummary.mesh_facts(c, folder, os.path.join(folder, c.airfoil.file), groups, measured_yplus(folder, c))
        text = json.dumps(facts)
        for private in (c.name, tmp, folder, getpass.getuser(), os.path.expanduser("~")):
            assert private not in text, private
        print("OK    facts for the AI summary: {} characters, no case name, folder or user name".format(len(text)))
        # Claude Code CLI: found, and a failing CLI gives a readable error (stand-in binary, nothing is sent)
        found, status = aisummary.cli_status()
        print("OK    " + status if found else "--    " + status)
        try:
            aisummary.write_summary(facts, binary=shutil.which("false"))
        except RuntimeError as e:
            print("OK    a failing CLI: " + str(e))
        else:
            raise AssertionError("write_summary should fail with a failing CLI")
        if live:
            text, info = aisummary.write_summary(facts)
            missing = [h for h in SECTIONS if h not in text]
            assert not missing, missing
            assert text.endswith(aisummary.ENDING)
            print("OK    AI summary ({} via {}; {} input, {} output tokens):\n{}".format(
                info["model"], info["via"], info["input tokens"], info["output tokens"], text))
        print("OK    step 5 quality and AI summary")
    finally:
        if keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
