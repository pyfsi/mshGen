'''
test_m3.py: M3 tests (mesh preview against blockMesh)

For a slice, every node of the Python preview is compared with the nearest blockMesh point in the same
plane (constant y); the distance is divided by the local cell size (smallest edge of the adjacent preview
cells). Also the number of cells in the slice must equal the faces of symRoot (root slice) / symTip.

Cases: cases/MegAWES_wing (if its polyMesh exists: root, a wing slice, tip), and with --openfoam new test
wings (NACA0012 straight, NACA0012 swept/tapered/twisted with dihedral, Mrev-v2 with an aileron layer).

Run (Anaconda module loaded):  python GUI/tests/test_m3.py [--openfoam]
'''
import os
import re
import shutil
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from core import geometry, preview  # noqa: E402
from core.case import Case  # noqa: E402
from core.model import mshgen_mesh  # noqa: E402
from core.planform import Planform  # noqa: E402
from core.runner import checkmesh_summary, run_allrun  # noqa: E402
from core.writer import write_case  # noqa: E402


def read_points(folder):
    '''constant/polyMesh/points (ASCII) -> (n, 3) array.'''
    t = open(os.path.join(folder, "constant", "polyMesh", "points")).read()
    t = re.sub(r"FoamFile\s*\{.*?\}", "", t, flags=re.S)
    t = re.sub(r"//.*", "", t)
    m = re.search(r"(\d+)\s*\(", t)
    n = int(m.group(1))
    body = t[m.end():].translate(str.maketrans("()", "  "))
    return np.fromstring(body, sep=" ", count=3 * n).reshape(n, 3)


def compare_slice(case, folder, points, s, mesh):
    y = case.blocks.slice_locations[s]
    plane = points[np.abs(points[:, 1] - y) < 1e-6 * (1 + abs(y))]
    blocks = preview.slice_mesh(case, os.path.join(folder, case.airfoil.file), s, mesh)
    tree = cKDTree(plane[:, [0, 2]])
    rel = []
    for b in blocks:
        P = b["P"][..., [0, 2]]
        dx = np.linalg.norm(np.diff(P, axis=0), axis=-1)
        dz = np.linalg.norm(np.diff(P, axis=1), axis=-1)
        loc = np.full(P.shape[:2], np.inf)
        loc[:-1] = np.minimum(loc[:-1], dx)
        loc[1:] = np.minimum(loc[1:], dx)
        loc[:, :-1] = np.minimum(loc[:, :-1], dz)
        loc[:, 1:] = np.minimum(loc[:, 1:], dz)
        d, _ = tree.query(P.reshape(-1, 2))
        rel.append(d / loc.ravel())
    rel = np.concatenate(rel)
    cells = int(sum((b["P"].shape[0] - 1) * (b["P"].shape[1] - 1) for b in blocks))
    return {"slice": s, "y": y, "nodes": len(rel), "plane points": len(plane), "cells": cells,
            "max dist/cell": float(rel.max()), "p99": float(np.percentile(rel, 99)),
            "median": float(np.median(rel))}


def check_case(folder, slices=None):
    case = Case.load(folder)
    af = os.path.join(folder, case.airfoil.file)
    mesh = mshgen_mesh(case, af, upto="edges")
    pts = read_points(folder)
    s = checkmesh_summary(folder)
    n = len(case.blocks.slice_locations)
    slices = slices if slices is not None else range(n)
    rows = [compare_slice(case, folder, pts, k, mesh) for k in slices]
    for r in rows:
        if r["slice"] == 0 and s:
            assert r["cells"] == s["patches"]["symRoot"], (r["cells"], s["patches"])
        if r["slice"] == n - 1 and s:
            assert r["cells"] == s["patches"]["symTip"], (r["cells"], s["patches"])
    return rows


def build_wing(cases, name, airfoil, planform=None, aileron=None):
    c = geometry.new_wing_case(name, airfoil, cases_dir=cases, planform=planform)
    c.options.aileron_layers = aileron
    c.environment.fluent = False
    folder = os.path.join(cases, name)
    write_case(folder, c)
    lines = []
    assert run_allrun(folder, on_line=lines.append) == 0, "\n".join(lines[-15:])
    return folder


def show(name, rows, tol=0.05):
    worst = max(r["max dist/cell"] for r in rows)
    print("{}  {:22s} slices {}: max deviation {:.3g} x local cell size (99 %: {:.3g}, median {:.2g})".format(
        "OK  " if worst < tol else "WARN", name, len(rows), worst, max(r["p99"] for r in rows),
        max(r["median"] for r in rows)))
    for r in rows:
        if r["max dist/cell"] >= tol:
            print("        slice {} (y = {:.4g}): max {:.3g}, 99 % {:.3g}".format(r["slice"], r["y"], r["max dist/cell"], r["p99"]))
    return worst


if __name__ == "__main__":
    repo = os.path.dirname(os.path.dirname(HERE))
    meg = os.path.join(repo, "cases", "MegAWES_wing")
    if os.path.isfile(os.path.join(meg, "constant", "polyMesh", "points")):
        show("MegAWES_wing", check_case(meg, slices=[0, 1, 3, 5]))
    if "--openfoam" in sys.argv:
        cases = tempfile.mkdtemp(prefix="mshgen_m3_")
        try:
            show("NACA0012 straight", check_case(build_wing(cases, "straight", "NACA0012")))
            p = Planform(half_span=8, root_chord=1.5, tip_chord=0.6, kink_y=1.5, le_sweep_deg=10,
                         twist_root_deg=2, twist_tip_deg=-3, dihedral_deg=6)
            show("NACA0012 swept", check_case(build_wing(cases, "swept", "NACA0012", p)))
            show("Mrev-v2 aileron", check_case(build_wing(cases, "aileron", "Mrev-v2", aileron=[0])))
        finally:
            shutil.rmtree(cases, ignore_errors=True)
