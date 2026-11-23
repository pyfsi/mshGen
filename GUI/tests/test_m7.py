'''
test_m7.py: the block optimiser of step 3b (core/blockopt.py, docs/block_optimiser.md)

On new wings (FFA_W3_211, NACA0012; created in a temporary folder):
  - the optimiser's own vertex computation equals mshGen's vertices in every slice
  - the suggestion (priority high) lowers the objective, passes mshGen's block check and has fewer cell faces
    above 40 deg (current cells) than mshGen's default vertices
  - with --openfoam: the default and the suggested case are built (blockMesh, checkMesh) with the same cells;
    the suggestion must give a valid mesh with a lower average non-orthogonality (only the aspect ratio check may
    fail: the wall segments change length with the same cell counts, step 4 adapts the cells)
Run (Anaconda module loaded):  python GUI/tests/test_m7.py [--openfoam] [--keep]
'''
import copy
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("MPLBACKEND", "Agg")
from core import blockcheck, blockopt, estimate, geometry  # noqa: E402
from core.airfoils import AirfoilData  # noqa: E402
from core.runner import checkmesh_summary, run_allrun  # noqa: E402
from core.writer import write_case  # noqa: E402

FOCUS = "high (0.25 chord)"


def build(folder, case):
    case.environment.fluent = False
    write_case(folder, case)
    lines = []
    code = run_allrun(folder, on_line=lines.append)
    assert code == 0, "\n".join(lines[-15:])
    s = checkmesh_summary(folder)
    assert s["cells"] == estimate.cell_count(case), (s["cells"], estimate.cell_count(case))
    return s


def main():
    openfoam, keep = "--openfoam" in sys.argv, "--keep" in sys.argv
    tmp = tempfile.mkdtemp(prefix="mshgen_m7_")
    try:
        for name in ("FFA_W3_211", "NACA0012"):
            c = geometry.new_wing_case("m7_" + name, name, cases_dir=tmp)
            f = os.path.join(tmp, c.name, c.airfoil.file)
            a = AirfoilData(c.airfoil.name, f)
            # 1. own vertices == mshGen's
            v = blockcheck.slice_vertices(c, f)
            err = 0.0
            for k in range(len(c.blocks.slice_locations)):
                sl = blockopt._Slice(c, a, k, False)
                xz = blockopt.slice_vertices(a, c.blocks.interior_points, c.blocks.exterior_points, sl)
                err = max(err, float(np.abs(xz - v[k][:, [0, 2]]).max()))
            assert err < 1e-9, err
            # 2. suggestion
            r = blockopt.suggest(c, a, f, focus=FOCUS)
            b, n = r["before"], r["after"]
            assert n["objective (smaller is better)"] < b["objective (smaller is better)"]
            share = "cells (current counts and gradings): faces above 40 deg (%)"
            assert n[share] <= b[share], (b[share], n[share])
            print("OK    {:11s} vertices = mshGen (max diff {:.1e}); objective {:.2f} -> {:.2f}; faces above 40 deg "
                  "{:.1f} % -> {:.1f} %; wall rows mean {:.1f} -> {:.1f} deg ({})".format(
                      name, err, b["objective (smaller is better)"], n["objective (smaller is better)"], b[share],
                      n[share], b["cells, first 5 rows at the wall: mean non-orthogonality (deg)"],
                      n["cells, first 5 rows at the wall: mean non-orthogonality (deg)"], r["note"]))
            if openfoam:
                s0 = build(os.path.join(tmp, c.name), c)
                d = copy.deepcopy(c)
                d.name = c.name + "_suggested"
                d.blocks.interior_points, d.blocks.exterior_points = r["interior_points"], r["exterior_points"]
                d.blocks.vertex_source, d.blocks.defaults_key = "custom", None
                folder = os.path.join(tmp, d.name)
                os.makedirs(folder)
                shutil.copyfile(f, os.path.join(folder, d.airfoil.file))
                s1 = build(folder, d)
                print("      checkMesh {:,} cells: non-orthogonality max {:.1f} -> {:.1f}, average {:.2f} -> {:.2f}; "
                      "skewness {:.2f} -> {:.2f}; {}".format(
                          s0["cells"], s0["max_non_orthogonality"], s1["max_non_orthogonality"],
                          s0["avg_non_orthogonality"], s1["avg_non_orthogonality"], s0["max_skewness"],
                          s1["max_skewness"], "mesh OK" if s1["mesh_ok"] else "; ".join(s1["warnings"])))
                assert s1["avg_non_orthogonality"] < s0["avg_non_orthogonality"]
                # same cell counts on wall segments of a new length: the aspect ratio check may fail until the
                # chordwise cells are adapted in step 4; any other failed check is an error
                other = [w for w in s1["warnings"] if "aspect ratio" not in w.lower()]
                assert not other, other
        print("OK    block optimiser")
    finally:
        if keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
