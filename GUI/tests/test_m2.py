'''
test_m2.py: M2 tests (domain and blocks)

  - slices: add at a cell face (cells unchanged) and remove again (layers restored), suggestions
  - edits on a new NACA0012 wing, each built with blockMesh/checkMesh when --openfoam is given:
      extra slice, moved chordwise division, dragged exterior vertices, R/c 5 -> 10 with recomputed
      vertices, far field moved, aileron cut-out on a Mrev-v2 wing
  - the block check agrees with blockMesh: a vertex move it reports as folding must make blockMesh fail

Run (Anaconda module loaded):  python GUI/tests/test_m2.py [--openfoam] [--keep]
'''
import copy
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from core import blockcheck, blocks, estimate, geometry  # noqa: E402
from core.case import Case  # noqa: E402
from core.runner import checkmesh_summary, run_allrun  # noqa: E402
from core.writer import write_case  # noqa: E402


def test_slices_roundtrip():
    c = Case.load(os.path.join(os.path.dirname(os.path.dirname(HERE)), "cases", "MegAWES_wing"))
    s0, r0, g0, n0 = list(c.blocks.slice_locations), list(c.meshing.layer_resolution), list(c.meshing.layer_grading), estimate.cell_count(c)
    for y in (2.0, 10.0, 20.0, 22.0, 30.0):
        d = copy.deepcopy(c)
        ynew = blocks.add_slice(d, y)
        k = d.blocks.slice_locations.index(ynew)
        assert estimate.cell_count(d) == n0
        blocks.remove_slice(d, k)
        assert d.blocks.slice_locations == s0 and d.meshing.layer_resolution == r0
        assert all(abs(a - b) <= 1e-9 * abs(b) for a, b in zip(d.meshing.layer_grading, g0)), (y, d.meshing.layer_grading)
    d = copy.deepcopy(c)
    s = blocks.suggest_slices(d)
    assert len(blocks.planform_kinks(d)) == 1 and s[0] == 0 and s[-1] == c.blocks.slice_locations[-1]


def build(folder, case, openfoam, expect_ok=True):
    case.environment.fluent = False
    write_case(folder, case)
    problems = blockcheck.check(case, os.path.join(folder, case.airfoil.file))
    if not openfoam:
        return ("blocks invalid" if problems else "blocks valid"), None
    lines = []
    code = run_allrun(folder, on_line=lines.append)
    if not expect_ok:
        assert problems, "block check should report this case"
        assert code != 0, "blockMesh should fail for this case"
        return "blockMesh failed as predicted", None
    assert not problems, problems
    assert code == 0, "\n".join(lines[-15:])
    s = checkmesh_summary(folder)
    assert s["cells"] == estimate.cell_count(case), (s["cells"], estimate.cell_count(case))
    return "built", s


def variants(cases):
    base = geometry.new_wing_case("base", "NACA0012", cases_dir=cases)
    out = [("base", base, True)]

    c = copy.deepcopy(base)
    blocks.add_slice(c, 2.0)
    out.append(("extra slice", c, True))

    c = copy.deepcopy(base)
    blocks.set_division(c, 1, suction=0.6, pressure=0.6)
    out.append(("division 2 at 0.6", c, True))

    c = copy.deepcopy(base)
    x, z = c.blocks.exterior_points[6]          # vertex 28 on the top boundary (stays before 27)
    blocks.move_exterior(c, 28, x + 0.15, z)
    x, z = c.blocks.exterior_points[9]          # vertex 31 on the C-arc, 15 deg up
    blocks.move_exterior(c, 31, -5.0, 1.3)
    out.append(("dragged 28 and 31", c, True))

    c = copy.deepcopy(base)
    blocks.set_mesh_scale(c, 10.0)
    geometry.recompute_vertices(c, os.path.join(cases, "base", "airfoil.txt"), reset_edges=False)
    out.append(("R/c 10, recomputed", c, True))

    c = copy.deepcopy(base)
    blocks.set_far_field(c, 15.0)
    out.append(("far field 15 m", c, True))

    m = geometry.new_wing_case("mrev", "Mrev-v2", cases_dir=cases)
    m.options.aileron_layers = [0]
    out.append(("Mrev-v2 aileron layer 0", m, True))

    c = copy.deepcopy(base)
    x, z = c.blocks.exterior_points[4]          # vertex 26 (top, above the TE) far upstream: blocks fold
    blocks.move_exterior(c, 26, -3.0, z)
    out.append(("26 dragged upstream (bad)", c, False))
    return out


if __name__ == "__main__":
    openfoam, keep = "--openfoam" in sys.argv, "--keep" in sys.argv
    cases = tempfile.mkdtemp(prefix="mshgen_m2_")
    try:
        test_slices_roundtrip()
        print("OK    slices: add at a cell face (cells unchanged), remove (layers restored), suggestions")
        for i, (name, case, ok) in enumerate(variants(cases)):
            folder = os.path.join(cases, "v{}".format(i))
            src = os.path.join(cases, "mrev" if "Mrev" in name else "base", "airfoil.txt")
            os.makedirs(folder, exist_ok=True)
            shutil.copyfile(src, os.path.join(folder, "airfoil.txt"))
            case.name = "v{}".format(i)
            result, s = build(folder, case, openfoam, expect_ok=ok)
            info = "" if s is None else "{:>9,} cells, max non-orth {:.1f}, max skew {:.2f}, {}".format(
                s["cells"], s["max_non_orthogonality"], s["max_skewness"],
                "mesh OK" if s["mesh_ok"] else "failed: " + "; ".join(s["warnings"]))
            if s is not None and "wingCut" in s["patches"]:
                info += ", wingCut {} faces".format(s["patches"]["wingCut"])
            print("OK    {:28s} {:30s} {}".format(name, result, info))
    finally:
        if keep:
            print("kept " + cases)
        else:
            shutil.rmtree(cases, ignore_errors=True)
