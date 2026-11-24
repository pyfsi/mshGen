'''
test_m8.py: the mesh optimiser of step 4 (core/meshopt.py, docs/mesh_optimiser.md)

  - the edge -> block resolution/grading port equals mshGen's blocks (random edges and layer gradings)
  - on new wings (NACA0012, FFA_W3_211; created in a temporary folder) and a copy of MegAWES_wing, level COARSE
    (NACA0012 also MEDIUM): every target is met (y+ median and tip face, chordwise and spanwise counts in their
    ranges, cells <= max, growth ratio <= max wall-normal / chordwise / wake line / spanwise), the quality is not
    worse than mshGen's default cells (mean non-orthogonality, share of faces above 40 deg) and the maximum
    non-orthogonality stays below 70 deg; no slice has folded cells; beyond the tip the first cell inside the airfoil
    contour (e08) is within the growth ratio of the wall-normal first cell outside it
  - with --openfoam: the optimised NACA0012 and FFA_W3_211 meshes (COARSE) are built (blockMesh, checkMesh): cells
    = estimate, checkMesh passes except possibly the aspect ratio check (wall cells for y+ against the spanwise cell
    size), y+ measured on the mesh (wall distance) within 15 % of the target
Run (Anaconda module loaded):  python GUI/tests/test_m8.py [--openfoam] [--keep]
'''
import copy
import os
import random
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("MPLBACKEND", "Agg")
from core import estimate, geometry, meshopt  # noqa: E402
from core.airfoils import AirfoilData  # noqa: E402
from core.case import Case  # noqa: E402
from core.model import mshgen_mesh  # noqa: E402
from core.results import measured_yplus  # noqa: E402
from core.runner import checkmesh_summary, run_allrun  # noqa: E402
from core.writer import write_case  # noqa: E402


def test_port(c, f):
    random.seed(3)
    for _ in range(3):
        cc = copy.deepcopy(c)
        cc.meshing.edges = {k: [random.randint(2, 30), random.uniform(0.1, 10), random.uniform(0.1, 10)]
                            for k in cc.meshing.edges}
        cc.meshing.layer_grading = [random.uniform(0.2, 5) for _ in cc.meshing.layer_grading]
        ref = mshgen_mesh(cc, f, upto="edges")
        m = mshgen_mesh(c, f, upto="edges")
        meshopt.set_model_edges(m, cc)
        for b1, b2 in zip(ref.blocks, m.blocks):
            assert list(b1.resolution) == list(b2.resolution), (b1.resolution, b2.resolution)
            assert np.allclose(b1.grading, b2.grading, rtol=1e-12), (b1.grading, b2.grading)
    print("OK    edge -> block port = mshGen ({} blocks)".format(len(ref.blocks)))


def row(rows, start):
    return next(r for r in rows if r["metric"].startswith(start))


def check(name, c, f, level):
    c.domain.mesh_level, c.meshing.targets = level, None
    t = meshopt.targets(c)
    a = AirfoilData(c.airfoil.name, f)
    t0 = time.time()
    r = meshopt.optimise(c, f, a, t)
    dt = time.time() - t0
    d = copy.deepcopy(c)
    d.meshing.edges = r["edges"]
    if r["layers"]:
        d.meshing.layer_resolution, d.meshing.layer_grading = list(r["layers"][0]), list(r["layers"][1])
    after, before = r["after"], r["before"]
    failed = [x["metric"] for x in after if x["ok"] is False]
    assert not failed, (name, level, failed)
    assert abs(row(after, "y+ on the wing, median")["value"] / t["yplus"] - 1) < 0.02
    if d.dim == "3D":
        assert abs(row(after, "y+ on the tip face")["value"] / t["yplus"] - 1) < 0.05
    for k in ("wall-normal", "chordwise", "wake line", "spanwise"):
        v = row(after, "largest growth ratio, " + k)["value"]
        assert v <= t["growth"] * 1.02, (k, v)
    assert estimate.cell_count(d) <= t["max_cells"]
    m0, m1 = (row(x, "non-orthogonality, mean")["value"] for x in (before, after))
    a0, a1 = (row(x, "faces above 40 deg")["value"] for x in (before, after))
    nmax = row(after, "non-orthogonality, max")["value"]
    assert m1 <= m0 + 0.5 and a1 <= a0 + 0.5 and nmax < 70, (m0, m1, a0, a1, nmax)
    ratio = meshopt.contour_ratio(d, f, mshgen_mesh(d, f, upto="edges"))
    if ratio is not None:                       # beyond the tip: e08 starts like the wall-normal cells outside
        assert 1 / t["growth"] <= ratio * 1.02 and ratio <= t["growth"] * 1.02, ratio
    rows, warn = meshopt.slice_table(d, f, mshgen_mesh(d, f, upto="edges"), t)
    assert not [w for k, w in warn if k == "danger"], warn
    print("OK    {:11s} {:6s} {:5.0f} s  cells {:,}; y+ {:.3g} (target {:g}); non-orth. mean {:.1f} -> {:.1f}, faces "
          "> 40 deg {:.1f} % -> {:.1f} %, max {:.1f}; e08 {} (contour ratio {}); {}".format(
              name, level, dt, estimate.cell_count(d), row(after, "y+ on the wing, median")["value"], t["yplus"],
              m0, m1, a0, a1, nmax, d.meshing.edges["e08"][:2],
              "{:.2f}".format(ratio) if ratio is not None else "-", r["note"]))
    return d


def build(folder, case, airfoil_src):
    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(airfoil_src, os.path.join(folder, case.airfoil.file))
    case.environment.fluent = False
    write_case(folder, case)
    lines = []
    code = run_allrun(folder, on_line=lines.append)
    assert code == 0, "\n".join(lines[-15:])
    s = checkmesh_summary(folder)
    assert s["cells"] == estimate.cell_count(case), (s["cells"], estimate.cell_count(case))
    other = [w for w in s["warnings"] if "aspect ratio" not in w.lower()]
    assert not other, other
    y = measured_yplus(folder, case)
    t = meshopt.targets(case)["yplus"]
    for patch, v in y.items():
        assert abs(v["y+ median"] / t - 1) < 0.15, (patch, v)
    print("      checkMesh {:,} cells: non-orthogonality max {:.1f}, average {:.2f}; skewness {:.2f}; {}; y+ median {}"
          .format(s["cells"], s["max_non_orthogonality"], s["avg_non_orthogonality"], s["max_skewness"],
                  "mesh OK" if s["mesh_ok"] else "; ".join(s["warnings"]),
                  ", ".join("{} {:.1f}".format(k, v["y+ median"]) for k, v in y.items())))


def main():
    openfoam, keep = "--openfoam" in sys.argv, "--keep" in sys.argv
    tmp = tempfile.mkdtemp(prefix="mshgen_m8_")
    try:
        built = []
        for name, levels in (("NACA0012", ("COARSE", "MEDIUM")), ("FFA_W3_211", ("COARSE",))):
            c = geometry.new_wing_case("m8_" + name, name, cases_dir=tmp)
            f = os.path.join(tmp, c.name, c.airfoil.file)
            if name == "NACA0012":
                test_port(c, f)
            for level in levels:
                d = check(name, c, f, level)
                if level == "COARSE":
                    built.append((d, f))
        src = os.path.join(REPO, "cases", "MegAWES_wing")
        c = Case.load(src)
        check("MegAWES", c, os.path.join(src, c.airfoil.file), "COARSE")
        if openfoam:
            for d, f in built:
                d.name = d.name + "_opt"
                build(os.path.join(tmp, d.name), d, f)
        print("OK    mesh optimiser")
    finally:
        if keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
