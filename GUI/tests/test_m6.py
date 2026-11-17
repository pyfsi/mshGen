'''
test_m6.py: M6 test, quality optimiser: "better or equal checkMesh results at the same cell count"

For MegAWES_wing (its existing build is the reference), a new FFA_W3_211 wing and a swept/tapered NACA0012
wing with dihedral: optimise with all variable groups, build the optimised case (and the reference for the
new wings) with blockMesh/checkMesh, and compare: same cells, non-orthogonality and skewness not worse,
no additional failed checks; also y+ unchanged (counts and wall/tip first cells are not touched).

Run (Anaconda module loaded; OpenFOAM via Allrun.pre):  python GUI/tests/test_m6.py [MegAWES|FFA|swept ...]
'''
import copy
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from core import geometry, quality, sizing  # noqa: E402
from core.case import Case  # noqa: E402
from core.planform import Planform  # noqa: E402
from core.runner import checkmesh_summary, run_allrun  # noqa: E402
from core.writer import write_case  # noqa: E402

ALL = ["grad1", "grad2", "wake", "span", "divisions", "exterior"]


def build(case, airfoil_src, folder):
    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(airfoil_src, os.path.join(folder, case.airfoil.file))
    case.environment.fluent = False
    case.name = os.path.basename(folder)
    write_case(folder, case)
    lines = []
    assert run_allrun(folder, on_line=lines.append) == 0, "\n".join(lines[-15:])
    return checkmesh_summary(folder)


def run(name, case, airfoil, work, reference=None):
    t = time.time()
    prop = quality.optimise(case, airfoil, ALL, weight_orth=0.5, max_evals=250)
    t_opt = time.time() - t
    opt = copy.deepcopy(case)
    sizing.apply_proposal(opt, prop)
    y0, y1 = sizing.predict(case, airfoil), sizing.predict(opt, airfoil)
    ref = reference or build(copy.deepcopy(case), airfoil, os.path.join(work, name + "_ref"))
    new = build(opt, airfoil, os.path.join(work, name + "_opt"))
    keys = ["non-orthogonality (deg)", "mean of the slice maxima of non-orthogonality (deg)", "skewness",
            "size ratio across blocks", "chordwise jump on the airfoil", "spanwise jump at a slice"]
    print("{}: optimiser {:.0f} s, {}".format(name, t_opt, prop["note"]))
    for k in keys:
        print("    {:52s} {:8.3f} -> {:8.3f}".format(k, prop["before"][k], prop["after"][k]))
    print("    y+ median {:.2f} -> {:.2f}, tip {} -> {}".format(
        y0["y+ median"], y1["y+ median"], round(y0.get("y+ on the tip face (spanwise first cell)", 0), 2),
        round(y1.get("y+ on the tip face (spanwise first cell)", 0), 2)))
    print("    checkMesh: cells {:,} -> {:,}; max non-orth {:.2f} -> {:.2f}; max skewness {:.3f} -> {:.3f}; "
          "failed checks {} -> {}; max aspect ratio {:.0f} -> {:.0f}".format(
              ref["cells"], new["cells"], ref["max_non_orthogonality"], new["max_non_orthogonality"],
              ref["max_skewness"], new["max_skewness"], ref["failed_checks"], new["failed_checks"],
              ref["max_aspect_ratio"], new["max_aspect_ratio"]))
    assert new["cells"] == ref["cells"]
    assert new["max_non_orthogonality"] <= ref["max_non_orthogonality"] + 0.05
    assert new["max_skewness"] <= ref["max_skewness"] + 0.01
    assert new["failed_checks"] <= ref["failed_checks"]
    assert abs(y1["y+ median"] - y0["y+ median"]) < 0.05 * y0["y+ median"]


if __name__ == "__main__":
    which = sys.argv[1:] or ["MegAWES", "FFA", "swept"]
    work = tempfile.mkdtemp(prefix="mshgen_m6_")
    try:
        if "MegAWES" in which:
            folder = os.path.join(REPO, "cases", "MegAWES_wing")
            ref = checkmesh_summary(folder)
            run("MegAWES", Case.load(folder), os.path.join(folder, "airfoil.txt"), work, reference=ref)
        if "FFA" in which:
            c = geometry.new_wing_case("ffa", "FFA_W3_211", cases_dir=work)
            run("FFA", c, os.path.join(work, "ffa", "airfoil.txt"), work)
        if "swept" in which:
            p = Planform(half_span=8, root_chord=1.5, tip_chord=0.6, kink_y=1.5, le_sweep_deg=10,
                         twist_root_deg=2, twist_tip_deg=-3, dihedral_deg=6)
            c = geometry.new_wing_case("swept", "NACA0012", cases_dir=work, planform=p)
            run("swept", c, os.path.join(work, "swept", "airfoil.txt"), work)
        print("OK    optimised meshes: same cells, checkMesh not worse")
    finally:
        shutil.rmtree(work, ignore_errors=True)
