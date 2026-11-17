'''
test_m4.py: M4 tests (sizing helpers)

  - geometric series: first fraction <-> grading, growth limit, first/last sizes
  - MegAWES_wing: the wall helper hits its y+ target on the block mesh, the cell budget its cell count
  - with --openfoam, "y+ ~ 50 reproduced automatically": the basic MegAWES example (generate_mesh_MegAWES_WING.py,
    mshGen default settings, y+ far too high) is set up only with the helpers (wall y+ 50, spanwise with a
    tip cell for y+ 50, chordwise and wake with growth 1.15), built, and y+ is measured on the OpenFOAM mesh
    (wall distance of the first cells) for every wall patch

Run (Anaconda module loaded):  python GUI/tests/test_m4.py [--openfoam]
'''
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from core import estimate, sizing  # noqa: E402
from core.case import Case  # noqa: E402
from core.importer import case_from_example  # noqa: E402
from core.model import mshgen_mesh  # noqa: E402
from core.planform import section_params  # noqa: E402
from core.results import measured_yplus  # noqa: E402
from core.runner import checkmesh_summary, run_allrun  # noqa: E402
from core.writer import write_case  # noqa: E402


def test_series():
    for n, g in ((10, 1.0), (64, 5800.0), (20, 0.05), (5, 3.0)):
        f = sizing.first_fraction(n, g)
        assert abs(sizing.grading_for_first_fraction(n, f) / g - 1) < 1e-9, (n, g)
    n, g = sizing.series_first_growth(10.0, 0.01, 1.2)
    assert g ** (1 / (n - 1)) <= 1.2 + 1e-9 and abs(sizing.first_fraction(n, g) * 10 - 0.01) < 1e-9
    n, g = sizing.series_first_last(5.0, 0.1, 0.5)
    first = 5.0 * sizing.first_fraction(n, g)
    assert abs(first / 0.1 - 1) < 0.15 and abs(first * g / 0.5 - 1) < 0.15, (n, g, first)


def test_megawes():
    folder = os.path.join(REPO, "cases", "MegAWES_wing")
    c = Case.load(folder)
    af = os.path.join(folder, "airfoil.txt")
    m = mshgen_mesh(c, af, "edges")
    for yp, mode in ((50.0, None), (30.0, 1.15), (100.0, None)):
        p = sizing.wall_proposal(c, af, m, yp, growth_max=mode)
        got = sizing.predict(c, af, p)["y+ median"]
        assert abs(got / yp - 1) < 0.01, (yp, got)
    for target in (1_000_000, 4_000_000):
        p = sizing.budget_proposal(c, af, m, target)
        got = sizing.predict(c, af, p)
        assert abs(got["cells"] / target - 1) < 0.03, (target, got["cells"])
        assert abs(got["y+ on the tip face (spanwise first cell)"] / 52.82 - 1) < 0.01


def auto_yplus50(openfoam):
    cases = tempfile.mkdtemp(prefix="mshgen_m4_")
    try:
        c = case_from_example(os.path.join(REPO, "examples", "generate_mesh_MegAWES_WING.py"), "auto50", cases_dir=cases)
        folder = os.path.join(cases, "auto50")
        af = os.path.join(folder, "airfoil.txt")
        before = sizing.predict(c, af)
        m = mshgen_mesh(c, af, "edges")
        # wall: y+ 50 with growth <= 1.15
        sizing.apply_proposal(c, sizing.wall_proposal(c, af, m, 50.0, growth_max=1.15))
        # chordwise and wake: TE/LE/max sizes as the defaults, growth 1.15
        from core import preview
        surf = preview.surface_cells(preview.slice_mesh(c, af, 0, m))
        up, lo = surf["upper"]["length"], surf["lower"]["length"]
        d_te, d_le, d_max = up[-1], min(up[0], lo[0]), max(up.max(), lo.max())
        sizing.apply_proposal(c, sizing.chordwise_proposal(c, m, d_te, d_le, d_max, 1.15))
        sizing.apply_proposal(c, sizing.wake_proposal(c, m, d_te, 1.15, 2.0))
        # spanwise: tip cell for y+ 50 with the tip chord, growth 1.15
        tip_h = sizing.first_height_for_yplus(c, 50.0, section_params(c.wing_stations, c.half_span)[0])
        d_root = estimate.spanwise_table(c)[0]["first cell (m)"]
        sizing.apply_proposal(c, sizing.spanwise_proposal(c, d_root, tip_h, 1.0, 1.15))
        after = sizing.predict(c, af)
        print("      before: cells {:,}, y+ median {:.1f}, tip {:.1f}".format(
            before["cells"], before["y+ median"], before.get("y+ on the tip face (spanwise first cell)", float("nan"))))
        print("      after (predicted): cells {:,}, y+ median {:.1f} (max {:.1f}), tip {:.1f}, chordwise jump {:.2f}, "
              "spanwise jump {:.2f}".format(after["cells"], after["y+ median"], after["y+ max"],
                                            after["y+ on the tip face (spanwise first cell)"],
                                            after["largest chordwise jump on the airfoil"],
                                            after["largest spanwise jump at a slice"]))
        assert abs(after["y+ median"] / 50 - 1) < 0.02
        if not openfoam:
            return
        c.environment.fluent = False
        write_case(folder, c)
        lines = []
        assert run_allrun(folder, on_line=lines.append) == 0, "\n".join(lines[-15:])
        s = checkmesh_summary(folder)
        assert s["cells"] == after["cells"]
        meas = measured_yplus(folder, c)
        for patch, v in meas.items():
            print("      measured {:8s} {:>6} faces: y+ median {:.1f} (min {:.1f}, max {:.1f})".format(
                patch, v["faces"], v["y+ median"], v["y+ min"], v["y+ max"]))
            assert 40 <= v["y+ median"] <= 60, (patch, v)
        print("      checkMesh: {:,} cells, max non-orth {:.1f}, max skewness {:.2f}, {}".format(
            s["cells"], s["max_non_orthogonality"], s["max_skewness"],
            "mesh OK" if s["mesh_ok"] else "failed: " + "; ".join(s["warnings"])))
    finally:
        shutil.rmtree(cases, ignore_errors=True)


if __name__ == "__main__":
    test_series()
    print("OK    geometric series")
    test_megawes()
    print("OK    MegAWES_wing: wall helper hits y+ 30/50/100 (median, within 1 %), budget 1M/4M cells, tip y+ kept")
    print("..    automatic y+ 50 setup of the basic MegAWES example" + (" (built)" if "--openfoam" in sys.argv else ""))
    auto_yplus50("--openfoam" in sys.argv)
    print("OK    automatic y+ 50 setup")
