'''
test_m1.py: M1 tests (geometry)

  - planform parameters <-> table, derived values
  - span change scales the slices; geometry change -> block vertices outdated -> recomputed,
    edges kept unless airfoil/mesh_scale/mesh_level changed
  - new wing case for every library airfoil, and a swept/tapered/twisted wing with dihedral:
    built with Allrun.pre (mshGen, blockMesh, checkMesh; no Fluent export) when --openfoam is given

Run (Anaconda module loaded):  python GUI/tests/test_m1.py [--openfoam] [--keep]
'''
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from core import airfoils, blockcheck, estimate, geometry  # noqa: E402
from core.case import Case  # noqa: E402
from core.planform import Planform, describe, fit  # noqa: E402
from core.runner import checkmesh_summary, run_allrun  # noqa: E402
from core.writer import write_case  # noqa: E402


def test_planform():
    p = Planform(half_span=10, root_chord=2, tip_chord=1, kink_y=2, le_sweep_deg=10, twist_root_deg=1,
                 twist_tip_deg=-2, dihedral_deg=5)
    st = p.to_stations()
    assert len(st) == 3 and st[1][1] == 2 and abs(st[2][4] + 8 * math.tan(math.radians(10))) < 1e-12
    q = fit(st)
    for k, v in p.to_dict().items():
        assert abs(getattr(q, k) - v) < 1e-9, (k, getattr(q, k), v)
    d = describe(Planform(half_span=5, root_chord=1, tip_chord=1).to_stations())
    assert abs(d["area, full wing (m2)"] - 10) < 1e-12 and abs(d["aspect ratio"] - 10) < 1e-12


def test_airfoil_data():
    '''Every library airfoil can be read and drawn (step 1) as mshGen reads it.'''
    for name in airfoils.names():
        a = airfoils.AirfoilData(name, airfoils.library_file(name))
        assert len(a.points) > 10 and a.outline().shape[1] == 2 and 0.05 < a.max_thickness < 0.4, name


def test_geometry_changes(cases):
    c = geometry.new_wing_case("geo", "NACA0012", cases_dir=cases)
    folder = os.path.join(cases, "geo")
    af = os.path.join(folder, "airfoil.txt")
    assert geometry.vertex_state(c, af) == "ok"
    # span x2 -> slices x2, vertices stay valid (root section unchanged)
    s0 = list(c.blocks.slice_locations)
    p = Planform(**c.wing.planform)
    p.half_span *= 2
    geometry.set_stations(c, p.to_stations(), scale_slices=True)
    assert all(abs(a - 2 * b) < 1e-12 for a, b in zip(c.blocks.slice_locations, s0))
    assert geometry.vertex_state(c, af) == "ok" and not c.validate()
    # root chord -> vertices outdated; recompute keeps the (tuned) edges
    c.meshing.edges["e00"] = [77, 123.0, 123.0]
    p.root_chord = 2.0
    geometry.set_stations(c, p.to_stations())
    assert geometry.vertex_state(c, af) == "outdated"
    assert geometry.recompute_vertices(c, af) is False and c.meshing.edges["e00"] == [77, 123.0, 123.0]
    assert geometry.vertex_state(c, af) == "ok"
    # airfoil change -> edges reset as well
    c.airfoil.name = "Mrev-v2"
    af2 = airfoils.library_file("Mrev-v2")
    assert geometry.vertex_state(c, af2) == "outdated" and geometry.edge_defaults_changed(c)
    assert geometry.recompute_vertices(c, af2) is True and c.meshing.edges["e00"] != [77, 123.0, 123.0]
    # 2D and back
    mem = {}
    geometry.set_dim(c, "2D", mem)
    assert c.blocks.slice_locations == [0.0, 1.0] and c.meshing.layer_resolution == [1] and not c.validate()
    geometry.set_dim(c, "3D", mem)
    assert c.blocks.slice_locations == [2 * s for s in s0]


def build(folder, openfoam):
    c = Case.load(folder)
    c.environment.fluent = False
    write_case(folder, c)
    if not openfoam:
        return None
    lines = []
    code = run_allrun(folder, on_line=lines.append)
    assert code == 0, "\n".join(lines[-20:])
    s = checkmesh_summary(folder)
    assert s["cells"] == estimate.cell_count(c), (s["cells"], estimate.cell_count(c))
    assert not blockcheck.check(c, os.path.join(folder, c.airfoil.file))
    return s


def test_new_wings(cases, openfoam=False):
    results = []
    for name in airfoils.names():
        if not airfoils.has_defaults(name):          # must be refused (mshGen's defaults cross)
            try:
                geometry.new_wing_case("refused", name, cases_dir=cases)
            except ValueError:
                results.append((name + " (refused)", "refused"))
                continue
            raise AssertionError("new wing with " + name + " should be refused")
        case = name.replace("-", "_") + "_wing"
        geometry.new_wing_case(case, name, cases_dir=cases)
        results.append((case, build(os.path.join(cases, case), openfoam)))
    # Too much sweep: the trailing edge passes the near-wake vertex -> block check must refuse it
    p = Planform(half_span=8, root_chord=1.5, tip_chord=0.6, kink_y=1.5, le_sweep_deg=25, twist_root_deg=2,
                 twist_tip_deg=-3, dihedral_deg=6)
    c = geometry.new_wing_case("swept25_wing", "NACA0012", cases_dir=cases, planform=p)
    problems = blockcheck.check(c, os.path.join(cases, "swept25_wing", "airfoil.txt"))
    assert problems and all("near wake" in q for q in problems), problems
    results.append(("swept25_wing", "invalid: " + problems[0]))
    # Within the limit: swept, tapered, twisted wing with dihedral
    p.le_sweep_deg = 10
    c = geometry.new_wing_case("swept10_wing", "NACA0012", cases_dir=cases, planform=p)
    assert not blockcheck.check(c, os.path.join(cases, "swept10_wing", "airfoil.txt"))
    results.append(("swept10_wing", build(os.path.join(cases, "swept10_wing"), openfoam)))
    return results


if __name__ == "__main__":
    openfoam, keep = "--openfoam" in sys.argv, "--keep" in sys.argv
    cases = tempfile.mkdtemp(prefix="mshgen_m1_")
    try:
        test_planform()
        print("OK    planform parameters <-> table, derived values")
        test_airfoil_data()
        print("OK    all library airfoils read and drawn")
        test_geometry_changes(cases)
        print("OK    span scaling, vertex recompute, edges kept/reset, 2D <-> 3D")
        for case, s in test_new_wings(cases, openfoam):
            if s == "refused":
                print("OK    {:24s} new wing refused: mshGen's default vertices do not work".format(case))
            elif isinstance(s, str):
                print("OK    {:24s} block check refuses it ({})".format(case, s))
            elif s is None:
                print("OK    {:24s} written (no OpenFOAM run)".format(case))
            else:
                print("{}  {:24s} {:>9,} cells, max non-orth {}, max skew {}, max AR {}{}".format(
                    "OK  " if s["mesh_ok"] else "WARN", case, s["cells"], s["max_non_orthogonality"],
                    s["max_skewness"], s["max_aspect_ratio"],
                    "" if s["mesh_ok"] else "  failed: " + "; ".join(s["warnings"])))
    finally:
        if keep:
            print("kept " + cases)
        else:
            shutil.rmtree(cases, ignore_errors=True)
