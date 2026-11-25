'''
test_m10.py: step 6 - a test run with OpenFOAM (simpleFoam) on a built mesh, its results and the AI review

On one built case: by default a new NACA0012 wing (mshGen's default cells) built in a temporary folder; or the mesh of
a case folder given as argument (e.g. a COARSE mesh), linked into a temporary folder and only read.
  - core/simulation.LogReader on a written log: residuals per field (the first solve of p per iteration), the
    start-up line of trapFpe is no error, "FOAM FATAL" is, "End" ends the run, lines written in two parts
  - the machine load: busy processes measured (6 busy loops for a second, only with free cores), the warning
    when the cores of the run and the busy processes exceed the machine, and the Run button of step 6 on a stand-in
    app (no run is started): the first click on a busy machine warns, the second runs, the button is the normal one
    again when the load drops; --load-only stops after this check
  - write_run: 0/ per turbulence model, constant/ (the mesh linked), system/ (function objects: force
    coefficients, y+, wall shear stress, vorticity, Q, surfaces), Allrun and settings.json; the reference values
    (half-wing area, mean aerodynamic chord) and the wake planes behind the most downstream trailing edge
  - Allrun with a stop request (stopAt writeNow) after a few iterations: simpleFoam writes the current iteration and
    ends, reconstructPar, exit code 0
  - the readers: residuals and state, force coefficients every iteration, y+ per wall patch, the surfaces of the
    write (wall, sections, wake planes), Cp along the sections, reversed flow, the tip vortex
  - the facts of the AI review: serialisable, without the case name, the folder or the user name
  - a failing Claude Code CLI gives a readable error (a stand-in binary: nothing is sent); with --live the review
    (Claude through the Claude Code CLI with your Claude login, sending the facts) has the four sections
Run (Anaconda module loaded; OpenFOAM via Allrun):
    python GUI/tests/test_m10.py [<built case folder>] [--cores N] [--live] [--keep] [--load-only]
'''
import getpass
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("MPLBACKEND", "Agg")
from core import aisummary, geometry, meshquality  # noqa: E402
from core import simulation as sm  # noqa: E402
from core.airfoils import AirfoilData  # noqa: E402
from core.case import Case  # noqa: E402
from core.planform import place, section_params  # noqa: E402
from core.results import measured_yplus  # noqa: E402
from core.runner import run_allrun  # noqa: E402
from core.writer import write_case  # noqa: E402

SECTIONS = ["### Convergence", "### Plausibility of the solution",
            "### Effects of the mesh on the solution and the convergence", "### Recommendations"]
ITERATIONS = 200        # requested; the run is stopped after STOP_AT iterations
STOP_AT = 6

RES = "smoothSolver:  Solving for {}, Initial residual = {}, Final residual = 1e-05, No Iterations 2\n"


def check_log_reader(tmp):
    run = os.path.join(tmp, "logtest")
    os.makedirs(run)
    log = os.path.join(run, "log.simpleFoam")
    with open(log, "w") as f:
        f.write("trapFpe: Floating point exception trapping enabled\nTime = 1\n\n" + RES.format("Ux", 1.0)
                + RES.format("p", 0.9) + RES.format("p", 0.5) + "ExecutionTime = 1.5 s  ClockTime = 2 s\n\nTime = 2\n\n"
                + RES.format("Ux", 0.4) + RES.format("p", 0.3)[:30])
    r = sm.LogReader(run)
    assert r.update()
    assert r.error is None, r.error
    assert r.iteration == 2 and r.exec_time == 1.5 and r.clock_time == 2.0
    assert r.res["Ux"] == ([1, 2], [1.0, 0.4]) and r.res["p"] == ([1], [0.9]), r.res      # p: the first solve only
    with open(log, "a") as f:
        f.write(RES.format("p", 0.3)[30:] + "\n--> FOAM FATAL ERROR: something\n\nEnd\n")
    assert r.update()
    assert r.res["p"] == ([1, 2], [0.9, 0.3]), r.res["p"]                                  # the line in two parts
    assert r.error and "FOAM FATAL" in r.error and r.ended
    assert not r.update()                                                                  # nothing new
    print("OK    log reader: residuals, start-up line, error, end, lines in two parts")


def check_load(case, folder):
    '''The machine load before a run: busy processes measured, the warning, and the run button (a stand-in app
    records the calls: no run is started).'''
    ncpu = os.cpu_count() or 1
    base = sm.busy_processes()
    n = 6
    if base + n + 2 <= ncpu:                        # only with free cores: n busy loops for about a second
        spin = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(n)]
        try:
            time.sleep(0.5)
            busy = sm.busy_processes()
        finally:
            for p in spin:
                p.kill()
                p.wait()
        assert n - 1 <= busy - base <= n + 3, (base, busy)
        msg = "busy processes measured: {:.0f} idle, {:.0f} with {} busy loops".format(base, busy, n)
    else:
        msg = "busy processes now: {:.0f} (no free cores for the measurement check)".format(base)
    assert sm.oversubscribed(8, 2, 36) is None and sm.oversubscribed(34, 2, 36) is None
    assert "at most 6 cores" in sm.oversubscribed(16, 30, 36)
    assert "no core is free" in sm.oversubscribed(4, 36, 36)

    from ui import step6_simulation as s6

    class App:                                       # stand-in for the GUI app: save() does not start the run
        def __init__(self):
            self.case, self.folder, self.airfoil_file = case, folder, os.path.join(folder, case.airfoil.file)
            self.step = type("Step", (), {"value": s6.SimulationStep.title})()
            self.notes, self.saves = [], 0

        def notify(self, kind, text):
            self.notes.append((kind, text))

        def save(self, then=None):
            self.saves += 1

        def changed(self, refresh_step=False):
            pass

        def refresh_view(self, step):
            pass

    for busy, warned in ((30, True), (2, False)):
        app = App()
        step = s6.SimulationStep(app)
        step.ncpu, step.cores.end = 36, 36
        step._fill_inputs()
        step.cores.value = 16
        step._load_samples.extend([busy] * 8)
        step.on_run()
        if warned:                                   # first click: warning, no save; second click: save and run
            assert app.saves == 0 and app.notes[-1][0] == "warning" and step.run_btn.name == s6.RUN_ANYWAY
            assert step._load_text[0] == "warning"
            step._load_samples.extend([4] * 8)       # the other jobs end: the button is the normal one again
            step.update_load_note()
            assert step.run_btn.name == s6.RUN_LABEL and step._load_text[0] is None and not step._overload_ok
            step._load_samples.extend([busy] * 8)
            step.on_run()
            assert app.saves == 0 and step.run_btn.name == s6.RUN_ANYWAY
            step.on_run()
        assert app.saves == 1 and step.run_btn.name == s6.RUN_LABEL, (busy, app.saves, step.run_btn.name)
    print("OK    machine load: " + msg + "; 16 cores with 30 busy of 36 warns, the second click runs")


def linked_case(src, tmp):
    '''A temporary case folder with case.json and the airfoil copied and the mesh of src linked (only read).'''
    c = Case.load(src)
    folder = os.path.join(tmp, "case")
    os.makedirs(os.path.join(folder, "constant"))
    for f in ("case.json", c.airfoil.file, "log.checkMesh"):
        if os.path.isfile(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), folder)
    for f in os.listdir(os.path.join(src, "constant")):
        if not f.endswith("Properties"):
            os.symlink(os.path.join(src, "constant", f), os.path.join(folder, "constant", f))
    return Case.load(folder), folder


def check_setup(c, folder):
    run = sm.write_run(c, folder)
    pts = dict(sm.patches(folder))
    walls = [n for n, t in pts.items() if t == "wall"]
    assert walls, pts
    for f in ["U", "p"] + sm.FIELDS[c.simulation.turbulence]:
        text = open(os.path.join(run, "0", f)).read()
        for patch in pts:
            assert patch in text, (f, patch)
    assert os.path.isfile(os.path.join(run, "constant", "polyMesh", "owner"))
    cd = open(os.path.join(run, "system", "controlDict")).read()
    for key in ("forceCoeffs", "yPlus", "wallShearStress", "vorticity", "surfaces", "section0", "endTime         {}"
                .format(ITERATIONS)):
        assert key in cd, key
    s = sm.load_settings(run)
    assert s["walls"] == walls and s["iterations"] == ITERATIONS and s["sections"] == sm.sections(c)
    ref = sm.reference(c)
    if c.dim == "3D":
        st = np.asarray(c.wing_stations, dtype=float)
        assert abs(ref["Aref"] - np.trapz(st[:, 2], st[:, 1])) < 1e-9
        te = max(place(np.array([[1.0, 0.0]]), *section_params(c.wing_stations, y), c.cog)[0][0]
                 for y in np.linspace(0, c.half_span, 41))
        planes = sm.wake_planes(c)
        assert len(planes) == len(sm.WAKE_PLANES) and all(x > te for x, _ in planes), (planes, te)
    rows = sm.settings_table(c)
    assert {"setting", "value"} == set(rows[0]) and len(rows) >= 14
    print("OK    write_run: {} patches ({} walls), {} fields, settings table with {} rows; Aref {:.4g} m2, "
          "lRef {:.4g} m".format(len(pts), len(walls), 2 + len(sm.FIELDS[c.simulation.turbulence]), len(rows),
                                 ref["Aref"], ref["lRef"]))
    return run


def run_and_stop(run):
    '''Allrun; once the log shows STOP_AT iterations, the stop request: the solver writes and ends.'''
    t0 = time.time()
    proc = sm.start(run)
    lines = []
    out = threading.Thread(target=lambda: lines.extend(ln.rstrip() for ln in proc.stdout), daemon=True)
    out.start()                                     # Allrun's output (simpleFoam writes to log.simpleFoam)
    reader = sm.LogReader(run)
    stopped = False
    while proc.poll() is None:
        time.sleep(1.0)
        reader.update()
        if not stopped and reader.iteration >= STOP_AT:
            sm.request_stop(run)
            stopped = True
        if time.time() - t0 > 1800:
            sm.kill(proc)
            raise AssertionError("the run did not end within 30 min")
    out.join(10)
    assert proc.returncode == 0, "\n".join(lines[-20:])
    assert stopped
    reader.update()
    assert reader.ended and reader.error is None, reader.error
    assert STOP_AT <= reader.iteration < ITERATIONS, reader.iteration
    assert "writeNow" in open(os.path.join(run, "system", "controlDict")).read()
    print("OK    Allrun with a stop request at iteration {}: ended at {} ({:.0f} s, {:.2f} s per iteration)".format(
        STOP_AT, reader.iteration, time.time() - t0, reader.clock_time / max(reader.iteration, 1)))
    return reader


def check_results(c, folder, run, reader):
    af = AirfoilData(c.airfoil.name, os.path.join(folder, c.airfoil.file))
    s = sm.load_settings(run)
    n = reader.iteration
    for f in ["p", "Ux", "Uz"] + [f for f in sm.FIELDS[c.simulation.turbulence] if f != "nut"]:
        it, v = reader.res[f]
        assert it == list(range(1, n + 1)) and all(x > 0 for x in v), (f, it)
    fh = sm.force_history(run)
    assert len(fh["Cl"]) == n and np.all(np.isfinite(fh["Cl"])) and np.all(np.isfinite(fh["Cd"])), len(fh["Cl"])
    yp = sm.yplus_history(run)
    assert set(yp) == set(s["walls"]), (set(yp), s["walls"])
    assert all(v[-1, 3] > 0 for v in yp.values())
    times = sm.surface_times(run)
    assert times and float(times[-1]) == n, times
    t = times[-1]
    assert os.path.isdir(os.path.join(run, t)), "reconstructPar wrote no time folder"
    wall = sm.wall_surface(c, af, run, t)
    assert len(wall["xyz"]) > 100 and wall["wallShearStress"].shape == (len(wall["xyz"]), 3)
    assert set(wall["patch"]) == set(s["walls"]), set(wall["patch"])
    # upper / lower side from the geometry (cases without split patches) against the patches (split cases)
    split = {"wingUp", "wingDown"} <= set(s["walls"])
    if split:
        s_geo, up_geo, side_geo = sm.classify(c, af, {"xyz": wall["xyz"]})
        assert np.array_equal(side_geo, wall["side"]) or c.dim != "3D"
        on = ~wall["side"] & np.isin(wall["patch"], ["wingUp", "wingDown"])
        wrong = np.sum(up_geo[on] != (wall["patch"][on] == "wingUp"))
        assert wrong <= 0.002 * on.sum(), (wrong, on.sum())
        assert np.max(np.abs(s_geo[on] - wall["s"][on])) < 1e-12
    a = math.radians(c.simulation.alpha)
    fd = (math.cos(a), 0.0, math.sin(a))
    for i, y in enumerate(s["sections"]):
        cs = sm.cp_section(c, af, wall, y, s["V"], fd)
        assert cs, y
        n_row = len(cs["upper"]["s"]) + len(cs["lower"]["s"])
        for side in ("upper", "lower"):
            o = cs[side]
            # the whole row: both sides around the chord (a row cut short on a twisted part lost one side)
            assert len(o["s"]) >= max(10, 0.3 * n_row), (y, side, len(o["s"]), n_row)
            assert o["s"].min() < 0.02 and o["s"].max() > 0.98, (y, side, o["s"].min(), o["s"].max())
            assert o["s"].min() > -0.02 and o["s"].max() < 1.02, (y, side, o["s"].min(), o["s"].max())
        pl = sm.surface(run, t, "section{}".format(i), ["p", "U"])         # cut points may be snapped to mesh points
        chord = section_params(c.wing_stations or [[0, 0, 1.0, 0, 0, 0]], y)[0]
        assert len(pl.get("xyz", [])) > 100 and np.abs(pl["xyz"][:, 1] - y).max() < 0.01 * chord, y
    share = sm.separated_share(wall, fd)
    assert 0.0 <= share["upper"] <= 1.0 and 0.0 <= share["lower"] <= 1.0, share
    msg = "{} wall faces, {} sections".format(len(wall["xyz"]), len(s["sections"]))
    if split:
        msg += ", upper/lower from the geometry: {} of {} faces differ from the patches".format(wrong, on.sum())
    if c.dim == "3D":
        for i, (x, c_tip) in enumerate(s["wake_planes"]):
            pl = sm.surface(run, t, "wake{}".format(i), ["U", "vorticity"])
            assert len(pl.get("xyz", [])) > 100 and np.abs(pl["xyz"][:, 0] - x).max() < 0.01 * c_tip, x
            core = sm.vortex_core(pl, c_tip, s["V"], c.half_span)
            assert abs(core["y"] - c.half_span) < 2 * c_tip, core
        msg += ", {} wake planes".format(len(s["wake_planes"]))
    print("OK    results at iteration {}: Cl {:.4f}, Cd {:.5f}; y+ average {}; {}; reversed flow {:.1%} / {:.1%} of "
          "the upper / lower side".format(t, fh["Cl"][-1], fh["Cd"][-1], ", ".join(
              "{} {:.3g}".format(k, v[-1, 3]) for k, v in sorted(yp.items())), msg, share["upper"], share["lower"]))
    return af


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    live, keep = "--live" in sys.argv, "--keep" in sys.argv
    cores = int(sys.argv[sys.argv.index("--cores") + 1]) if "--cores" in sys.argv else min(8, os.cpu_count() or 1)
    if "--cores" in sys.argv:
        args.remove(str(cores))
    tmp = tempfile.mkdtemp(prefix="mshgen_m10_")
    try:
        check_log_reader(tmp)
        if args:
            c, folder = linked_case(os.path.abspath(args[0]), tmp)
            print("OK    mesh of {} linked".format(args[0]))
        else:
            c = geometry.new_wing_case("m10_testrun", "NACA0012", cases_dir=tmp)
            folder = os.path.join(tmp, c.name)
            c.environment.fluent = False
            write_case(folder, c)
            lines = []
            t0 = time.time()
            assert run_allrun(folder, on_line=lines.append) == 0, "\n".join(lines[-15:])
            print("OK    NACA0012 wing built ({:.0f} s)".format(time.time() - t0))
        check_load(c, folder)
        if "--load-only" in sys.argv:
            return
        c.simulation.cores, c.simulation.iterations = cores, ITERATIONS
        c.simulation.turbulence, c.simulation.alpha = "kOmegaSST", 4.0
        run = check_setup(c, folder)
        reader = run_and_stop(run)
        af = check_results(c, folder, run, reader)
        groups = meshquality.analyse(folder)
        facts = sm.solution_facts(c, folder, af, run, groups, measured_yplus(folder, c))
        text = json.dumps(facts)
        for private in (c.name, tmp, folder, getpass.getuser(), os.path.expanduser("~")):
            assert private not in text, private
        for key in ("run", "convergence", "mesh"):
            assert key in facts, key
        print("OK    facts for the AI review: {} characters, no case name, folder or user name".format(len(text)))
        try:
            aisummary.write_review(facts, binary=shutil.which("false"))
        except RuntimeError as e:
            print("OK    a failing CLI: " + str(e))
        else:
            raise AssertionError("write_review should fail with a failing CLI")
        if live:
            review, info = aisummary.write_review(facts)
            missing = [h for h in SECTIONS if h not in review]
            assert not missing, missing
            print("OK    AI review ({} via {}; {} input, {} output tokens):\n{}".format(
                info["model"], info["via"], info["input tokens"], info["output tokens"], review))
        print("OK    step 6 test run")
    finally:
        if keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
