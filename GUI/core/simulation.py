'''
simulation.py: step 6, a test run with OpenFOAM (simpleFoam) on the built mesh, and its results

The run lives in <case>/run/: constant/polyMesh links to the case's mesh, 0/, constant/ and system/ are written
here, Allrun runs decomposePar, mpirun simpleFoam -parallel and reconstructPar -latestTime. Inputs (case.simulation):
cores, free-stream velocity V (case.flow.U), angle of attack, turbulence model, iterations, spanwise sections.

Set-up (DEFAULTS, shown in the GUI): steady incompressible RANS (simpleFoam, SIMPLE), freestream velocity and
pressure at the inlet and outlet (the flow direction set by the angle of attack in the x-z plane), no-slip walls with
Spalding's wall function (any y+), symmetry at the root and the far-field side, 0.5 % inflow turbulence with a
viscosity ratio of 10, second-order upwind for U and first order for the turbulence, limited corrections for the
non-orthogonality.

Outputs (function objects, read while the solver runs): residuals (log), force coefficients (every iteration),
y+ per wall patch, and at every write: each wall patch (p, y+, wall shear stress), the spanwise section planes (p, U)
and crossflow planes behind the tip (U, vorticity, Q) as raw surface files. The wall faces are mapped back to x/c and
the upper or lower side of the local section (classify).
'''
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time

import numpy as np

from .planform import place, reference_values, section_params

RUN_DIR = "run"
SETTINGS_FILE = "settings.json"
TURBULENCE = {"kOmegaSST": "k-omega SST", "SpalartAllmaras": "Spalart-Allmaras", "realizableKE": "realizable k-epsilon",
              "laminar": "laminar (no turbulence model)"}
FIELDS = {"kOmegaSST": ["k", "omega", "nut"], "SpalartAllmaras": ["nuTilda", "nut"],
          "realizableKE": ["k", "epsilon", "nut"], "laminar": []}
INTENSITY = 0.005           # inflow turbulence intensity
VISCOSITY_RATIO = 10.0      # inflow nu_t / nu
NUTILDA_RATIO = 3.0         # Spalart-Allmaras inflow nuTilda / nu
WAKE_PLANES = (0.5, 3.0)    # crossflow planes behind the most downstream trailing edge, in tip chords
NOSE = 0.05                 # x/c of the nose, left out of the reversed-flow test (see reversed_flow)
CP_GRID = (0.0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)   # x/c (AI facts)


# ---------------------------------------------------------------------------------------------- #
# Inputs and reference values
# ---------------------------------------------------------------------------------------------- #

def default_sections(case):
    '''Spanwise positions of the section planes: 25, 50, 75 and 95 % of the half-span (2D: the middle).'''
    if case.dim != "3D" or not case.half_span:
        return [0.5]
    b = case.half_span
    return [round(f * b, 3) for f in (0.25, 0.5, 0.75, 0.95)]


def sections(case):
    s = case.simulation.sections
    return list(s) if s else default_sections(case)


def reference(case):
    '''
    Reference values of the force coefficients, exact for the planform rows (planform.reference_values): the
    projected area of the half wing, the mean aerodynamic chord and its quarter chord as CofR (y = 0, the root).
    2D: chord 1 x width, the quarter chord.
    '''
    st = case.wing_stations
    if case.dim == "3D" and st:
        r = reference_values(st, case.cog)
        return {"Aref": r["area"], "lRef": r["mac"], "CofR": (r["quarter_chord"][0], 0.0, r["quarter_chord"][1]),
                "about": "the quarter chord of the mean aerodynamic chord"}
    area = float(case.blocks.slice_locations[-1] - case.blocks.slice_locations[0])
    x, z = place(np.array([[0.25, 0.0]]), *section_params(st or [[0, 0, 1.0, 0, 0, 0]], 0.0), case.cog)
    return {"Aref": area, "lRef": 1.0, "CofR": (float(x[0]), 0.0, float(z[0])), "about": "the quarter chord"}


def inflow(case):
    sim = case.simulation
    V, a = float(case.flow.U), math.radians(sim.alpha)
    nu = float(case.flow.nu)
    k = 1.5 * (INTENSITY * V) ** 2
    return {"U": (V * math.cos(a), 0.0, V * math.sin(a)), "k": k, "omega": k / (nu * VISCOSITY_RATIO),
            "epsilon": 0.09 * k ** 2 / (nu * VISCOSITY_RATIO), "nuTilda": NUTILDA_RATIO * nu,
            "nut": nu * VISCOSITY_RATIO if sim.turbulence != "SpalartAllmaras" else 0.0,
            "liftDir": (-math.sin(a), 0.0, math.cos(a)), "dragDir": (math.cos(a), 0.0, math.sin(a))}


WALL_BCS = {"kOmegaSST": "k: kqRWallFunction; omega: omegaWallFunction",
            "realizableKE": "k: kqRWallFunction; epsilon: epsilonWallFunction", "SpalartAllmaras": "nuTilda: 0"}


def setup_rows(model):
    '''The set-up that does not depend on the inputs: (setting, value), for the table and the AI review.'''
    turb = model != "laminar"
    return [
        ("solver", "simpleFoam: steady, incompressible RANS, SIMPLE, 1 non-orthogonal corrector"),
        ("angle of attack", "the direction of the inflow U = V (cos a, 0, sin a), x downstream, z up; the wing keeps "
                            "its planform twist"),
        ("inlet and outlet (C-boundary, back)", "freestreamVelocity / freestreamPressure (p = 0)"
         + (", inletOutlet for the turbulence" if turb else "")),
        ("walls", "no-slip" + ("; nut: Spalding wall function (any y+); " + WALL_BCS[model] if turb else "")),
        ("root and far-field side", "symmetry"),
        ("schemes", "U: bounded Gauss linearUpwind; turbulence: bounded Gauss upwind; gradients cellLimited Gauss "
                    "linear 1 (all fields); laplacian and snGrad limited corrected 0.33"),
        ("linear solvers, relaxation", "p: GAMG (relTol 0.05); U and turbulence: smoothSolver (relTol 0.1); "
                                       "relaxation p 0.3, U 0.7, turbulence 0.7"),
        ("stops at", "the iterations or residuals p < 1e-5, U and turbulence < 1e-6")]


def settings_table(case):
    '''Rows (setting, value) of the set-up, for the GUI table.'''
    sim, inf, ref = case.simulation, inflow(case), reference(case)
    V, nu = case.flow.U, case.flow.nu
    c0 = case.root_section()[0]
    fixed = dict(setup_rows(sim.turbulence))
    rows = [
        ("solver", fixed["solver"]),
        ("turbulence model", TURBULENCE[sim.turbulence]),
        ("free-stream velocity V", "{:g} m/s, angle of attack {:g} deg (the inflow direction, in the x-z plane; the "
                                   "wing keeps its twist): U = ({:.4g}, 0, {:.4g}) m/s".format(
                                       V, sim.alpha, inf["U"][0], inf["U"][2])),
        ("viscosity, Reynolds number", "nu = {:g} m2/s, Re = {:.3g} (root chord {:g} m)".format(nu, V * c0 / nu, c0)),
        ("inflow turbulence", "intensity {:g} %, nu_t/nu = {:g}: k = {:.3g} m2/s2, omega = {:.3g} 1/s".format(
            100 * INTENSITY, VISCOSITY_RATIO, inf["k"], inf["omega"]) if sim.turbulence != "laminar" else "-")]
    rows += [(k, fixed[k]) for k in ("inlet and outlet (C-boundary, back)", "walls", "root and far-field side",
                                     "schemes", "linear solvers, relaxation")]
    rows += [
        ("stops at", "{} iterations or residuals p < 1e-5, U and turbulence < 1e-6".format(sim.iterations)),
        ("force coefficients", ("walls; Aref = {:.4g} m2 (projected area of the half wing), lRef = {:.4g} m (mean "
                                "aerodynamic chord), CofR = {} (x = {:.4g} m, z = {:.4g} m); from the planform "
                                "rows" if case.dim == "3D" else "walls; Aref = {:.4g} m2 (chord 1 x width), lRef = "
                                "{:.4g} m, CofR = {} (x = {:.4g} m, z = {:.4g} m)").format(
                                    ref["Aref"], ref["lRef"], ref["about"], ref["CofR"][0], ref["CofR"][2])),
        ("written", "every {} iterations (last two kept), binary".format(write_interval(sim.iterations))),
        ("parallel", "{} cores (decomposePar, scotch; mpirun)".format(sim.cores)),
        ("section planes (y, m)", ", ".join("{:g}".format(y) for y in sections(case))),
    ]
    if case.dim == "3D":
        rows.append(("tip vortex planes", "{} tip chords behind the most downstream trailing edge (x = {})".format(
            " and ".join("{:g}".format(v) for v in WAKE_PLANES),
            ", ".join("{:.4g} m".format(x) for x, _ in wake_planes(case)))))
    return [{"setting": a, "value": b} for a, b in rows]


def write_interval(iterations):
    return max(20, int(iterations) // 10)


# ---------------------------------------------------------------------------------------------- #
# Writing the run
# ---------------------------------------------------------------------------------------------- #

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  written by the mshGen GUI (step 6); rewritten for every run
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _vec(v):
    return "({:.8g} {:.8g} {:.8g})".format(*v)


def patches(case_folder):
    '''[(name, type)] of the mesh's patches.'''
    from .results import patches as _patches
    return [(n, t) for n, t, _, _ in _patches(case_folder)]


def _field(name, cls, dims, internal, bcs):
    out = HEADER.format(cls=cls, obj=name) + "dimensions      {};\n\ninternalField   {};\n\nboundaryField\n{{\n".format(
        dims, internal)
    for patch, bc in bcs:
        out += "    {}\n    {{\n".format(patch) + "".join("        {:<16}{};\n".format(k, v) for k, v in bc) + "    }\n"
    return out + "}\n"


def _bcs(field, pts, inf, model):
    '''Boundary conditions of a field for every patch (by name and type).'''
    out = []
    for name, typ in pts:
        if typ in ("symmetry", "symmetryPlane", "empty", "wedge"):
            out.append((name, [("type", typ)]))
            continue
        far = name in ("inlet", "outlet") or typ == "patch"
        if field == "U":
            bc = [("type", "freestreamVelocity"), ("freestreamValue", "$internalField")] if far else [("type", "noSlip")]
        elif field == "p":
            bc = [("type", "freestreamPressure"), ("freestreamValue", "$internalField")] if far else \
                [("type", "zeroGradient")]
        elif field == "nut":
            bc = [("type", "calculated"), ("value", "$internalField")] if far else \
                [("type", "nutUSpaldingWallFunction"), ("value", "uniform 0")]
        elif field == "nuTilda":
            bc = [("type", "freestream"), ("freestreamValue", "$internalField")] if far else \
                [("type", "fixedValue"), ("value", "uniform 0")]
        else:                                               # k, omega, epsilon
            wall = {"k": "kqRWallFunction", "omega": "omegaWallFunction", "epsilon": "epsilonWallFunction"}[field]
            bc = [("type", "inletOutlet"), ("inletValue", "$internalField"), ("value", "$internalField")] if far \
                else [("type", wall), ("value", "$internalField")]
        out.append((name, bc))
    return out


def write_run(case, case_folder):
    '''Write <case>/run/ for the current inputs (fresh: earlier results are removed). Returns the run folder.'''
    sim = case.simulation
    run = os.path.join(case_folder, RUN_DIR)
    if os.path.isdir(run):
        for f in os.listdir(run):
            p = os.path.join(run, f)
            if f.startswith("processor") or f in ("postProcessing", "system", "constant", "0") or \
                    re.fullmatch(r"\d+(\.\d+)?", f) or f.startswith("log.") or f.startswith("ai_review"):
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
    for d in ("0", "constant", "system"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    os.symlink(os.path.join("..", "..", "constant", "polyMesh"), os.path.join(run, "constant", "polyMesh"))
    pts = patches(case_folder)
    walls = [n for n, t in pts if t == "wall"]
    inf = inflow(case)
    model = sim.turbulence
    # 0/
    fields = {"U": ("volVectorField", "[0 1 -1 0 0 0 0]", "uniform " + _vec(inf["U"])),
              "p": ("volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0")}
    for f in FIELDS[model]:
        dims = {"k": "[0 2 -2 0 0 0 0]", "omega": "[0 0 -1 0 0 0 0]", "epsilon": "[0 2 -3 0 0 0 0]",
                "nuTilda": "[0 2 -1 0 0 0 0]", "nut": "[0 2 -1 0 0 0 0]"}[f]
        fields[f] = ("volScalarField", dims, "uniform {:.8g}".format(inf[f]))
    for f, (cls, dims, internal) in fields.items():
        with open(os.path.join(run, "0", f), "w") as fh:
            fh.write(_field(f, cls, dims, internal, _bcs(f, pts, inf, model)))
    # constant/
    with open(os.path.join(run, "constant", "transportProperties"), "w") as fh:
        fh.write(HEADER.format(cls="dictionary", obj="transportProperties")
                 + "transportModel  Newtonian;\n\nnu              {:.8g};\n".format(case.flow.nu))
    with open(os.path.join(run, "constant", "turbulenceProperties"), "w") as fh:
        body = "simulationType  laminar;\n" if model == "laminar" else (
            "simulationType  RAS;\n\nRAS\n{{\n    RASModel        {};\n    turbulence      on;\n    printCoeffs     on;\n"
            "}}\n".format(model))
        fh.write(HEADER.format(cls="dictionary", obj="turbulenceProperties") + body)
    # system/
    with open(os.path.join(run, "system", "controlDict"), "w") as fh:
        fh.write(_control_dict(case, walls, inf))
    with open(os.path.join(run, "system", "fvSchemes"), "w") as fh:
        fh.write(HEADER.format(cls="dictionary", obj="fvSchemes") + FV_SCHEMES)
    with open(os.path.join(run, "system", "fvSolution"), "w") as fh:
        fh.write(HEADER.format(cls="dictionary", obj="fvSolution") + FV_SOLUTION)
    with open(os.path.join(run, "system", "decomposeParDict"), "w") as fh:
        fh.write(HEADER.format(cls="dictionary", obj="decomposeParDict")
                 + "numberOfSubdomains {};\n\nmethod          scotch;\n".format(max(1, int(sim.cores))))
    with open(os.path.join(run, "Allrun"), "w") as fh:
        fh.write(ALLRUN.format(module=case.environment.openfoam_module, cores=max(1, int(sim.cores))))
    os.chmod(os.path.join(run, "Allrun"), 0o755)
    open(os.path.join(run, "run.foam"), "w").close()
    with open(os.path.join(run, SETTINGS_FILE), "w") as fh:
        json.dump({"V": case.flow.U, "nu": case.flow.nu, "alpha": sim.alpha, "turbulence": model,
                   "iterations": sim.iterations, "cores": sim.cores, "sections": sections(case),
                   "wake_planes": wake_planes(case), "walls": walls, "reference": reference(case),
                   "mesh_time": _mesh_time(case_folder)}, fh, indent=1)
    return run


def _mesh_time(case_folder):
    p = os.path.join(case_folder, "constant", "polyMesh", "owner")
    return os.path.getmtime(p) if os.path.isfile(p) else None


def wake_planes(case):
    '''x positions of the crossflow planes behind the wing (3D), with the tip chord: [(x, c_tip)]. The planes are
    behind the most downstream trailing edge, so that they do not cut the wing.'''
    if case.dim != "3D" or not case.half_span:
        return []
    st = case.wing_stations
    te = max(float(place(np.array([[1.0, 0.0]]), *section_params(st, y), case.cog)[0][0])
             for y in np.linspace(0.0, case.half_span, 41))
    c_tip = section_params(st, case.half_span)[0]
    return [(round(te + f * c_tip, 4), c_tip) for f in WAKE_PLANES]


def _control_dict(case, walls, inf):
    sim, ref = case.simulation, reference(case)
    wl = " ".join(walls)
    surf = ""
    for w in walls:                                     # one surface per wall patch: the patch tells the side
        surf += "        wall_{0}\n        {{\n            type            patch;\n            patches         ({0});\n" \
                "            interpolate     false;\n        }}\n".format(w)
    for i, y in enumerate(sections(case)):
        surf += ("        section{}\n        {{\n            type            cuttingPlane;\n"
                 "            planeType       pointAndNormal;\n            pointAndNormalDict\n            {{\n"
                 "                point   (0 {:.8g} 0);\n                normal  (0 1 0);\n            }}\n"
                 "            interpolate     false;\n        }}\n").format(i, y)
    for i, (x, _) in enumerate(wake_planes(case)):
        surf += ("        wake{}\n        {{\n            type            cuttingPlane;\n"
                 "            planeType       pointAndNormal;\n            pointAndNormalDict\n            {{\n"
                 "                point   ({:.8g} 0 0);\n                normal  (1 0 0);\n            }}\n"
                 "            interpolate     false;\n        }}\n").format(i, x)
    turb = sim.turbulence != "laminar"
    return HEADER.format(cls="dictionary", obj="controlDict") + """
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {iters};
deltaT          1;
writeControl    timeStep;
writeInterval   {wi};
purgeWrite      2;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        log             no;
        patches         ({walls});
        rho             rhoInf;
        rhoInf          1;
        liftDir         {lift};
        dragDir         {drag};
        CofR            {cofr};
        pitchAxis       (0 1 0);
        magUInf         {V:.8g};
        lRef            {lref:.8g};
        Aref            {aref:.8g};
    }}
{yplus}
    wallShearStress
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        patches         ({walls});
        executeControl  writeTime;
        writeControl    writeTime;
        log             no;
    }}
    vorticity
    {{
        type            vorticity;
        libs            (fieldFunctionObjects);
        executeControl  writeTime;
        writeControl    writeTime;
        log             no;
    }}
    Q
    {{
        type            Q;
        libs            (fieldFunctionObjects);
        executeControl  writeTime;
        writeControl    writeTime;
        log             no;
    }}
    surfaces
    {{
        type            surfaces;
        libs            (sampling);
        writeControl    writeTime;
        surfaceFormat   raw;
        interpolationScheme cell;
        fields          (p U {extra}wallShearStress vorticity Q);
        surfaces
        {{
{surf}        }}
    }}
}}
""".format(iters=int(sim.iterations), wi=write_interval(sim.iterations), walls=wl, lift=_vec(inf["liftDir"]),
           drag=_vec(inf["dragDir"]), cofr=_vec(ref["CofR"]), V=case.flow.U, lref=ref["lRef"], aref=ref["Aref"],
           yplus=YPLUS if turb else "", extra="yPlus " if turb else "", surf=surf)


YPLUS = """    yPlus
    {
        type            yPlus;
        libs            (fieldFunctionObjects);
        executeControl  writeTime;
        writeControl    writeTime;
        log             no;
    }"""

FV_SCHEMES = """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,nuTilda) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear limited corrected 0.33;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited corrected 0.33;
}

wallDist
{
    method          meshWave;
}
"""

FV_SOLUTION = """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.05;
    }

    "(U|k|omega|epsilon|nuTilda)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      no;

    residualControl
    {
        p               1e-5;
        U               1e-6;
        "(k|omega|epsilon|nuTilda)" 1e-6;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        "(k|omega|epsilon|nuTilda)" 0.7;
    }
}
"""

ALLRUN = """#!/bin/bash
# Allrun: test run of the mshGen GUI (step 6) on the mesh of this case (constant/polyMesh -> ../../constant/polyMesh)
cd "$(dirname "$0")" || exit 1
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge
module load "{module}"
source "$FOAM_BASH"
NP={cores}
if [ "$NP" -gt 1 ]; then
    echo "== decomposePar ($NP cores)"
    decomposePar -force > log.decomposePar 2>&1 || {{ tail -20 log.decomposePar; exit 1; }}
    echo "== simpleFoam (parallel)"
    mpirun -np "$NP" simpleFoam -parallel > log.simpleFoam 2>&1
    code=$?
    echo "== reconstructPar (latest time)"
    reconstructPar -latestTime > log.reconstructPar 2>&1 || true
else
    echo "== simpleFoam"
    simpleFoam > log.simpleFoam 2>&1
    code=$?
fi
tail -5 log.simpleFoam
echo "== simpleFoam exit code $code"
exit $code
"""


# ---------------------------------------------------------------------------------------------- #
# Running
# ---------------------------------------------------------------------------------------------- #

def start(run, on_line=None):
    '''Start Allrun (own process group, so that it can be stopped as a whole): the Popen object.'''
    return subprocess.Popen(["bash", "./Allrun"], cwd=run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, start_new_session=True)


def request_stop(run):
    '''Ask simpleFoam to write the current iteration and end (stopAt writeNow; read at the next iteration).'''
    p = os.path.join(run, "system", "controlDict")
    if os.path.isfile(p):
        t = open(p).read()
        t = re.sub(r"\nstopAt\s+\w+;", "\nstopAt          writeNow;", t)
        with open(p, "w") as fh:
            fh.write(t)


def kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def procs_running():
    '''Threads running or ready to run on this machine right now (all users), from /proc/stat, without the thread
    that reads it; None where /proc/stat is missing (not Linux).'''
    try:
        with open("/proc/stat") as fh:
            for line in fh:
                if line.startswith("procs_running"):
                    return max(0, int(line.split()[1]) - 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def busy_processes(samples=5, spacing=0.05):
    '''
    Busy processes on the machine (all users): the median of a few procs_running samples (elsewhere the 1-minute
    load average). MPI processes count while they wait for each other, because they poll: this is the demand on the
    cores, which the CPU usage (at most 100 %) cannot show.
    '''
    vals = []
    for i in range(samples):
        v = procs_running()
        if v is None:
            return float(os.getloadavg()[0])
        vals.append(v)
        if i + 1 < samples:
            time.sleep(spacing)
    return float(np.median(vals))


def oversubscribed(cores, busy, ncpu):
    '''Warning if a run with this many cores would give more busy processes than the machine has cores, else None.
    In a parallel run every process waits for the slowest at each step, so sharing cores slows it down far more
    than in proportion (a test run next to a Fluent job: 2 minutes per iteration instead of seconds).'''
    busy = int(round(busy))
    if cores + busy <= ncpu:
        return None
    free = ncpu - busy
    hint = ("choose at most {} cores, or wait for the other jobs to finish".format(free) if free >= 1
            else "no core is free now: wait for the other jobs to finish")
    return ("This machine has {} cores and {} busy processes now (all users): with {} cores for this run, {} processes "
            "would share {} cores and the run would be very slow (its processes wait for each other at every step); "
            "{}.".format(ncpu, busy, cores, cores + busy, ncpu, hint))


# ---------------------------------------------------------------------------------------------- #
# Results
# ---------------------------------------------------------------------------------------------- #

def load_settings(run):
    p = os.path.join(run, SETTINGS_FILE)
    return json.load(open(p)) if os.path.isfile(p) else None


_RES = re.compile(r"Solving for (\w+), Initial residual = ([-+0-9.eE]+), Final residual = ([-+0-9.eE]+)")


class LogReader:
    '''Reads log.simpleFoam incrementally: initial residuals per field and iteration, execution time, state.'''

    def __init__(self, run):
        self.path = os.path.join(run, "log.simpleFoam")
        self.pos = 0
        self.iteration = 0
        self.res = {}                # field -> ([iterations], [initial residuals]) (first solve per iteration)
        self.exec_time = 0.0         # CPU time of the master (s)
        self.clock_time = 0.0        # wall-clock time (s)
        self.converged = None        # iteration at which residualControl stopped the run
        self.error = None
        self.ended = False           # the solver wrote "End"
        self._seen = set()
        self._buf = ""

    def update(self):
        if not os.path.isfile(self.path):
            return False
        size = os.path.getsize(self.path)
        if size < self.pos:                                 # a new run
            self.__init__(os.path.dirname(self.path))
        if size == self.pos:
            return False
        with open(self.path, errors="replace") as fh:
            fh.seek(self.pos)
            chunk = fh.read()
            self.pos = fh.tell()
        lines = (self._buf + chunk).split("\n")
        self._buf = lines.pop()
        for line in lines:
            if line.startswith("Time = "):
                try:
                    self.iteration = int(float(line.split("=")[1]))
                except ValueError:
                    pass
                self._seen = set()
                continue
            m = _RES.search(line)
            if m:
                f = m.group(1)
                if f in self._seen:                         # p again (non-orthogonal corrector)
                    continue
                self._seen.add(f)
                it, val = self.res.setdefault(f, ([], []))
                it.append(self.iteration)
                val.append(float(m.group(2)))
                continue
            if line.startswith("ExecutionTime"):
                m2 = re.match(r"ExecutionTime = ([0-9.eE+-]+) s\s+ClockTime = ([0-9.eE+-]+) s", line)
                if m2:
                    self.exec_time, self.clock_time = float(m2.group(1)), float(m2.group(2))
            elif "solution converged in" in line:
                m2 = re.search(r"converged in (\d+)", line)
                self.converged = int(m2.group(1)) if m2 else self.iteration
            elif "FOAM FATAL" in line or line.startswith("#0 ") or "Floating point exception (core dumped)" in line:
                self.error = self.error or line.strip()
            elif line.strip() == "End":
                self.ended = True
        return True


def _table(path):
    '''Columns of a postProcessing .dat file with a "# Time ..." header: {name: array}.'''
    names, rows = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                parts = line[1:].split()
                if parts and parts[0] == "Time":
                    names = parts
                continue
            vals = line.split()
            if vals:
                rows.append(vals)
    if not names or not rows:
        return {}
    out = {}
    for i, n in enumerate(names):
        try:
            out[n] = np.array([float(r[i]) for r in rows if len(r) > i])
        except ValueError:
            out[n] = [r[i] for r in rows if len(r) > i]
    return out


def force_history(run):
    '''{Time, Cd, Cl, CmPitch, ...} from postProcessing/forceCoeffs/*/coefficient.dat (all restarts joined).'''
    base = os.path.join(run, "postProcessing", "forceCoeffs")
    if not os.path.isdir(base):
        return {}
    out = {}
    for t in sorted(os.listdir(base), key=_num):
        p = os.path.join(base, t, "coefficient.dat")
        if os.path.isfile(p):
            for k, v in _table(p).items():
                out[k] = np.r_[out[k], v] if k in out else v
    return out


def yplus_history(run):
    '''{patch: (time, min, max, average)} from postProcessing/yPlus/*/yPlus.dat.'''
    base = os.path.join(run, "postProcessing", "yPlus")
    out = {}
    if not os.path.isdir(base):
        return out
    for t in sorted(os.listdir(base), key=_num):
        p = os.path.join(base, t, "yPlus.dat")
        if not os.path.isfile(p):
            continue
        for line in open(p):
            if line.startswith("#"):
                continue
            v = line.split()
            if len(v) >= 5:
                out.setdefault(v[1], []).append([float(v[0]), float(v[2]), float(v[3]), float(v[4])])
    return {k: np.array(v) for k, v in out.items()}


def _num(s):
    try:
        return float(s)
    except ValueError:
        return float("inf")


def surface_times(run):
    base = os.path.join(run, "postProcessing", "surfaces")
    if not os.path.isdir(base):
        return []
    return sorted((d for d in os.listdir(base) if _num(d) != float("inf")), key=_num)


def _raw(path):
    '''A raw surface file: (points (n x 3), values (n x k)).'''
    a = np.loadtxt(path, comments="#", ndmin=2)
    return a[:, :3], a[:, 3:]


def surface(run, time, name, fields):
    '''{"xyz": points, field: values} of a sampled surface at a write time (fields that are missing are skipped).'''
    base = os.path.join(run, "postProcessing", "surfaces", str(time))
    out = {}
    for f in fields:
        p = os.path.join(base, "{}_{}.raw".format(f, name))
        if not os.path.isfile(p):
            continue
        xyz, v = _raw(p)
        out.setdefault("xyz", xyz)
        out[f] = v[:, 0] if v.shape[1] == 1 else v
    return out


# mshGen's wall patches that tell the side ("extra": the first layers of wingUp). Not wingTip: without SPLIT it holds
# the wing surface of the tip layer and the tip cap (with SPLIT only the cap); classify() finds the cap by its plane.
SIDE_PATCHES = {"wingUp": "upper", "extra": "upper", "wingDown": "lower", "wingCut": "side"}
WALL_FIELDS = ["p", "yPlus", "wallShearStress"]


def mesh_sections(case, y):
    '''
    (chord, beta, dx, dz) at the spanwise positions y as the mesh has them: mshGen places the sections at the slices
    (planform.section_params) and blockMesh interpolates linearly between the slices (straight spanwise edges), so
    planform rows between two slices do not show in the mesh (MegAWES: 24 of its 25 rows; on a curved tip the
    chord differs by 10 % and more). Beyond the tip: the tip section.
    '''
    st = case.wing_stations or [[0, 0, 1.0, 0, 0, 0]]
    s = np.unique(np.asarray(case.blocks.slice_locations, dtype=float))
    p = np.array([section_params(st, v) for v in s])
    yy = np.clip(np.abs(np.asarray(y, dtype=float)), s.min(), s.max())
    return tuple(np.interp(yy, s, p[:, i]) for i in range(4))


def local_coords(case, airfoil, xyz):
    '''
    Section coordinates of wall points (x/c, z/c), from the section at the point's y as the mesh has it
    (mesh_sections). blockMesh interpolates the placed points between the slices, not the twist: up to about 1 % of
    the chord apart, which classify() corrects per row of faces.
    '''
    c, beta, dx, dz = mesh_sections(case, xyz[:, 1])
    cog = case.cog
    X = (xyz[:, 0] + cog[0] + dx) / c
    Z = (xyz[:, 2] + cog[2] - dz) / c
    a = np.radians(-beta)                                   # place(): rotation by -beta about the LE
    return np.cos(a) * X + np.sin(a) * Z, -np.sin(a) * X + np.cos(a) * Z


def _splines(airfoil):
    '''(x, z) of the upper side, the lower side and the camber line, sorted by x (normalised).'''
    out = []
    for arr in (airfoil.upper, airfoil.lower, np.asarray(airfoil.camber)):
        o = np.argsort(arr[:, 0])
        out.append((arr[o, 0], arr[o, 1]))
    return out


def _corrected(xn, zn, y, airfoil):
    '''
    x/c and z/c corrected per row of faces, and the upper side (above the camber line). A row that spans the chord is
    scaled to its LE and TE faces; then a z offset a + b x/c (shift and rotation) is fitted where the airfoil is
    thick (0.05 < x/c < 0.85) to the nearer side, twice. Without it, the TE faces (thinner than the error) are split
    wrongly (MegAWES test run: 148 of 9,943 faces; with it 1, the TE face itself).
    '''
    (ux, uz), (lx, lz), (cx, cz) = _splines(airfoil)
    xs, zs = xn.copy(), zn.copy()
    lab, _ = _rows(y)
    for r in np.unique(lab):
        idx = np.where(lab == r)[0]
        x0, x1 = xn[idx].min(), xn[idx].max()
        if x0 < 0.05 and x1 > 0.95:
            k = 1.0 / (x1 - x0)
            xs[idx], zs[idx] = (xn[idx] - x0) * k, zn[idx] * k
        for _ in range(2):
            up = zs[idx] >= np.interp(xs[idx], cx, cz)
            fit = (xs[idx] > 0.05) & (xs[idx] < 0.85)
            if fit.sum() < 6:
                break
            e = zs[idx] - np.where(up, np.interp(xs[idx], ux, uz), np.interp(xs[idx], lx, lz))
            A = np.column_stack((np.ones(fit.sum()), xs[idx][fit]))
            a, b = np.linalg.lstsq(A, e[fit], rcond=None)[0]
            zs[idx] -= a + b * xs[idx]
    return xs, zs, zs >= np.interp(xs, cx, cz)


def classify(case, airfoil, wall):
    '''
    Wall faces: x/c along the local chord (0 at the LE, 1 at the TE), upper side, and side faces (the tip cap and the
    walls of aileron cut-outs, which are neither side). The side comes from the patch where the wing surface is split
    (wingUp / wingDown), otherwise from the geometry (_corrected). Faces in the tip plane are side faces.
    '''
    xyz = wall["xyz"]
    n = len(xyz)
    xn, zn = local_coords(case, airfoil, xyz)
    kind = np.array([SIDE_PATCHES.get(p, "") for p in wall["patch"]]) if "patch" in wall else np.full(n, "")
    side = kind == "side"
    if case.dim == "3D" and case.half_span and case.blocks.slice_locations[-1] > case.half_span * 1.001:
        ymax = float(xyz[:, 1].max())                       # the tip cap: the plane of the largest y
        side |= xyz[:, 1] >= ymax - 1e-6 * (1.0 + abs(ymax))
    s, upper = xn.copy(), kind == "upper"
    geo = (kind == "") & ~side
    rest = ~geo & ~side
    if geo.any():
        s[geo], _, upper[geo] = _corrected(xn[geo], zn[geo], xyz[geo, 1], airfoil)
    if rest.any():                                          # the patch tells the side; x/c corrected the same way
        s[rest] = _corrected(xn[rest], zn[rest], xyz[rest, 1], airfoil)[0]
    return s, upper, side


def wall_surface(case, airfoil, run, time, fields=WALL_FIELDS):
    '''
    The wall faces at a write time, all wall patches joined: {"xyz", fields, "patch", "s" (x/c), "upper", "side"}
    ({} if there are none). Runs written before the surfaces per patch: the joined "wall" surface, no "patch".
    '''
    parts = []
    for p in (load_settings(run) or {}).get("walls", []):
        d = surface(run, time, "wall_" + p, fields)
        if d:
            d["patch"] = np.full(len(d["xyz"]), p)
            parts.append(d)
    if parts:
        out = {k: np.concatenate([d[k] for d in parts]) for k in parts[0] if all(k in d for d in parts)}
    else:
        out = surface(run, time, "wall", fields)
    if out:
        out["s"], out["upper"], out["side"] = classify(case, airfoil, out)
    return out


def cp_section(case, airfoil, wall, y, V, flow_dir=(1.0, 0.0, 0.0)):
    '''
    Pressure, y+ and friction along the airfoil at the spanwise position y, from the wall faces (wall_surface): the
    row of faces whose centres are closest to y (side faces left out). Returns {"upper"/"lower": {"s": x/c, "Cp",
    "yPlus", "tau_x"}, "y": row y} or None.
    '''
    if not wall or "p" not in wall:
        return None
    xyz = wall["xyz"]
    cand = np.where(~wall["side"])[0]
    if not len(cand):
        return None
    lab, rows = _rows(xyz[cand, 1])
    sel = cand[lab == int(np.argmin(np.abs(rows - y)))]     # the whole row (see _rows)
    if len(sel) < 6:
        return None
    s, upper = wall["s"][sel], wall["upper"][sel]
    cp = wall["p"][sel] / (0.5 * V ** 2)
    out = {"y": float(np.mean(xyz[sel, 1]))}
    for name, mask in (("upper", upper), ("lower", ~upper)):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(s[idx])]
        o = {"s": s[idx], "Cp": cp[idx], "xz": xyz[sel][idx][:, [0, 2]]}
        if "yPlus" in wall:
            o["yPlus"] = wall["yPlus"][sel][idx]
        if "wallShearStress" in wall:
            # wallShearStress is the stress of the fluid on the wall reversed (OpenFOAM's sign): its component against
            # the flow direction is positive where the flow is attached
            o["tau_x"] = -(np.asarray(wall["wallShearStress"])[sel][idx] @ np.asarray(flow_dir))
        out[name] = o
    return out


def reversed_flow(s, tau_x):
    '''Wall faces with reversed flow: the wall shear against the flow direction, behind the nose (x/c > NOSE; around
    the nose the attached flow runs forward from the stagnation point, against the flow direction).'''
    return (np.asarray(tau_x) < 0) & (np.asarray(s) > NOSE)


def _narrow(ys, new, d):
    '''True if every row (sorted ys split where new) is narrow against the gaps to its neighbouring rows.'''
    cut = np.where(new)[0]
    start, end = np.r_[0, cut + 1], np.r_[cut, len(ys) - 1]
    gaps = d[cut]
    return bool(np.all(ys[end] - ys[start] <= 0.1 * np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])))


def _rows(y):
    '''
    Row label of each wall face and the y of the rows (their smallest y). The faces of one spanwise row share y up to
    rounding where the wing is straight; where it is twisted and tapered, the centres of the warped faces spread a
    little (1e-5 of the row height on a 5 deg twist, about 1 % on 30 deg over two rows), while the rows are a cell
    height apart. The sorted y are split at the gaps above the largest jump between the sizes of the largest fifth
    of the gaps; a split that leaves a row wide against the gaps around it is not used (then the next jump, finally
    every gap above 1e-6 (1 + |y|)).
    '''
    y = np.asarray(y, dtype=float)
    order = np.argsort(y)
    ys = y[order]
    d = np.diff(ys)
    tiny = 1e-9 * (1.0 + np.abs(ys[1:]))
    new = d > 1e-6 * (1.0 + np.abs(ys[1:]))
    g = np.sort(d[d > tiny])[::-1]
    if len(g):
        m = min(len(g), max(1, len(ys) // 5))
        top = np.r_[g[:m], g[m] if m < len(g) else tiny.min()]
        ratio = top[:-1] / np.maximum(top[1:], 1e-300)
        for k in np.argsort(ratio)[::-1][:10]:
            trial = d > math.sqrt(top[k] * top[k + 1])
            if _narrow(ys, trial, d):
                new = trial
                break
    new = np.r_[True, new]
    lab = np.empty(len(y), dtype=int)
    lab[order] = np.cumsum(new) - 1
    return lab, ys[new]


def _trapz_weights(v):
    '''Weights of sorted positions v for a trapezoidal integral.'''
    if len(v) < 2:
        return np.ones(len(v))
    d = np.diff(v)
    return 0.5 * (np.r_[d, 0.0] + np.r_[0.0, d])


def separated_share(wall, flow_dir, bands=None):
    '''Share of the upper and the lower side with reversed flow (reversed_flow), weighted by the chord and the span
    (side faces left out), from the wall faces (wall_surface): {"upper": f, "lower": f}, or None. With bands (n),
    also the shares in n equal spanwise bands from the root to the tip: {"upper bands": [...], "lower bands": [...]}.'''
    if not wall or "wallShearStress" not in wall:
        return None
    xyz, s, upper, keep = wall["xyz"], wall["s"], wall["upper"], ~wall["side"]
    rev = reversed_flow(s, -(np.asarray(wall["wallShearStress"]) @ np.asarray(flow_dir)))
    lab, ys = _rows(xyz[:, 1])
    wy = _trapz_weights(ys) if len(ys) > 1 else np.ones(1)
    span = ys[[r for r in range(len(ys)) if (keep & (lab == r)).any()]] if keep.any() else ys
    out = {}
    for side, mask in (("upper", upper), ("lower", ~upper)):
        num, den = np.zeros(len(ys)), np.zeros(len(ys))
        for r in range(len(ys)):
            idx = np.where(keep & mask & (lab == r))[0]
            if len(idx) < 2:
                continue
            idx = idx[np.argsort(s[idx])]
            w = _trapz_weights(s[idx]) * wy[r]
            num[r], den[r] = float(np.sum(w[rev[idx]])), float(np.sum(w))
        out[side] = num.sum() / den.sum() if den.sum() > 0 else 0.0
        if bands and len(span):
            edges = np.linspace(span.min(), span.max(), bands + 1)
            k = np.clip(np.searchsorted(edges, ys, side="right") - 1, 0, bands - 1)
            out[side + " bands"] = [float(num[k == b].sum() / den[k == b].sum()) if den[k == b].sum() > 0 else None
                                    for b in range(bands)]
    return out


def vortex_core(plane, chord, V, tip_y=None):
    '''Location and strength of the tip vortex in a crossflow plane: the largest |streamwise vorticity| within two
    tip chords of the tip (tip_y; anywhere if None).'''
    if not plane or "vorticity" not in plane:
        return None
    w = np.asarray(plane["vorticity"])[:, 0]
    if tip_y is not None:
        w = np.where(np.abs(plane["xyz"][:, 1] - tip_y) < 2 * chord, w, 0.0)
    i = int(np.argmax(np.abs(w)))
    return {"y": float(plane["xyz"][i, 1]), "z": float(plane["xyz"][i, 2]),
            "omega_x c/V": float(w[i] * chord / V)}


def stale(case_folder, run):
    '''True if the mesh was built again after the run was written.'''
    s = load_settings(run)
    return bool(s and s.get("mesh_time") and _mesh_time(case_folder) and _mesh_time(case_folder) > s["mesh_time"] + 1)


# ---------------------------------------------------------------------------------------------- #
# Facts for the AI review of the solution
# ---------------------------------------------------------------------------------------------- #

def _r(v, n=4):
    return float("{:.{}g}".format(float(v), n))


def solution_facts(case, case_folder, airfoil, run, quality_groups=None, yplus_prediction=None):
    '''Technical facts of a test run and its mesh for the AI review (no names or paths).'''
    from . import aisummary, meshopt
    st = load_settings(run) or {}
    V = float(st.get("V", case.flow.U))
    a = math.radians(float(st.get("alpha", case.simulation.alpha)))
    fd = (math.cos(a), 0.0, math.sin(a))
    c0 = case.root_section()[0]
    model = st.get("turbulence", case.simulation.turbulence)
    facts = {"run": {
        "turbulence model": TURBULENCE.get(model, model),
        "free-stream velocity (m/s)": V, "angle of attack (deg)": st.get("alpha"),
        "Reynolds number (root chord)": _r(V * c0 / case.flow.nu), "Mach number": _r(V / 340.0, 3),
        "inflow turbulence": "intensity {:g} %, nu_t/nu {:g}".format(100 * INTENSITY, VISCOSITY_RATIO),
        "set-up": dict(setup_rows(model) if model in TURBULENCE else []),
        "iterations requested": st.get("iterations"), "cores": st.get("cores"),
        "written every (iterations)": write_interval(st.get("iterations", case.simulation.iterations)),
        "mesh built again after the run": stale(case_folder, run)}}
    log = LogReader(run)
    log.update()
    conv = {"iterations done": log.iteration, "stopped by the residual criteria at": log.converged,
            "solver error": log.error, "wall-clock time of the solver (s)": _r(log.clock_time, 3)}
    for f, (it, v) in log.res.items():
        v = np.asarray(v)
        n = max(2, len(v) // 5)
        tail = np.log10(np.maximum(v[-n:], 1e-300))
        conv["initial residual " + f] = {"first": _r(v[0], 3), "last": _r(v[-1], 3),
                                         "orders of magnitude dropped": _r(np.log10(v[0] / max(v[-1], 1e-300)), 3),
                                         "trend over the last 20 % (orders per 100 iterations)":
                                             _r((tail[-1] - tail[0]) / max(it[-1] - it[-n], 1) * 100, 2)}
    facts["convergence"] = conv
    fh = force_history(run)
    if fh.get("Cl") is not None and len(fh["Cl"]):
        n = max(2, len(fh["Cl"]) // 5)
        forces = {}
        for k in ("Cl", "Cd", "CmPitch"):
            v = np.asarray(fh[k])
            forces[k] = {"last": _r(v[-1]), "mean of the last 20 %": _r(v[-n:].mean()),
                         "min / max of the last 20 %": [_r(v[-n:].min()), _r(v[-n:].max())]}
        forces["L/D (last)"] = _r(fh["Cl"][-1] / fh["Cd"][-1]) if fh["Cd"][-1] else None
        ref = st.get("reference") or reference(case)
        forces["reference values of this run"] = {
            "Aref (m2)": _r(ref["Aref"]), "lRef (m)": _r(ref["lRef"]),
            "moment about": "{} (x = {:.4g} m, z = {:.4g} m), positive nose up".format(
                ref.get("about", "the quarter chord at the root"), ref["CofR"][0], ref["CofR"][2]),
            "computed from": ("the planform rows (see mesh, planform): projected area of the half wing and its mean "
                              "aerodynamic chord" if "about" in ref else
                              "the planform rows, by an older version of the GUI: lRef by the trapezoidal rule on "
                              "chord^2, which overestimates the mean aerodynamic chord on tapered parts (exact value: "
                              "mesh, planform)") if case.dim == "3D" else "chord 1 x the width of the 2D slab"}
        facts["force coefficients (half wing; history every iteration)"] = forces
    yp = yplus_history(run)
    if yp:
        sol = {k: {"min": _r(v[-1, 1], 3), "max": _r(v[-1, 2], 3), "average": _r(v[-1, 3], 3)} for k, v in yp.items()}
        t = meshopt.targets(case)
        facts["y+ (first cell centre)"] = {
            "from the solution, last write": sol, "target of the mesh": t["yplus"],
            "predicted while meshing (flat plate, wall distance)": {
                k: _r(v["y+ median"], 3) for k, v in (yplus_prediction or {}).items()}}
    times = surface_times(run)
    if times:
        wall = wall_surface(case, airfoil, run, times[-1])
        share = separated_share(wall, fd, bands=5 if case.dim == "3D" else None)
        if share:
            facts["share of the wing with reversed flow at the wall (separation; chord- and span-weighted, the nose "
                  "x/c < {:g} left out)".format(NOSE)] = {
                (k + " (5 equal spanwise bands, root to tip)" if k.endswith("bands") else k + " side"):
                    [None if v is None else _r(v, 3) for v in val] if isinstance(val, list) else _r(val, 3)
                for k, val in share.items()}
        secs = {}
        for y in st.get("sections", []):
            cs = cp_section(case, airfoil, wall, y, V, fd)
            if not cs:
                continue
            d = {}
            chord = float(mesh_sections(case, cs["y"])[0])                 # local chord of the mesh
            for side in ("upper", "lower"):
                o = cs[side]
                if len(o["s"]) < 3:
                    continue
                d[side] = {"Cp at x/c " + ", ".join("{:g}".format(v) for v in CP_GRID):
                           [_r(v, 3) for v in np.interp(CP_GRID, o["s"], o["Cp"])]}
                if "tau_x" in o:
                    rev = o["s"][reversed_flow(o["s"], o["tau_x"])]
                    d[side]["reversed flow at x/c (range)"] = [_r(rev.min(), 3), _r(rev.max(), 3)] if len(rev) \
                        else None
                ds = np.hypot(*np.diff(o["xz"], axis=0).T) / chord          # spacing of the face centres
                d[side]["chordwise cell length / local chord: at the LE, the TE, largest"] = [
                    _r(ds[0], 3), _r(ds[-1], 3), _r(ds.max(), 3)]
                d[side]["faces"] = len(o["s"])
            sides = [(cs[k]["Cp"], cs[k]["s"], k) for k in ("upper", "lower") if len(cs[k]["s"])]
            if sides:
                cp_max = max(sides, key=lambda v: v[0].max())
                i = int(np.argmax(cp_max[0]))
                d["stagnation point (largest Cp)"] = {"Cp": _r(cp_max[0][i], 3), "x/c": _r(cp_max[1][i], 3),
                                                      "side": cp_max[2]}
            up = cs["upper"]
            if len(up["s"]):
                i = int(np.argmin(up["Cp"]))
                d["suction peak"] = {"Cp": _r(up["Cp"][i], 3), "x/c": _r(up["s"][i], 3)}
            d["local chord (m)"] = _r(chord)
            secs["y = {:.4g} m".format(cs["y"])] = d
        facts["sections (pressure coefficient Cp = p / (0.5 V^2), x/c along the local chord)"] = secs
        if case.dim == "3D" and case.half_span:
            b2 = case.half_span
            xt, zt = place(np.array([[1.0, 0.0]]), *section_params(case.wing_stations, b2), case.cog)
            vort = {"tip trailing edge (x, z) (m)": [_r(xt[0]), _r(zt[0])], "tip y (m)": _r(b2)}
            for i, (x, c_tip) in enumerate(st.get("wake_planes", [])):
                pl = surface(run, times[-1], "wake{}".format(i), ["vorticity"])
                core = vortex_core(pl, c_tip, V, b2)
                if core:
                    vort["crossflow plane {}".format(i + 1)] = {
                        "x (m)": _r(x),
                        "behind the most downstream trailing edge (tip chords)": WAKE_PLANES[i],
                        "behind the tip trailing edge (tip chords)": _r((x - xt[0]) / c_tip, 3),
                        "core y - tip y (tip chords)": _r((core["y"] - b2) / c_tip, 3),
                        "core z - tip trailing edge z (m)": _r(core["z"] - zt[0], 3),
                        "streamwise vorticity omega_x c_tip / V": _r(core["omega_x c/V"], 3)}
            facts["tip vortex (largest streamwise vorticity within 2 tip chords of the tip)"] = vort
    facts["mesh"] = aisummary.mesh_facts(case, case_folder, os.path.join(case_folder, case.airfoil.file),
                                         quality_groups, yplus_prediction)
    return facts
