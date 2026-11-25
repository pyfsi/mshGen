'''
aisummary.py: AI summary of a built mesh (step 5), written by Claude

mesh_facts() collects technical facts of the case and its last build: geometry (airfoil, planform), domain, flow
(U, nu, Reynolds and Mach number), targets of step 4, cell counts, checkMesh results, the groups of flagged and
lower-quality cells (core/meshquality.py), y+ measured on the walls and known limitations of mshGen. It contains no
case name, description, user name or file paths. The GUI shows the facts that are sent.

write_summary() sends them to Claude (MODEL) through the Claude Code CLI of the VS Code extension (or `claude` on
PATH, or MSHGEN_CLAUDE_BIN), headless (claude -p) with your Claude login: no API key and no extra bill (it counts
towards your Claude plan). The CLI runs without tools, keeps no session and runs in an empty temporary folder.
'''
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

MODEL = "claude-opus-5-5"
ENDING = "Finally, don't forget to bring cake to the office!"
OLD_ENDINGS = ("Enjoy your simulations! If it is successful, bring cake to the office.",)    # summaries saved before
CLI_TIMEOUT = 900           # s
SUMMARY_FILE = "ai_summary.md"

SYSTEM = """You review computational meshes for a CFD engineer and write a short, practical summary of one mesh.

The mesh was made with mshGen: a structured C-grid around a wing section, built by OpenFOAM's blockMesh from 27
blocks per spanwise layer (13 inside the airfoil contour, used only beyond the wing tip, and 14 around the airfoil).
3D cases are half wings with a symmetry plane at the root; slices beyond the tip extend the domain spanwise. The
facts you get are measured on the mesh (checkMesh) or are the settings of the case; y+ is a flat-plate estimate
with the local chord, not a flow result. Use only these facts; if something you would need is not in them, say
that it is not known instead of guessing. Give numbers where they matter.

Write Markdown with exactly these sections, in this order, each as a level-3 heading:
### What is good
### Suggested modelling approach
(turbulence modelling first: model, wall treatment that fits the y+, transition, compressibility, steady or
unsteady; then solver settings that matter for this mesh, such as non-orthogonal correctors)
### What can be improved
(concrete changes to the mesh or the case, most important first)
### Things to consider and assumptions
### A crazy idea
(one bold but physically sensible idea: an interesting simulation, application or follow-up study for this wing)

Then one short paragraph that motivates the engineer to run the simulations. Use bullet points inside the sections,
keep the whole summary under about 600 words, and do not add other headings or a title. End the summary with this
sentence, exactly as written and on its own line: """ + ENDING

REVIEW_FILE = "ai_review.md"
REVIEW_SYSTEM = """You review the result of a CFD test run for a CFD engineer and judge how the mesh affected it.

The run is a steady incompressible RANS simulation (OpenFOAM simpleFoam, SIMPLE) on a structured C-grid made with
mshGen: 27 blocks per spanwise layer, 13 inside the airfoil contour (used only beyond the wing tip) and 14 around the
airfoil; 3D cases are half wings with a symmetry plane at the root. The facts you get: the settings of the run, its
convergence (residuals, force coefficients over the iterations), y+ from the solution next to the prediction made
while meshing, pressure and friction along the airfoil at a few spanwise sections, the share of the wall with
reversed flow, the tip vortex in crossflow planes, and the facts of the mesh (checkMesh, flagged cells). Use only
these facts; if something you would need is not in them, say that it is not known instead of guessing. Give numbers.
It is a short test run on a coarse mesh: judge it as such.

Write Markdown with exactly these sections, in this order, each as a level-3 heading:
### Convergence
(did it converge: residual drop, force coefficients settled or still drifting, what limits it)
### Plausibility of the solution
(lift, drag, pressure distribution, separation, tip vortex: what is expected for this airfoil, Reynolds number and
angle of attack, and what looks wrong)
### Effects of the mesh on the solution and the convergence
(y+ against the prediction and the wall treatment, resolution at the leading and trailing edge, in the wake and at
the tip, the flagged and lower-quality cells, the domain size)
### Recommendations
(concrete changes to the mesh, the set-up or the run, most important first)

Use bullet points inside the sections, keep the whole review under about 500 words, and do not add other headings
or a title."""


def _planform(case):
    '''The planform: its rows (exact: mshGen interpolates linearly between them), the parameters it was made from
    (if any) and its reference values, so that the area and the mean aerodynamic chord need not be estimated.'''
    from .planform import reference_values
    st = np.asarray(case.wing_stations, dtype=float)
    y, c, beta = st[:, 1], st[:, 2], st[:, 3]
    b2 = float(y.max())
    ref = reference_values(st)

    def r4(v):
        return float("{:.4g}".format(float(v)))
    out = {"half-span (m)": b2, "root chord (m)": float(c[0]), "tip chord (m)": float(c[-1]),
           "taper ratio (tip / root chord)": r4(c[-1] / c[0]),
           "rows (chord, twist and offsets linear in y between the rows; LE x from the root LE, downstream; z up)": [
               {"y (m)": r4(r[1]), "chord (m)": r4(r[2]), "twist (deg, nose up)": r4(r[3]), "LE x (m)": r4(-r[4]),
                "z (m)": r4(r[5])} for r in st],
           "projected area of the half wing (m2)": r4(ref["area"]),
           "aspect ratio (full wing)": r4((2 * b2) ** 2 / (2 * ref["area"])),
           "mean aerodynamic chord (m)": r4(ref["mac"]),
           "mean aerodynamic chord: y and LE x (m)": [r4(ref["y_mac"]), r4(ref["x_le_mac"])],
           "twist root / tip (deg)": [float(beta[0]), float(beta[-1])],
           "x of leading / trailing edge at the root and the tip (m, mesh coordinates)": {
               "root": [round(v, 4) for v in _edges_x(case, 0.0)], "tip": [round(v, 4) for v in _edges_x(case, b2)]}}
    if case.wing.planform:
        p = case.wing.planform
        k = float(p.get("kink_y", 0.0))
        out["made from"] = ("{}LE sweep {:g} deg and dihedral {:g} deg{}".format(
            "constant chord and twist up to y = {:g} m (kink), then linear to the tip; ".format(k) if k > 0 else
            "chord and twist linear from the root to the tip; ", p.get("le_sweep_deg", 0.0),
            p.get("dihedral_deg", 0.0), " outboard of the kink" if k > 0 else ""))
    return out


def _edges_x(case, y):
    from .planform import place, section_params
    sec = section_params(case.wing_stations, y)
    return [float(place(np.array([[f, 0.0]]), *sec, case.cog)[0][0]) for f in (0.0, 1.0)]


def mesh_facts(case, folder, airfoil_file, quality_groups=None, yplus=None):
    '''Technical facts of the case and its last build (no names or paths), for the summary.'''
    from . import estimate, meshopt
    from .airfoils import AirfoilData
    from .estimate import layer_types
    from .meshquality import summary_rows
    from .runner import checkmesh_summary
    c0 = case.root_section()[0]
    facts = {"case type": "3D half wing" if case.dim == "3D" else "2D airfoil section"}
    try:
        a = AirfoilData(case.airfoil.name, airfoil_file)
        facts["airfoil"] = {"name": case.airfoil.name, "max thickness / chord": round(a.max_thickness, 4),
                            "max camber / chord": round(a.max_camber, 4)}
    except Exception:  # noqa: BLE001
        facts["airfoil"] = {"name": case.airfoil.name}
    if case.dim == "3D" and case.wing_stations:
        facts["planform"] = _planform(case)
    types = layer_types(case)
    s = case.blocks.slice_locations
    facts["domain"] = {
        "C-grid radius / root chord": case.domain.mesh_scale, "outlet": "2 radii behind the C-grid centre",
        "spanwise slices (y, m)": [round(v, 4) for v in s],
        "layers": types, "aileron cut-out layers": case.options.aileron_layers or [],
        "wing surface split in upper and lower patch": bool(case.options.split)}
    U, nu = case.flow.U, case.flow.nu
    facts["flow"] = {"free-stream velocity U (m/s)": U, "kinematic viscosity nu (m2/s)": nu,
                     "Reynolds number (root chord)": U * c0 / nu, "Mach number (with 340 m/s)": U / 340.0}
    t = meshopt.targets(case)
    facts["mesh level and targets (step 4)"] = dict(t, level=case.domain.mesh_level)
    e = case.meshing.edges
    facts["cells"] = {
        "total": estimate.cell_count(case),
        "chordwise per airfoil side": sum(int(e[k][0]) for k in meshopt.CHORD),
        "wall-normal (airfoil to outer boundary)": int(e["e00"][0]),
        "wake (near + far)": int(e["e02"][0]) + int(e["e01"][0]),
        "spanwise per layer": [int(n) for n in case.meshing.layer_resolution] if case.dim == "3D" else None}
    cm = checkmesh_summary(folder)
    if cm:
        facts["checkMesh"] = {k: cm.get(k) for k in ("cells", "hexahedra", "polyhedra", "max_aspect_ratio",
                                                       "max_non_orthogonality", "avg_non_orthogonality",
                                                       "max_skewness", "failed_checks", "mesh_ok", "warnings")}
    if quality_groups:
        facts["flagged and lower-quality cells (x downstream, y spanwise, z up; the wing root at y = 0)"] = \
            summary_rows(quality_groups, xyz=True)
    if yplus:
        facts["y+ at the first cell centre, per wall patch (flat-plate estimate)"] = {
            k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in yplus.items()}
    notes = ["Beyond the tip, the blocks inside the airfoil contour are joined to their neighbours by a "
             "non-conformal interface (mergePatchPairs), which creates polyhedral cells.",
             "The largest non-orthogonality is often at the trailing-edge wedge beyond the tip; it follows from the "
             "trailing-edge angle and cannot be removed by grading.",
             "High aspect ratio cells (checkMesh: above 1000) come from thin wall-normal cells for the target y+ "
             "next to long spanwise cells, and from the thin first spanwise cells at the tip face."]
    if "flow" not in types:
        notes = notes[2:]
    facts["known properties of mshGen meshes"] = notes
    return facts


def find_claude_cli():
    '''Path of the Claude Code CLI: MSHGEN_CLAUDE_BIN, `claude` on PATH, or the newest VS Code extension's binary.'''
    env = os.environ.get("MSHGEN_CLAUDE_BIN")
    if env and os.access(env, os.X_OK):
        return env
    found = shutil.which("claude")
    if found:
        return found
    home = os.path.expanduser("~")
    cands = []
    for root in (".vscode", ".vscode-server", ".vscode-oss", ".cursor", ".cursor-server"):
        cands += glob.glob(os.path.join(home, root, "extensions", "anthropic.claude-code-*", "resources", "native-binary",
                                        "claude"))
    cands += [os.path.join(home, ".local", "bin", "claude"), os.path.join(home, ".claude", "local", "claude")]

    def version(path):
        m = re.search(r"claude-code-(\d+)\.(\d+)\.(\d+)", path)
        return tuple(int(v) for v in m.groups()) if m else (0, 0, 0)
    cands = [c for c in cands if os.path.isfile(c) and os.access(c, os.X_OK)]
    return max(cands, key=version) if cands else None


def cli_status():
    '''(found, text) about the Claude Code CLI, for the GUI.'''
    cli = find_claude_cli()
    if cli:
        return True, "Claude Code CLI: `{}`".format(cli.replace(os.path.expanduser("~"), "~"))
    return False, ("No Claude Code CLI found: install the Claude Code extension in VS Code (or the CLI), or set "
                   "MSHGEN_CLAUDE_BIN to the `claude` binary.")


def _cli_env():
    '''Environment of the CLI: without the markers of a running Claude Code session (when the GUI is started from one)
    and without API keys, so that your Claude login is used.'''
    drop = ("CLAUDECODE", "CLAUDE_CODE_", "CLAUDE_PID", "CLAUDE_AGENT_SDK", "CLAUDE_EFFORT", "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN")
    return {k: v for k, v in os.environ.items() if not k.startswith(drop)}


def _run_cli(system, prompt, binary=None):
    '''
    One headless Claude Code call (your Claude login): (text, info). No tools, no session kept, in an empty temporary
    folder. Raises RuntimeError with a readable message (no CLI, not logged in, usage limits, ...).
    '''
    binary = binary or find_claude_cli()
    if not binary:
        raise RuntimeError("No Claude Code CLI found. Install the Claude Code extension in VS Code (or the CLI), or set "
                           "MSHGEN_CLAUDE_BIN to the `claude` binary.")
    cmd = [binary, "-p", "--model", MODEL, "--effort", "high", "--output-format", "json", "--tools", "",
           "--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config", "--system-prompt", system]
    work = tempfile.mkdtemp(prefix="mshgen_ai_")             # empty folder: no project files or instructions
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=work, env=_cli_env(),
                              timeout=CLI_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Claude Code did not answer within {} s.".format(CLI_TIMEOUT)) from e
    finally:
        shutil.rmtree(work, ignore_errors=True)
    try:
        out = json.loads(proc.stdout)
    except ValueError as e:
        msg = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError("Claude Code failed: " + (msg[-1] if msg else "exit code {}".format(proc.returncode))) from e
    if out.get("is_error"):
        reason = out.get("result") or out.get("api_error") or "error"
        if "login" in reason.lower() or out.get("api_error_status") == 401:
            reason += " (log in once in the Claude Code extension in VS Code, or run `claude` and /login)"
        raise RuntimeError("Claude Code: " + reason)
    text = (out.get("result") or "").strip()
    if not text:
        raise RuntimeError("Claude Code returned no text.")
    u = out.get("usage") or {}
    models = list((out.get("modelUsage") or {}).keys())
    info = {"model": models[0] if models else MODEL, "via": "Claude Code",
            "input tokens": int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0))
            + int(u.get("cache_creation_input_tokens", 0)),
            "output tokens": int(u.get("output_tokens", 0)),
            "cost (USD, counted towards your Claude plan)": out.get("total_cost_usd")}
    return text, info


def write_summary(facts, binary=None):
    '''The summary of the mesh by Claude (Claude Code CLI, your login): (text, info).'''
    text, info = _run_cli(SYSTEM, "Facts of the mesh (JSON):\n\n" + json.dumps(facts, indent=1), binary)
    if not text.endswith(ENDING):
        text = text.rstrip() + "\n\n" + ENDING
    return text, info


def write_review(facts, binary=None):
    '''The review of a test run (step 6) by Claude (Claude Code CLI, your login): (text, info).'''
    return _run_cli(REVIEW_SYSTEM, "Facts of the test run and its mesh (JSON):\n\n" + json.dumps(facts, indent=1),
                    binary)


def save(folder, text, info, name=SUMMARY_FILE):
    with open(os.path.join(folder, name), "w") as f:
        f.write("<!-- {} -->\n".format(json.dumps(info)) + text + "\n")


def load(folder, name=SUMMARY_FILE):
    '''(text, info, modified time) of a saved summary or review, or None.'''
    p = os.path.join(folder, name)
    if not os.path.isfile(p):
        return None
    raw = open(p).read()
    info = {}
    if raw.startswith("<!-- "):
        head, _, raw = raw.partition(" -->\n")
        try:
            info = json.loads(head[5:])
        except ValueError:
            info = {}
    text = raw.strip()
    for old in OLD_ENDINGS:                  # shown with the current ending (the file is not changed)
        if text.endswith(old):
            text = text[:-len(old)].rstrip() + "\n\n" + ENDING
    return text, info, os.path.getmtime(p)
