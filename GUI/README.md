# mshGen GUI

A browser GUI (Panel) to set up, build and check mshGen wing meshes. Every mesh is a **case folder**
in `mshGen/cases/<case>/`, which can also be rebuilt without the GUI. The plan and status are in
[PLAN.md](PLAN.md); known limitations of mshGen are in [../LIMITATIONS.md](../LIMITATIONS.md).

Status: **M6** (all milestones of the plan). All steps are editable:
- geometry: airfoil, 2D/3D, COG, planform;
- domain: R/c, far field;
- spanwise slicing (3a): slices, aileron cut-out layers, patch `extra`;
- airfoil blocking (3b): airfoil and exterior vertices (tables or dragged), wingUp/wingDown;
- meshing: mesh level and targets (y+, cell ranges, growth, cell budget), flow conditions, a mesh
  optimiser that computes all cells and gradings from the targets, editable tables, and a 2D mesh
  preview identical to blockMesh's.

The build step runs the case with a live log and shows the checkMesh summary, the flagged cells, the
Fluent zones, an AI summary of the mesh, a 3D view of the walls and the images. The test run (step
6) runs OpenFOAM's simpleFoam on the mesh and shows the convergence, y+, the pressure distribution,
contours and the tip vortex while it runs, with an AI review of the solution.

## Tutorial video

[docs/mshGen_GUI_tutorial.mp4](docs/mshGen_GUI_tutorial.mp4) shows every input and option in a
complete session: a new wing case through all six steps, from a checked OpenFOAM and Fluent mesh to
a short test run with OpenFOAM and its AI review. Each frame outlines the explained element and shows a pop-up message next to it; there
is no audio.

The chapters (embedded in the MP4, and listed in
[docs/mshGen_GUI_tutorial_chapters.txt](docs/mshGen_GUI_tutorial_chapters.txt)) are: introduction,
window and sidebar, new case, steps 1–6, files.

The video is recorded from the real GUI by `tools/tutorial/record.py`:
- it drives a headless Firefox through the session;
- the 3D views are drawn with Plotly's static export, because headless Firefox has no WebGL.

After changes to the GUI, regenerate it (about 20 minutes; needs OpenFOAM and ParaView for the
build in step 5, OpenFOAM for the test run in step 6 (60 iterations on at most 16 free cores) and the
Claude Code CLI for the AI review):

```
python GUI/tools/tutorial/record.py        # writes GUI/docs/mshGen_GUI_tutorial.mp4
```

## Start the GUI

```
mshGen/GUI/run_gui.sh            # optional: port, default 5006
```

1. The script prints the URL (`http://localhost:5006/app`) and the login: your user name and the
   password stored in `~/.config/mshgen_gui/users.json`. The file is created on first start and is
   readable only by you.
2. In VS Code (Remote), the port is forwarded automatically (PORTS tab). Open the URL in your
   local browser. `/app?case=MegAWES_wing&step=3b` opens a given case and step directly
   (steps 1, 2, 3a, 3b, 4, 5, 6).
3. The server listens on this machine only (127.0.0.1). The password keeps other users logged in on
   the same machine out.

Stop it with Ctrl+C.

### AI summary

The AI summary of step 5 and the AI review of step 6 are written by Claude (`claude-opus-5-5`) through
**Claude Code**: the GUI runs the CLI of the Claude Code extension in VS Code headless (`claude -p`) with your
Claude login. No API key and no package are needed, and there is no extra bill: it counts towards your Claude
plan, like a question in Claude Code (about 3,000 input and 3,000 output tokens per summary, about twice the
input for a review). The CLI runs without tools, keeps no session, and starts in an empty temporary folder.

The GUI finds the CLI in the VS Code extension (`~/.vscode*/extensions/anthropic.claude-code-*`), as `claude`
on PATH, or at `MSHGEN_CLAUDE_BIN`. If the CLI is not logged in, log in once in the Claude Code extension.

### 3D views and WebGL

The 3D views (step 1 wing, step 5 walls) are Plotly figures that need WebGL in the browser. A browser
on a remote desktop or in an X session often has no WebGL; Plotly then only says "WebGL is not
supported by your browser". The GUI checks the browser when the page opens:
- **with WebGL**: interactive views (drag to rotate, scroll to zoom);
- **without WebGL**: the same surfaces are drawn on the server with matplotlib (3D axes, equal
  scales, orthographic, about 1–2 s per view), with view buttons (3D, top, front, side) and
  azimuth / elevation / zoom sliders. The picture is zoomed to fill the frame.

The URL parameter `view3d` (auto / interactive / image) overrides the choice, e.g.
`/app?case=MegAWES_wing&step=5&view3d=image`. Interactive views are
nicer: in Firefox, WebGL can often be enabled in `about:config` with `webgl.force-enabled = true`.

## In the GUI

**Header**: a short fact or quote of the day (one per day of the year, `core/quotes.py`), and
**Undo** / **Redo** next to the theme toggle: every edit of the case (up to 100 steps back, until
the case is reloaded), including Make mesh and the mesh optimiser.

**Sidebar**
- **Case**: select a case from `cases/`.
- **Save** writes `case.json` and all case files. If the geometry changed, it first recomputes the
  block vertices. **Revert** reloads the saved case.
- **Status**: unsaved changes, estimated cells, folder, and problems if there are any.
- **Steps**, each with its status: ✔ done/valid, ⚠ needs attention (e.g. block vertices outdated,
  more cells than the maximum, settings changed after the last build, a test run of an older mesh),
  ✖ error (geometry problems, invalid blocks), ○ not done yet (no mesh, no test run), ▶ running.
- **New case**: either a **New wing** (library airfoil, default planform, mshGen defaults) or
  **From example** (any script in `examples/`).

**1. Geometry** (editable)
- Case type 2D/3D, airfoil (mshGen library), mesh origin (COG).
- Planform: either **Parameters** (half-span, root/tip chord, constant-chord part, LE sweep, twist at
  root and tip, dihedral), which generate the table, or the **Table** itself (station, y, chord,
  twist, dx, dz; insert or delete rows). Changing the span also scales the slices (option).
- Plots (zoom with the mouse wheel; click a legend entry to hide it); the 3D view takes the full
  page width below:
  - the airfoil as mshGen reads it: points, removed point, splines, camber line;
  - the root section in mesh coordinates;
  - top view with the near-wake line;
  - front view;
  - 3D view (drag to rotate; an image with view controls if the browser has no WebGL).
- **Derived values** (below the plots): span, area, aspect ratio, taper, mean aerodynamic chord and
  its spanwise position (exact for the planform rows), sweep and dihedral outboard, twist.
- **Block vertices** follow the geometry: they are recomputed with mshGen's defaults when the
  root section, COG or airfoil changes, either on Save or with the button. The edge resolution is
  kept unless the airfoil changed.
- **Block check**: runs live and shows every block that would be inverted or folded in a slice,
  so blockMesh can't fail on it. With mshGen's fixed exterior vertices, every trailing edge must
  stay upstream of the near-wake line (see `../LIMITATIONS.md` item 7).

**2. Domain** (editable)
- C-grid radius R/c (the block vertices follow it; optimise the cells again in step 4).
- Far field (`symTip`) position.
- **Make mesh (slices, block vertices, cells)**: steps 3 and 4 in one go, in the background (1–3
  minutes): suggest slices (3a), recompute outdated block vertices, suggest block vertices (3b),
  optimise the cells and gradings for the targets (4). One Undo reverts it; check the result in
  steps 3 and 4, then build in step 5.
- Plots: the root plane with the C-domain and the root section, and the domain seen from above
  with the planform and patches. (The exterior block vertices and the slices are in step 3.)

**3a. Spanwise slicing** (editable): where the block vertices are placed along the span. Cells and
gradings are not set here (step 4).
- **Slices**: edit y in the table to move a slice (root and tip are fixed).
  - **Add slice** places the new slice at a cell face of the current spanwise mesh.
  - **Delete** merges the two layers.
  - **Suggest slices** places them at the root, planform kinks, around the tip, and the far field.
  - **Scale the slices when the span changes** (in step 1), so the tip slice stays at the tip.
- **Top view** of the half wing with the numbered slices, the aileron layers and the near-wake line.
- Per layer: aileron cut-out layers, ARTIFICIOUS (first wing layers as patch `extra`).

**3b. Airfoil blocking** (editable): the blocks in each slice.
- **Airfoil vertices**: one table with vertex, boundary (suction, pressure, camber) and position
  along that spline (0 = LE, 1 = TE, about x/c), the same in every slice. TE (0) and LE (13) are
  fixed; 14 only places 20 and 21; 16–21 lie halfway between their neighbours (set by mshGen); 15 is
  not used by mshGen. In aileron slices mshGen places 4, 5, 6 at the cut itself.
- **Exterior vertices**: x, z, each on its boundary (top/bottom, C-arc, outlet; 22 is free).
- Both can be **dragged in the plot** with the point-draw tool: airfoil vertices stay on their
  spline, exterior vertices on their boundary.
- **Suggest block vertices** moves the vertices for orthogonal cells, with priority close to the
  airfoil. It judges the block layout (no cells): grid lines normal to the wall, corners near 90°,
  smooth expansion, wake line along the flow. The suggestion is drawn dashed, with a before/after
  table; Apply or Discard. Details: [docs/block_optimiser.md](docs/block_optimiser.md).
- SPLIT: wing surface as wingUp / wingDown.
- **Top view** with the selected slice in red (scaled to the wing and the slices of the case), and the
  **block layout** of that slice, exactly as mshGen generates it. Colours show valid, skewed and folded blocks; hover a block for its number and
  corner angles. Zoom presets for the whole domain, the airfoil and the trailing edge.
- Cells and gradings along the block edges are set in step 4.

**Block vertices and block check** (step 3b, below the plots; the sidebar shows its result as the
status of step 3b):
- Where the vertices come from: mshGen defaults, outdated, or custom.
- Recompute button.
- Warnings for missing mshGen defaults and for vertices off the outer boundary.
- Block check: errors for folded blocks, warnings for skewed ones (corners below 10° or above 170°).

**4. Meshing** (editable)
- **Mesh level** COARSE / MEDIUM / FINE: loads the presets of the targets below. The level also names
  the settings files; it no longer loads mshGen's default cells.
- **Targets** (stored in the case, editable): y+ at the first cell centre, chordwise cells per side
  (min/max), spanwise cells on the wing (min/max), max growth ratio between neighbouring cells, max
  number of cells. **Reset to the level presets** restores them.
- **Flow conditions** U and ν (for y+).
- **Optimise mesh**: computes the cells and gradings of all edges (e00–e10) and of the spanwise layers
  from the targets, in 0.5–2 minutes, with the progress next to the button.
  - y+, the counts, the cell budget and the growth inside every edge and layer are constraints.
  - Leading edge, trailing edge and wake refinement follow from rules: chordwise sizes follow the
    radius of curvature, the TE cell is as small as the LE cell, the wake starts with the TE cell, the
    wake line is graded like the wall, spanwise sizes are refined towards the tip.
  - The outer-side gradings are optimised for non-orthogonality and size jumps on the exact block
    mesh of the root and tip slices. Beyond the tip the blocks inside the airfoil contour count as
    well (e08 and the chordwise grad1 become variables, against the fit of the chordwise cells).
  - A card in the step explains the metrics. **Undo** (under the button) restores the previous cells
    and gradings; the result table (before / after / target) is in the detailed analysis.
  - Details: [docs/mesh_optimiser.md](docs/mesh_optimiser.md).
- **Edges** e00–e10: cells, grad1 (expansion ratio last/first cell on the airfoil side) and grad2
  (outer side), editable; the plots follow. Next to the table, the **edge plot** shows the block
  layout of the selected slice with every block edge coloured by its edge parameter and the cell
  nodes along it (hover for cells and grading; a row selected in the table is highlighted; the
  blocks inside the airfoil contour, with e08, exist only beyond the tip).
- **Spanwise layers**: cells and grading, editable. The table also shows the first and last cell and
  the size jump to the next layer, with a warning above the max growth.
- **Cell count** (exact) against the maximum.
- **Top view** of the wing with the slices, the selected slice and the spanwise cell faces.
- **2D mesh preview of any slice**, computed with blockMesh's own point interpolation, so it is the
  real mesh:
  - colour by cell size, aspect ratio (in the slice, or 3D with the spanwise size), size ratio to
    the neighbours, or smallest corner angle;
  - zoom to the whole domain, airfoil, LE or TE; hover a cell for its values.
- **Detailed analysis**:
  - the result table of the last optimisation: every metric before and after, with its target;
  - every slice in one table: y, layers, cells, y+, cell sizes, aspect ratios, size ratios inside
    and across blocks, non-orthogonality, skewness, smallest angle, folded cells, spanwise size and
    jump; with errors, warnings (checkMesh limits, targets) and known mshGen limitations listed below;
  - the spanwise cell size along the span (slices dashed, the selected slice red);
  - measured cell sizes per edge parameter in the selected slice;
  - plots of the selected slice: chordwise cell length along the airfoil (largest jump), first
    wall-normal cell height, wall-normal and wake distributions.

**5. Build & check**
- **Options**: images (14 ParaView views; on by default), Fluent export (off by default for new
  cases), and the environment modules (Python, OpenFOAM, ParaView).
- **Save and build mesh**: saves the case, runs the block check, then `Allrun.pre` with a live log.
  checkMesh also writes the quality fields (non-orthogonality, skewness, aspect ratio per cell), the
  cell centres and the sets of flagged cells and faces.
- **Mesh (last build)**: the checkMesh summary, and next to it the **cells and faces that fail
  checkMesh** (failed), those it warns about (warning) and **lower-quality cells** (non-orthogonality
  50–70°, skewness 2–4): a table (count, y range) and a top and a side view with the wing outline
  (whole domain or zoomed to the wing; click a legend entry to hide a group; hover for the value and
  position). For a mesh built before this was part of the build: **Analyse the mesh quality** runs
  checkMesh again (about 30 s).
- **Fluent mesh**: zones (type, faces; `inlet` is written as pressure-outlet, so set velocity-inlet
  in Fluent).
- **AI summary of the mesh**: Claude (`claude-opus-5-5`) reviews the mesh: what is good, a suggested
  modelling approach (turbulence model, wall treatment for the y+, ...), what can be improved, things
  to consider and assumptions, a crazy idea, and a motivating end ("Finally, don't forget to bring
  cake to the office!"). **Write AI summary** sends
  technical facts of the case and its build to Claude through Claude Code with your login (no API
  key, see [AI summary](#ai-summary)): geometry, flow, targets, cell counts, checkMesh results, the
  flagged groups with their position, y+ on the walls (no case name, description, user name or file
  paths). The facts sent can be opened below the button. The summary is saved in the case folder
  (`ai_summary.md`) and shown again later, marked when the mesh was built again afterwards.
- **3D view of the wall patches**, with cell edges (drag to rotate, or an image, see
  [3D views and WebGL](#3d-views-and-webgl)), and the images: the root plane (overview, airfoil,
  LE, TE, first cells), the wing from above and below, the planform, the tip, cuts at constant x near
  the tip, and the mesh inside the wing contour beyond the tip (the blocks inside the airfoil, with
  the contour drawn in orange).

**6. Test run** (OpenFOAM simpleFoam on the mesh of the last build; use a COARSE mesh for quick tests:
about 1 s per iteration on 16 cores for 1.4 million cells)
- **Inputs**: cores (slider up to the cores of this machine), free-stream velocity V (the same as U in
  step 4), angle of attack (in the x-z plane), turbulence model (k-ω SST, Spalart-Allmaras, realizable
  k-ε, laminar), iterations, and the spanwise section planes (default 25/50/75/95 % of the half-span).
- **Machine load** (under the cores, live): the busy processes on this machine (all users, e.g. a
  Fluent job). If the chosen cores plus the busy processes exceed the cores of the machine, a warning
  says how many cores are free, and **Save and run** asks for a second click (**Run anyway**). Sharing
  cores slows a parallel run far more than in proportion, because its processes wait for each other at
  every step: next to a 29-process Fluent job, a 34-core test run took 2 minutes per iteration.
- **Settings of the test run**: a table of everything else: steady incompressible RANS (SIMPLE),
  freestream velocity/pressure at the inlet and outlet, no-slip walls with Spalding's wall function
  (any y+), symmetry at the root and the far field, 0.5 % inflow turbulence with ν_t/ν = 10,
  linearUpwind for U and upwind for the turbulence, force coefficients on the half wing (reference
  values exact for the planform rows: the projected area, the mean aerodynamic chord and the moment
  about its quarter chord), the write interval (every iterations/10, at least 20).
- **Save and run**: saves the case and writes `run/` in the case folder (the mesh is linked, earlier
  results are removed), then runs `run/Allrun`: decomposePar, `mpirun simpleFoam -parallel`,
  reconstructPar of the last time. **Stop** asks simpleFoam to write the current iteration and end
  (clicked again after a minute, it kills the run). The progress line shows the iteration, the time per
  iteration and the time left; the log follows the solver. Rebuilding the mesh in step 5 is refused
  while a run uses it; results of an older mesh are marked.
- **Convergence** (live): the initial residuals of all equations (log scale), and Cl, Cd and Cm (pitch)
  over the iterations (hover for values). The ranges follow the data: all residuals, and the force
  coefficients of the last 80 % of the iterations (the first ones are far off); after a zoom they stay
  until the reset tool.
- The results below are read at every write while the solver runs:
  - **y+**: per wall patch, the solution's min / average / max next to the prediction of step 4 (flat
    plate, local chord) and the target; y+ along the airfoil at the selected section.
  - **Pressure distribution**: Cp along the upper and lower side at a section (the row of wall faces
    nearest to the section plane; red x: reversed flow at the wall). The side of a face comes from the
    patch (wingUp / wingDown) or, for a wing surface that is not split, from the geometry: the section
    as the mesh has it (the slice sections interpolated linearly, as blockMesh does; planform rows
    between two slices do not show in the mesh), corrected per row of faces. The faces of a row share
    y up to a small spread on twisted parts; the rows are found from the gaps between them. The tip
    cap is found by its plane (without SPLIT, wingTip also holds the wing surface of the tip layer).
    The share of the upper and lower
    side with reversed flow (chord- and span-weighted; the nose x/c < 0.05 is left out, where the
    attached flow runs forward from the stagnation point), and the wing seen from above: Cp on both
    sides, reversed flow dotted, the section planes dashed.
  - **Contours**: Cp and |U|/V in the section plane around the airfoil.
  - **Tip vortex** (3D): streamwise vorticity ω_x·c_tip/V with the in-plane velocity in two crossflow
    planes, 0.5 and 3 tip chords behind the most downstream trailing edge.
- **AI review of the solution**: Claude reviews the convergence, the plausibility of the solution, the
  effects of the mesh on the solution and the convergence, and gives recommendations. **Write AI review**
  sends technical facts of the run and its mesh (the set-up, residuals, force coefficients, y+ against
  the prediction, Cp at the sections with the stagnation point, suction peak and the cell lengths at
  the LE and TE, reversed flow, the tip vortex, the mesh facts of step 5; no case name, user name or
  paths) through Claude Code with your login (see [AI summary](#ai-summary)). The
  facts can be opened below the button; the review is saved as `run/ai_review.md`.

## Case folder

```
cases/<case>/
  case.json            all settings (single source of truth)
  airfoil.txt          airfoil coordinates
  wing.txt             planform           -+
  settings/*.csv       mshGen settings     |  written from case.json
  generate_mesh.py     mshGen script       |  ("Write case files" or mshgen_case.py write)
  Allrun.pre           build script        |
  images.json          ParaView views     -+  (GUI/tools/visualize_mesh.py)
  Allrun.images        optional, case-specific extra images (called by Allrun.pre --images)
  system/ constant/    OpenFOAM files (blockMeshDict is written by generate_mesh.py)
  output_files/ constant/polyMesh/ fluentInterface/ images/ log.*     generated
  constant/C, constant/nonOrthoAngle, skewness, aspectRatio, wallDistance   generated (checkMesh, step 5)
  ai_summary.md        AI summary of the last build (step 5, optional)
  run/                 test run (step 6): 0/, constant/ (polyMesh linked to ../../constant/polyMesh),
                       system/, Allrun, settings.json, log.*, processor*/, postProcessing/ (forceCoeffs,
                       yPlus, surfaces), the last two written times, ai_review.md (optional)
```

Without the GUI: `./Allrun.pre [--images]` in the case folder rebuilds everything from these files.

## Command line

Same functions as the GUI (Anaconda module loaded):

```
python GUI/mshgen_case.py list
python GUI/mshgen_case.py new my_wing NACA0012           # new wing, default planform
python GUI/mshgen_case.py import generate_mesh_NACA2420_RIGHT_WING.py my_wing2
python GUI/mshgen_case.py validate my_wing
python GUI/mshgen_case.py blocks my_wing                 # block check
python GUI/mshgen_case.py write my_wing
python GUI/mshgen_case.py build my_wing [--images]       # refuses invalid blocks (--force to override)
python GUI/mshgen_case.py check my_wing
```

## Tests

```
python GUI/tests/test_m0.py              # about 2 minutes, no OpenFOAM needed
python GUI/tests/test_m1.py [--openfoam] # with --openfoam about 4 minutes (blockMesh + checkMesh)
python GUI/tests/test_m2.py [--openfoam] # with --openfoam about 5 minutes
```

**test_m10** (`python GUI/tests/test_m10.py [<built case folder>] [--cores N] [--live] [--load-only]`,
needs OpenFOAM): step 6 on one built case: by default a new NACA0012 wing built in a temporary folder
(about 3 minutes); or the mesh of a case folder (e.g. a COARSE mesh), linked and only read.
- The log reader: residuals (the first solve of p per iteration), errors, the end, lines in two parts.
- The machine load: busy processes measured (with free cores: 6 busy loops for a second), the warning,
  and Run on a busy machine (a stand-in app: the first click warns, the second runs; back to normal
  when the load drops). `--load-only` stops after this check (no simulation).
- write_run: the fields per patch, the function objects, the reference values, the wake planes behind
  the most downstream trailing edge.
- Allrun with a stop request after 6 iterations: the solver writes and ends, reconstructPar, exit code 0.
- The readers: residuals and force coefficients every iteration, y+ per wall patch, the surfaces of the
  write (each wall patch, sections, wake planes), Cp along the sections, reversed flow, the tip vortex.
- For a split wing: the upper / lower side from the geometry agrees with the patches (MegAWES COARSE:
  1 of 9,936 faces differs, the trailing-edge face).
- The facts for the AI review hold no case name, folder or user name; a failing CLI gives a readable
  message. With --live: a review with the four sections.

**test_m9** (`python GUI/tests/test_m9.py [--live]`, about 5 minutes, needs OpenFOAM): step 5 on a
new NACA0012 wing.
- The build writes the quality fields and one cell centre per cell; every set of checkMesh becomes a
  group of its size, failed checks are marked failed, lower-quality cells are in their range.
- For an older build, Analyse the mesh quality (run_quality) gives the same groups.
- The facts for the AI summary hold no case name, folder or user name; the Claude Code CLI is
  found, and a failing CLI gives a readable message.
- With --live (calls Claude through Claude Code with your login): a summary with the five sections
  and the closing sentence.

**test_m8** (`python GUI/tests/test_m8.py [--openfoam]`, about 4 minutes, with --openfoam about 12 minutes): the mesh optimiser of step 4
on new NACA0012 (COARSE, MEDIUM) and FFA_W3_211 wings and a copy of MegAWES_wing.
- The Python port of mshGen's edge → block resolution and grading equals mshGen's blocks.
- Every target is met: y+ (wing median, tip face), chordwise and spanwise counts, cells, growth
  ratios. Mean non-orthogonality and faces above 40° are not worse than with mshGen's defaults, the
  largest non-orthogonality stays below 70°, and no slice has folded cells.
- With --openfoam, the COARSE meshes of the new wings are built: cells = estimate, no failed check
  other than the aspect ratio, and y+ measured on the mesh within 15 % of the target.

**test_m7** (`python GUI/tests/test_m7.py [--openfoam]`, with --openfoam about 15 minutes): the block
optimiser of step 3b on new FFA_W3_211 and NACA0012 wings.
- Its own vertices equal mshGen's.
- The suggestion lowers the objective, passes the block check and has fewer cell faces above 40°.
- With --openfoam, both meshes are built: lower average non-orthogonality, and no failed check other
  than the aspect ratio.

**test_m6** (`python GUI/tests/test_m6.py [MegAWES|FFA|swept]`, about 15 minutes, needs OpenFOAM): the
earlier quality optimiser at a fixed cell count (`core/quality.py`, no longer in the GUI).
- Each mesh is optimised with all variable groups, then built.
- checkMesh before and after must show the same cells, non-orthogonality and skewness that are not
  worse, no additional failed checks, and unchanged y+.

**test_m5** (`python GUI/tests/test_m5.py`, about 5 minutes, needs OpenFOAM and ParaView):
- The full chain is driven through the app: new wing, planform, flow, split and an extra slice, the
  mesh optimiser (COARSE, y+ 30), then the build with Fluent and images.
- It checks the cells, y+ measured per wall patch, the Fluent zones, the 14 images and the 3D view.

**test_m4** (`python GUI/tests/test_m4.py [--openfoam]`): the sizing functions of `core/sizing.py`
(the step 4 helpers before the mesh optimiser).
- The wall helper hits y+ 30/50/100 on the block mesh, and the cell budget hits its total.
- The basic MegAWES example is set up only with the helpers for y+ 50, built, and y+ is measured
  per wall patch on the OpenFOAM mesh.

**test_m3** (`python GUI/tests/test_m3.py [--openfoam]`):
- Every preview node is compared with the nearest blockMesh point in the same slice, for MegAWES and
  for new wings (straight, swept with dihedral, aileron).
- The cells per slice are compared with symRoot/symTip.

**test_m2:**
- A slice added at a cell face leaves the cell count unchanged, and removing it restores the
  layers.
- Each edit type is built with blockMesh: extra slice, moved division, dragged vertices, R/c 10,
  far field, aileron.
- A vertex move the block check reports as folding must make blockMesh fail.

**test_m0:**
- For every example script, it runs the original, then imports it as a case and rebuilds it from
  `case.json`.
- The blockMesh lists must be identical and the cell estimate must match.
- The repository is not written to: originals run on a temporary copy of `default_files`.

**test_m1:**
- Planform parameters ↔ table.
- Slices scaled with the span; vertex recompute (edges kept or reset); 2D ↔ 3D.
- New wings for every usable airfoil, and a swept, tapered, twisted wing with dihedral, built with
  blockMesh. The cell count must match the estimate.
- A wing with too much sweep must be refused by the block check.

## Code

- `app.py`: Panel app
- `run_gui.sh`: launcher
- `mshgen_case.py`: command line
- `core/`: logic without GUI dependency
  - `case.py`: case schema
  - `importer.py`: case from an example
  - `mshgen_io.py`: mshGen file formats and defaults
  - `writer.py`: case files
  - `runner.py`: build and checkMesh
  - `estimate.py`: cell count, spanwise sizes, sections
  - `airfoils.py`: airfoil library and what mshGen does per name
  - `planform.py`: planform parameters ↔ table, derived values
  - `geometry.py`: geometry changes, vertex recompute, new wings
  - `blockcheck.py`: inverted, folded and skewed blocks
  - `blocks.py`: slices, airfoil and exterior vertices, domain
  - `blockopt.py`: block optimiser of step 3b ([docs/block_optimiser.md](docs/block_optimiser.md))
  - `model.py`: mshGen in memory
  - `preview.py`: 2D slice mesh (blockMesh's interpolation), cell metrics
  - `meshquality.py`: cells and faces of a built mesh that fail checkMesh or have a lower quality
  - `aisummary.py`: facts of a built mesh, its AI summary and the AI review of a test run (Claude
    through the Claude Code CLI)
  - `simulation.py`: test run of step 6 (simpleFoam set-up, Allrun, log and result readers, facts for
    the AI review)
  - `meshopt.py`: mesh optimiser of step 4 ([docs/mesh_optimiser.md](docs/mesh_optimiser.md)),
    analysis of every slice
  - `sizing.py`: geometric series, y+, sizing functions
  - `results.py`: built mesh: y+, patches, wall surfaces, Fluent zones
  - `views.py`: camera positions of the ParaView images (`images.json`)
  - `quality.py`: quality metrics in the slices, quality optimiser at a fixed cell count (test_m6)
- `tools/visualize_mesh.py`: ParaView images of any case
- `tools/tutorial/`: recorder of the tutorial video
  - `record.py`: scenario and video assembly
  - `browser.py`: headless Firefox via Marionette
  - `page.js`: widget helpers and pop-ups in the page
- `docs/`: tutorial video and its chapters, `block_optimiser.md`, `mesh_optimiser.md`
- `ui/`:
  - `common.py`, `vertexpanel.py`, `view3d.py` (3D views: WebGL probe, server-rendered images)
  - `step1_geometry.py`, `step2_domain.py`, `step3_blocks.py` (3a slicing, 3b blocking)
  - `step4_meshing.py`
  - `build.py` (step 5)
  - `step6_simulation.py` (step 6)
- `templates/`: `generate_mesh.py`, `Allrun.pre`, OpenFOAM files
- `tests/`
