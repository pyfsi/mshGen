# mshGen GUI – plan

Living document: update it as milestones are done or the plan changes.

## Decisions

| Topic | Decision (date) |
|---|---|
| GUI framework | Panel (in the browser), served from the cluster, opened through VS Code port forwarding (17.11.2026) |
| Case folders | `cases/<case>/` in the repository root; `MegAWES_wing` moved to `cases/MegAWES_wing` (17.11.2026) |
| mshGen core | No changes for now. Limitations and suggested improvements are listed in `../LIMITATIONS.md` (17.11.2026) |
| Airfoils | One airfoil along the whole span (may be extended later) (17.11.2026) |
| Airfoil choice | Only mshGen's existing airfoils; adding new ones later (LIMITATIONS.md item 18) (17.11.2026) |
| First milestone | M0 (17.11.2026) |

## Design principles

- **mshGen stays the engine.** The GUI writes the inputs the examples use (airfoil file, wing file,
  the three settings CSVs, `params`/`flags`) and calls the unchanged `Mesh` class. `Mesh` reads its
  CSVs from the folder it is given, so every case keeps its own settings.
- **One folder per mesh**, rebuildable without the GUI:
  ```
  cases/<case>/
    case.json            all settings (single source of truth, reloadable in the GUI)
    airfoil.txt          airfoil coordinates
    wing.txt             planform table, written from case.json
    settings/            interior/exterior points + resolution/grading CSVs, written from case.json
    generate_mesh.py     standalone mshGen script, written from case.json
    Allrun.pre           mshGen -> blockMesh -> checkMesh -> foamMeshToFluent
    system/ constant/    OpenFOAM case files (blockMeshDict written by generate_mesh.py)
    output_files/ images/ fluentInterface/ log.*   generated
  ```
- **Logic separate from the interface.** `GUI/core` has no GUI dependency, is used by the GUI and
  by the command line (`GUI/mshgen_case.py`), and is tested in `GUI/tests`.

## Folder layout

```
GUI/
  PLAN.md               this file
  README.md             how to start and use the GUI
  run_gui.sh            module purge/load + panel serve (localhost, password login)
  app.py                Panel app: case selection, steps, build
  mshgen_case.py        command line: list / import / validate / write / build / check
  core/
    case.py             case schema <-> case.json
    importer.py         case from an example script (records the Mesh inputs, no meshing)
    mshgen_io.py        mshGen file formats (settings CSVs, wing file), mshGen defaults in a temp folder
    writer.py           writes a case folder (wing, settings, generate_mesh.py, Allrun.pre, system, constant)
    runner.py           runs Allrun.pre, streams the log, parses checkMesh
    estimate.py         cell count, spanwise cell sizes, section placement
    airfoils.py         airfoil library, what mshGen does per airfoil name, airfoil data/splines (M1)
    planform.py         planform parameters <-> table, derived values, 3D surface (M1)
    geometry.py         geometry changes: vertex defaults/recompute, span->slices, 2D/3D, new wing (M1)
    blockcheck.py       inverted/folded/skewed blocks in any slice, before blockMesh (M1, M2)
    blocks.py           slices, chordwise divisions, exterior vertices, domain size (M2)
    model.py            mshGen in memory: vertices, blocks, edges (M3)
    preview.py          2D mesh of a slice with blockMesh's interpolation, cell metrics (M3)
    sizing.py           geometric series, y+, sizing functions (M4; the GUI helpers were replaced by meshopt.py)
    meshopt.py          mesh optimiser of step 4: all cells and gradings from targets; every-slice analysis
    results.py          built mesh: y+ per wall patch, patches, wall surfaces, Fluent zones (M4, M5)
    views.py            images.json: camera positions/zoom windows of the ParaView images (M5)
    quality.py          quality metrics (checkMesh non-orthogonality/skewness in the slices), optimiser (M6)
    blockopt.py         block optimiser of step 3b
  templates/            generate_mesh.py, Allrun.pre, OpenFOAM system/ and constant/ files
  tools/                visualize_mesh.py: ParaView images of any case (pvbatch, images.json)
  ui/                   common.py, vertexpanel.py, step1_geometry.py, step2_domain.py, step3_blocks.py,
                        step4_meshing.py, build.py
  tests/                test_m0.py: all examples reproduced through case.json; test_m1.py: geometry;
                        test_m2.py: domain and blocks; test_m3.py: preview against blockMesh;
                        test_m4.py: sizing helpers, automatic y+ 50 setup; test_m5.py: full chain in the app;
                        test_m6.py: optimiser, checkMesh before/after; test_m7.py: block optimiser;
                        test_m8.py: mesh optimiser of step 4
```

## Notes from M0

- `case_from_example` imports any example (2D/3D, aileron, DEFAULT=True/False); with DEFAULT=True the
  defaults are computed in a temporary folder, so `meshGeneration/default_files` is never written.
- The airfoil name must stay the mshGen name (e.g. `Mrev-v2`) because mshGen selects built-in
  behaviour by name (LIMITATIONS.md item 3). M1 has to decide how new airfoils get a name/behaviour.
- Case-specific extra images: an executable `Allrun.images` in the case folder is run by
  `Allrun.pre --images` (MegAWES_wing: ParaView views and chordwise cell sizes). These scripts have
  MegAWES-specific camera positions; generic versions belong in M5.
- `.gitignore`: `cases/` is ignored as a whole (user decision 17.11.2026).

## Notes from M1

- Step 1 edits an in-memory copy of the case; **Save** writes case.json and all files, **Revert** reloads.
  Slow work (mshGen defaults, ~4 s) runs in a background thread.
- Airfoil library (`core/airfoils.py`): Mrev-v2, NACA0012, NACA2420, FFA_W3_211; Mrev-v2_aileron only
  via its examples (mshGen's default vertices cross for it). For each airfoil the GUI shows what mshGen
  does with that name and which mesh_scale/levels have default resolutions.
- Planform: parameters (half-span, root/tip chord, constant-chord part, LE sweep, twist root/tip,
  dihedral) generate the table; or edit the table directly (insert/delete rows). Changing the span
  scales the slices (option).
- Block vertices follow the geometry: they are mshGen's defaults for (airfoil, root section, COG,
  mesh_scale, mesh_level), stored with that key in case.json (`blocks.vertex_source`, `defaults_key`)
  and recomputed on Save when the key changes. The edge resolution/grading is kept unless the airfoil,
  mesh_scale or mesh_level changed (the tuned MegAWES e00 etc. survive geometry edits).
- **Block check** (`core/blockcheck.py`, 0.1 s, live in step 1, on Save, before Build): every active block
  must be a counter-clockwise quadrilateral in every slice. It found the main geometric limit of mshGen:
  exterior vertices do not follow the sections, so the TE must stay upstream of the near-wake vertex
  (LIMITATIONS.md item 7). The top view shows that line.
- For M2 (domain, blocks): the block check is the basis for editing vertices and slices; the near-wake
  limit could be lifted by shifting exterior vertices per slice, but that needs mshGen changes.

## Notes from M2

- Step 2 (domain): R/c with the levels mshGen has defaults for, far field (last slice), U and nu
  (Re, flat-plate u_tau, first estimate of the first cell and its y+ from e00: 45 for MegAWES, measured
  ~50). Plots: root plane with exterior vertices, domain from above with planform, slices and patches.
- Step 3 (blocks): slices (add at the nearest cell face = mesh unchanged; delete = layers merged;
  move; suggest: root, planform kinks merged within 0.1 half-span, tip +/- 0.1 half-span, far field),
  chordwise divisions (camber vertex follows), all 16 airfoil vertices (advanced), exterior vertices
  (table, or drag in the plot; projected onto their boundary: top/bottom z = +-R, C-arc radius R,
  outlet x = 2R, corners fixed), aileron layers, SPLIT/ARTIFICIOUS.
- Block layout plot (Bokeh) per slice: blocks coloured by status, hover with block number and corner
  angles, edge names e00-e10 (airfoil edges only when zoomed in), zoom presets.
- Block check extended with skewness (corner angles < 10 or > 170 deg = warning); this flags the new
  FFA_W3_211 wing (corners down to 7 deg), which checkMesh also reports (non-orthogonality 83 deg).
- Shared vertex panel (steps 1-3): vertex source, recompute, default availability, exterior vertices
  off the boundary (custom vertices after an R/c change), block check.
- URL parameters: `/app?case=<name>&step=<1-5>`.
- Verified in a headless browser: page layout of steps 1-3; the browser -> server path of a vertex
  move (projection, block redraw, "unsaved changes"). Not verified: an actual mouse drag with the
  point-draw tool (needs a real browser); the exterior-vertex table is the fallback.
- Open for M4: spanwise sizing at the far field (moving the far field to 15 m on the test wing gives
  aspect ratio 1600 in 1528 cells); suggested slices get first-guess cell counts only.

## Notes from M3

- Step 4 (meshing, manual): e00-e10 cells/grad1/grad2 and spanwise cells/grading per layer in editable
  tables (first/last spanwise cell and the size jump to the next layer derived), estimated cell count.
- Mesh preview (`core/preview.py`, `core/model.py`): mshGen runs in memory (vertices, blocks, edges, no
  files, ~0.3 s); each block face of the slice is meshed with blockMesh's own algorithm, taken from the
  OpenFOAM-v2112 source (`src/mesh/blockMesh/blocks/block/blockCreate.C`, block::createPoints: weighted
  straight-edge points of all 12 edges + curved-edge corrections). A plain transfinite interpolation was
  off by up to ~1 cell in blocks whose opposite edges are graded very differently (far wake: 38 vs 3800).
  Result: preview = blockMesh to 4e-7 x local cell size (MegAWES, 4 slices; new test wings in test_m3).
  Curved block faces (projection) are not used by mshGen, so their correction is not implemented.
- Cell metrics per cell: size, aspect ratio in the slice and in 3D (with the spanwise size at the slice;
  MegAWES tip slice 7840 vs checkMesh 7535), neighbour size ratio (inside blocks), smallest corner angle,
  folded cells. Along the airfoil: chordwise length and first wall-normal height per cell, largest jump
  (reproduces the e04 jump at the former aileron hinge: 3.49 with grad1 0.2, 1.14 now).
- Plots: preview (colour by size / aspect ratio / 3D aspect ratio / neighbour ratio / angle; zooms whole,
  airfoil, LE, TE), surface cell lengths, first cell height, wall-normal and wake distributions, spanwise
  cell size; tables: slice summary, measured sizes per edge parameter.

## Notes from M4

- Helpers (`core/sizing.py`, `ui/helpers.py`), each a proposal (old -> new + predicted result measured
  on the block mesh of a copy of the case) that is applied on request:
  - wall y+: y+ at the first cell centre, flat-plate u_tau with the local chord; e00 solved for the
    median (or max) over all wing slices, keeping the wall-normal cells or the fewest cells below a growth
    ratio; e09/e10 gradings scaled with e00. First cells scale exactly with the first-cell fraction of e00,
    so the target is hit (within 1 %).
  - chordwise: TE size, LE size, largest size, growth -> e03-e07 (cells, grad1) via a size function along
    the airfoil arc; sizes continuous at the divisions.
  - wake: TE size, growth, largest size -> e02, e01.
  - spanwise: root size, tip size (= y+ of the tip face), far-field size, growth -> every layer; targets
    reduced where a layer is too short for the growth limit; continuous at the slices.
  - cell budget: all counts scaled to a total; optionally keeping the wall and tip first cells.
- Step 5 measures y+ per wall patch on the built mesh (`core/results.py`, wall distance of the first
  cells, same u_tau as the prediction): MegAWES measured wingUp 51.0 / wingDown 48.4 / wingTip 52.8,
  predicted 51.2 (wing median) / 52.8 (tip).
- Defaults of the helper inputs are taken from the current mesh, so a proposal with unchanged inputs
  reproduces it closely (MegAWES: wall 5960 vs 5800 hand-tuned, spanwise 10/36/41/41/30 vs 10/38/47/47/32).
- Not (yet) a helper: grad2 (outer side of the chordwise blocks), e08 (flow layers only), the outer cells
  of e00 (follow from n0, g0; shown in the measured sizes).
- M4 check (basic MegAWES example, helpers only, built): wing y+ median 47.9 measured vs 50.5 predicted,
  wingTip 50.2 vs 50.2, cells exactly as predicted (1,950,503). The predicted wing y+ is ~5 % high: it uses
  the cell height along the grid line from the wall (h/2), the measurement the true wall distance of the
  cell centre. A correction (or the true normal distance in the preview) is a possible refinement.

## Notes from M5

- Step 5 (build & check): options Fluent export (case.json `environment.fluent`), images, environment
  modules (python/OpenFOAM/ParaView, case.json `environment`); results: checkMesh, y+ per wall patch
  measured on the mesh (automatic after a build), patches (type, faces), Fluent zones (types, face counts,
  read from the .msh; hint that `inlet` is written as pressure-outlet), 3D view of the wall patches with
  cell edges (Plotly, from the polyMesh), image gallery.
- ParaView images for any case: `GUI/tools/visualize_mesh.py` reads `images.json` (written by the writer,
  `core/views.py`): patch names, root LE/TE/chord, first wall cell, R, tip section, cut position (45 % of
  the tip chord), tip spanwise cell. Same 13 views as the former MegAWES-only script; the MegAWES case's
  own scripts (Allrun.images, visualize_mesh.py, chordwise_cells.py) were removed (the chordwise check is
  in step 4). `Allrun.pre --images` starts from an empty images folder; a case can still add an executable
  Allrun.images.
- The 3D wall view (Plotly) needs WebGL in the browser (not available in the headless test browser).
- Bug fixed (from M1): step 1 could not draw FFA_W3_211 and NACA2420 (mshGen's RawData keeps the point
  list only for airfoils with a point clean-up); found through images.json, now covered by test_m1.

## Steps in the GUI

Layout: steps on the left, inputs in the middle, a live plot on the right, a status bar with
warnings and the estimated cell count.

1. **Geometry**
   - Airfoil: library (`meshGeneration/airfoils/`), NACA 4-digit generator, or import a file
     (detect Selig/Lednicer order, normalise to chord 1, remove duplicate points, close the TE or warn,
     optional resampling); camber line estimated, editable control points. Plot: airfoil, splines, camber.
   - Planform: parametric (span, root/tip chord, taper break, sweep, twist, dihedral) or table editor.
     Plots: top view, front view, 3D surface. Also `COG` and 2D/3D.
2. **Domain**: C-radius R/c, far-field (`symTip`) distance beyond the tip, flow conditions (U, nu)
   for y+. Plot: 2D C-domain and 3D domain outline. Fixed by mshGen (shown, not editable):
   wake length 2R, near-wake zone 0.5c, sizes based on the root chord.
3. **Blocks**: slices (suggested at root, planform kinks, tip, far field; editable), chordwise
   divisions (block corners on the suction/pressure/camber splines, suction/pressure pairs kept
   aligned), outer vertices (C-boundary and wake). Plot: block layout of a selected slice with block
   numbers, edge names (e00–e10) and warnings for skewed blocks.
4. **Meshing**
   - Manual: e00–e10 cell counts and gradings, spanwise cells and grading per layer, each with a 1D
     cell-size plot.
   - Helpers (fill in the manual values, still editable):
     - y+ -> first cell height (flat-plate Cf) -> n0/g0
     - first cell + far-field cell size -> cell count and grading of wall-normal and wake edges
     - refinement: target sizes at LE, TE, tip, wake; gradings solved for a continuous cell size
       between edges and slices (max neighbour ratio, e.g. 1.2)
     - cell budget: target total cell count, scales all counts
   - Preview: 2D mesh of any slice computed in Python (about a second), coloured by cell size,
     aspect ratio or neighbour ratio, zoom on LE/TE.
5. **Build and check**: write the case, run `Allrun.pre` with live log, checkMesh summary,
   measured y+ per wall patch, 3D view of the wall patches (Plotly), optional ParaView images.
   Optimiser: adjusts vertex positions, gradings and counts (within limits) to reduce the largest
   size jump, aspect ratio and skewness at a fixed cell count, using the fast Python metrics and
   confirming with checkMesh at the end.
6. **Test run** (added 24.11.2026): a short simpleFoam run on the mesh with live convergence, y+, pressure
   distribution, contours and tip vortex, and an AI review of the solution and the effects of the mesh.

## Milestones

| | Delivers | Check | Status |
|---|---|---|---|
| M0 | Skeleton: case.json, writer, runner, command line, minimal app (case selection, summary, build with live log) | `cases/MegAWES_wing` rebuilt from case.json gives blockMesh lists identical to the example script | done 17.11.2026: MegAWES_wing lists, polyMesh and .msh identical; all 15 examples reproduced (`tests/test_m0.py`) |
| M1 | Step 1 geometry (existing airfoils only, decision 17.11.2026) | New wing with a library airfoil and own planform, see the plots, build it | done 17.11.2026: new wings for the 4 usable airfoils and a swept/tapered/twisted wing with dihedral built and checked (`tests/test_m1.py --openfoam`) |
| M2 | Steps 2–3, domain and blocks | Live block-layout plots for each slice | done 17.11.2026: every edit type built with blockMesh (`tests/test_m2.py --openfoam`); block check agrees with blockMesh on a folding vertex move |
| M3 | Step 4 manual mode + 2D mesh preview + cell count | Preview matches the blockMesh result | done 17.11.2026: preview points equal blockMesh's (MegAWES: max 4e-7 x local cell size; `tests/test_m3.py`) |
| M4 | Helpers: y+, boundary size, refinement, cell budget | y+ ~ 50 case reproduced automatically | done 17.11.2026: basic MegAWES example set up only with the helpers, y+ measured on the built mesh (`tests/test_m4.py --openfoam`) |
| M5 | Step 5 build, checkMesh, 3D view, Fluent export | Full chain from the GUI | done 17.11.2026: new wing -> geometry, flow, blocks, helpers -> build with Fluent and images, all through the app (`tests/test_m5.py`) |
| M6 | Quality optimiser | Better or equal checkMesh results at the same cell count | done 17.11.2026: MegAWES, FFA_W3_211 and swept wings optimised and built; checkMesh better or equal, same cells, same y+ (`tests/test_m6.py`) |
| later | mshGen improvements from `../LIMITATIONS.md`, if agreed | Existing examples still produce identical lists | |

## Limits from mshGen's topology (not hidden by the GUI)

- Fixed 27-block C-grid; sharp (closed) trailing edge; one airfoil along the span.
- Refinement only by sizing edges and slices: no local refinement boxes in a structured mesh.
- See `../LIMITATIONS.md` for the full list.

## Notes from M6

- Quality metrics (`core/quality.py`) from the exact slice meshes: non-orthogonality as checkMesh, in the
  slice plane, for faces inside blocks and across block boundaries, with cell area centroids: equal to
  checkMesh's maximum (MegAWES 66.30 vs 66.3, FFA 82.91 vs 82.9, NACA0012 67.49 vs 67.5, NACA2420 58.73 vs
  58.7, Mrev-v2 65.73 vs 65.7). Skewness with checkMesh's formula for internal faces (checkMesh's maximum,
  probably on boundary faces, is ~0.5 higher; used as a guard only), 3D aspect ratio, size ratios inside
  and across blocks, chordwise jump, spanwise jump; where the worst non-orthogonality is.
- Optimiser: coordinate pattern search over the selected groups (gradings: chordwise airfoil/outer side,
  wake, spanwise except the tip layers; divisions; exterior vertices), cell counts fixed. Objective:
  non-orthogonality (worst and mean of the slice maxima) and the size ratios, weighted by a priority;
  a value that gets worse than at the start is penalised, so only improvements are kept. Block check
  for moved vertices. Root, tip and far-field slices (and one more), ~0.4-0.5 s per evaluation.
- Result (test_m6, 250 evaluations, 2 min each, then built):
  - MegAWES: checkMesh skewness 1.949 -> 1.832, aspect ratio 7535 -> 5582, non-orth 66.30 -> 66.29,
    size ratio across blocks 3.9 -> 2.1 (wake vs trailing edge at the tapered slices)
  - FFA_W3_211 wing: skewness 2.077 -> 1.831, failed checks 1 -> 0, spanwise jump 1.36 -> 1.08
  - swept NACA0012 wing: skewness 1.645 -> 1.052, non-orth 67.49 -> 67.36, failed checks 1 -> 0,
    chordwise jump 1.42 -> 1.21
- The worst non-orthogonality is usually at the trailing-edge wedge (block 0) beyond the tip, set by
  the trailing-edge angle; it cannot be optimised away (LIMITATIONS.md item 24). Beyond the tip the
  blocks inside the airfoil contour have a ~100 size jump to the wall-normal cells (item 23); the
  optimiser does not use that value.
- UI: tab "Optimise quality" in the step 4 helpers (groups, priority, evaluations, live progress; the
  proposal shows the changes, the quality before/after and the predicted cells/y+).

## Step 4 simplified: mesh optimiser (24.11.2026)

- Step 4 has a mesh level (presets), editable targets (y+, chordwise and spanwise cell ranges, max growth, max
  cells; stored in `meshing.targets`), U and nu, and **Optimise mesh** (`core/meshopt.py`,
  [docs/mesh_optimiser.md](docs/mesh_optimiser.md)): all edge cells/gradings and the spanwise layers follow from
  the targets (sizing rules, then the outer gradings optimised on the exact block mesh, then a y+ correction).
  The helpers and the "Optimise quality" tab are gone (`ui/helpers.py` removed); `core/sizing.py` and
  `core/quality.py` remain as libraries (tests m4, m6).
- The level no longer loads mshGen's default cells and no longer outdates the block vertices (mshGen's default
  vertices are the same for every level); mshGen defaults are asked with an available level.
- Top view with the spanwise cell faces, spanwise size plot; a "Detailed analysis" section with every slice in one
  table and warnings (checkMesh limits, targets, known mshGen limitations).
- e08 (beyond the tip) starts with the wall-normal first cell at the airfoil contour and grows to the e07 cells inside
  (it was 1 uniform cell: large cells inside the extension of the wing). Edge plot next to the edges table.
- Beyond the tip the quality search includes the blocks inside the airfoil contour (tip slice, weight 0.5), with e08
  and grad1 of e03-e07 as variables against the fit of the chordwise cells; e08 matched at both ends.

## Step 5: flagged cells and AI summary (24.11.2026)

- Allrun.pre: checkMesh writes the quality fields (nonOrthoAngle, skewness, aspectRatio, wallDistance) and
  postProcess the cell centres; no mshGen plots any more (images: ParaView views only).
- Step 5 shows, next to the checkMesh summary, the cells and faces of checkMesh's sets (failed / warning) and the
  lower-quality cells (non-orthogonality 50-70 deg, skewness 2-4) in a table and a top and side view
  (`core/meshquality.py`); older builds: "Analyse the mesh quality" (`runner.run_quality`). The y+ and patch tables
  are gone.
- AI summary (`core/aisummary.py`): technical facts (no names or paths) to Claude (claude-opus-5-5, effort high)
  through the Claude Code CLI of the VS Code extension with the user's login (no API key, no SDK), five sections
  and a fixed closing sentence, saved as ai_summary.md. Test: tests/test_m9.py.

## Step 6: test run with OpenFOAM (24.11.2026)

- Inputs (case.json `simulation`): cores, alpha, turbulence model (kOmegaSST, SpalartAllmaras, realizableKE,
  laminar), iterations, section planes; V is `flow.U` (one value for y+ and the run). Defaults in a table.
- `core/simulation.py` writes `<case>/run/` (polyMesh linked, rewritten for every run): simpleFoam, freestream
  BCs, Spalding wall function (any y+), symmetry at root and far field, linearUpwind for U, upwind for the
  turbulence, residual control; function objects forceCoeffs (every iteration), yPlus, wallShearStress, vorticity,
  Q, raw surfaces (wall, section planes, two crossflow planes behind the most downstream TE). Allrun: decomposePar,
  mpirun simpleFoam, reconstructPar -latestTime. Stop: stopAt writeNow (runTimeModifiable), then kill.
- Live in the GUI (read incrementally): residuals, Cl/Cd/Cm, progress; at every write: y+ (solution vs the step 4
  prediction and the target), Cp at a section, reversed flow (wall shear against the flow behind the nose x/c > 0.05,
  where the attached flow runs forward from the stagnation point; chord- and span-weighted), Cp on the wing from
  above, section contours of Cp and |U|/V, tip vortex (omega_x c_tip/V with in-plane vectors).
- Wall faces -> x/c and side: one surface per wall patch (wingUp/wingDown give the side). Without split, the local
  section from the planform at the face's y, corrected per row of faces (scaled to its LE/TE faces, then a z offset
  a + b x/c fitted where the airfoil is thick): between slices the mesh is blockMesh's interpolation of the slice
  sections, up to 0.6 % of the chord off the planform (MegAWES, 4 slices), more than the thickness near the TE. The
  camber line alone put 148 of 9,936 faces on the wrong side, the corrected geometry 1.
- Cutting planes: OpenFOAM snaps cut points to mesh points; in the long wake cells up to 8 mm off the plane.
- Machine load: a 34-core test run next to a 29-process Fluent job (63 processes, 36 cores) took 2 minutes per
  iteration. Step 6 shows the busy processes (procs_running of /proc/stat, median of the last samples; MPI processes
  count while they wait, CPU usage cannot show this) and warns before a run that would share cores (second click
  runs anyway).
- AI review (`aisummary.write_review`): facts of the run and of the mesh (step 5 facts) to Claude through the Claude
  Code CLI; four sections (convergence, plausibility, effects of the mesh, recommendations); `run/ai_review.md`.
- Developed and tested on one COARSE mesh only (MegAWES, 1.43 million cells; 300 iterations on 16 cores in about
  5 minutes). Test: tests/test_m10.py (one built case; default a new NACA0012 wing).

