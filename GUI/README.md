# mshGen GUI

A browser GUI (Panel) to set up, build and check mshGen wing meshes. Every mesh is a **case folder**
in `mshGen/cases/<case>/`, which can also be rebuilt without the GUI. The plan and status are in
[PLAN.md](PLAN.md); known limitations of mshGen are in [../LIMITATIONS.md](../LIMITATIONS.md).

Status: **M6** (all milestones of the plan). All steps are editable:
- geometry: airfoil, 2D/3D, COG, planform;
- domain: R/c, far field;
- spanwise slicing (3a): slices, aileron cut-out layers, patch `extra`;
- airfoil blocking (3b): airfoil and exterior vertices (tables or dragged), wingUp/wingDown;
- meshing: level, flow conditions, cells and gradings, with sizing helpers (y+, chordwise, wake, spanwise, cell budget) and
  a 2D mesh preview identical to blockMesh's.

The build step runs the case with a live log and shows the checkMesh summary, y+ measured on the
mesh, the Fluent zones, a 3D view of the walls and the images. A quality optimiser (step 4)
improves non-orthogonality, skewness and size jumps at a fixed cell count.

## Tutorial video

[docs/mshGen_GUI_tutorial.mp4](docs/mshGen_GUI_tutorial.mp4) shows every input and option in a
complete session: a new wing case through all five steps, ending with a checked OpenFOAM and
Fluent mesh. Each frame outlines the explained element and shows a pop-up message next to it; there
is no audio.

The chapters (embedded in the MP4, and listed in
[docs/mshGen_GUI_tutorial_chapters.txt](docs/mshGen_GUI_tutorial_chapters.txt)) are: introduction,
window and sidebar, new case, steps 1–5, files.

The video is recorded from the real GUI by `tools/tutorial/record.py`:
- it drives a headless Firefox through the session;
- the 3D views are drawn with Plotly's static export, because headless Firefox has no WebGL.

After changes to the GUI, regenerate it (about 15 minutes; needs OpenFOAM and ParaView for the
build in step 5):

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
   (steps 1, 2, 3a, 3b, 4, 5).
3. The server listens on this machine only (127.0.0.1). The password keeps other users logged in on
   the same machine out.

Stop it with Ctrl+C.

### 3D views and WebGL

The 3D views (step 1 wing, step 5 walls) are Plotly figures that need WebGL in the browser. A browser
on a remote desktop or in an X session often has no WebGL; Plotly then only says "WebGL is not
supported by your browser". The GUI checks the browser when the page opens:
- **with WebGL**: interactive views (drag to rotate, scroll to zoom);
- **without WebGL**: the same surfaces are drawn on the server with matplotlib (3D axes, equal
  scales, orthographic, about 1–2 s per view), with view buttons (3D, top, front, side) and
  azimuth / elevation / zoom sliders. The picture is zoomed to fill the frame.

The sidebar switch **3D views: auto / interactive / image** overrides the choice, and so does the
URL parameter `view3d`, e.g. `/app?case=MegAWES_wing&step=5&view3d=image`. Interactive views are
nicer: in Firefox, WebGL can often be enabled in `about:config` with `webgl.force-enabled = true`.

## In the GUI

**Sidebar**
- **Case**: select a case from `cases/`.
- **Save** writes `case.json` and all case files. If the geometry changed, it first recomputes the
  block vertices. **Revert** reloads the saved case.
- **Status**: unsaved changes, estimated cells, block vertices, block check, problems.
- **New case**: either a **New wing** (library airfoil, default planform, mshGen defaults) or
  **From example** (any script in `examples/`).

**1. Geometry** (editable)
- Case type 2D/3D, airfoil (mshGen library), mesh origin (COG).
- Planform: either **Parameters** (half-span, root/tip chord, constant-chord part, LE sweep, twist at
  root and tip, dihedral), which generate the table, or the **Table** itself (station, y, chord,
  twist, dx, dz; insert or delete rows). Changing the span also scales the slices (option).
- Plots (zoom with the mouse wheel; click a legend entry to hide it):
  - the airfoil as mshGen reads it: points, removed point, splines, camber line;
  - the root section in mesh coordinates;
  - top view with the near-wake line;
  - front view;
  - 3D view (drag to rotate; an image with view controls if the browser has no WebGL).
- **Derived values** (left, under the planform): span, area, aspect ratio, taper, MAC, sweep, twist.
- **Block vertices** follow the geometry: they are recomputed with mshGen's defaults when the
  root section, COG or airfoil changes, either on Save or with the button. The edge resolution is
  kept unless the airfoil changed.
- **Block check**: runs live and shows every block that would be inverted or folded in a slice,
  so blockMesh can't fail on it. With mshGen's fixed exterior vertices, every trailing edge must
  stay upstream of the near-wake line (see `../LIMITATIONS.md` item 7).

**2. Domain** (editable)
- C-grid radius R/c (the mshGen levels available for it are listed).
- Far field (`symTip`) position.
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
- **Top view** with the selected slice in red, and the **block layout** of that slice, exactly as
  mshGen generates it. Colours show valid, skewed and folded blocks; hover a block for its number and
  corner angles. Zoom presets for the whole domain, the airfoil and the trailing edge.
- Cells and gradings along the block edges are set in step 4.

**Blocks** in the sidebar (all steps): a status line for the block vertices and the block check; its
**Details** open by themselves when the vertices are outdated or a block is invalid. Step 3b, where
the vertices are edited, shows the same panel in full:
- Where the vertices come from: mshGen defaults, outdated, or custom.
- Recompute button.
- Warnings for missing mshGen defaults and for vertices off the outer boundary.
- Block check: errors for folded blocks, warnings for skewed ones (corners below 10° or above 170°).

**4. Meshing** (editable)
- **Starting resolution**: mshGen's level (`mesh_level`, e.g. COARSE); only levels mshGen has for
  this airfoil and R/c are offered. Choosing one loads its default cells and gradings (and the
  default block vertices, unless they are custom).
- **Flow conditions** U and ν: Reynolds number, plus a first estimate of the wall-normal first cell
  and its y+; the helpers below use them.
- **Helpers**: each computes a proposal, shows the old → new values and the predicted result
  (measured on the block mesh, next to the current values), and is applied with **Apply proposal**.
  The inputs start from the current mesh.
  - **Wall y+**: target y+ at the first cell centre (median or maximum over the wing), keeping the
    wall-normal cells or limiting the growth ratio. Sets e00; e09/e10 are scaled with it.
  - **Chordwise**: cell size at the TE and the LE, largest cell and growth ratio. Sets e03–e07.
  - **Wake**: first wake cell, largest cell and growth ratio. Sets e02 and e01.
  - **Spanwise**: cell size on the wing, at the tip (this sets the y+ of the tip face) and beyond the
    tip, plus the growth ratio. Sets cells and grading of every layer, continuous at the slices.
  - **Cell budget**: scales all counts to a target total, optionally keeping the wall and tip y+.
  - **Optimise quality**: improves non-orthogonality, skewness and size jumps at a **fixed cell count**
    (the counts and the wall/tip first cells stay as they are, so y+ does not change). Choose what it
    may change: chordwise gradings (airfoil side, outer side), wake gradings, spanwise gradings,
    chordwise divisions, exterior vertices. Also choose the priority (size jumps vs orthogonality) and
    the number of evaluations (about 0.4 s each).
    - It measures on the exact block mesh of the root, tip and far-field slices, using checkMesh's
      definitions in the slice plane.
    - It accepts only changes that make no quality value worse than now, and shows the progress.
    - The proposal lists the quality before and after.
    - Build the mesh (step 5) to confirm the result with checkMesh.
- **Edges** e00–e10: cells, grad1 (expansion ratio last/first cell on the airfoil side) and grad2
  (outer side).
- **Spanwise layers**: cells and grading. The table also shows the first and last cell and the size
  jump to the next layer, with a warning above 1.3.
- **Estimated cell count** (exact).
- **2D mesh preview of any slice**, computed with blockMesh's own point interpolation, so it is the
  real mesh:
  - colour by cell size, aspect ratio (in the slice, or 3D with the spanwise size), size ratio to
    the neighbours, or smallest corner angle;
  - zoom to the whole domain, airfoil, LE or TE; hover a cell for its values.
- **Slice summary and measured cell sizes per edge parameter** (e.g. the first wall cell height).
- **Plots**: chordwise cell length along the airfoil (largest jump), first wall-normal cell height,
  wall-normal and wake distributions, spanwise cell size.

**5. Build & check**
- **Options**: Fluent export, images (mshGen plots and 13 ParaView views), and the environment
  modules (Python, OpenFOAM, ParaView).
- **Save and build mesh**: saves the case, runs the block check, then `Allrun.pre` with a live log.
- **Results**:
  - checkMesh summary;
  - **y+ on the walls** per patch, measured on the mesh (automatic after a build);
  - patches (type, faces);
  - **Fluent zones** (type, faces; `inlet` is written as pressure-outlet, so set velocity-inlet in
    Fluent);
  - **3D view of the wall patches**, with cell edges (drag to rotate, or an image, see
    [3D views and WebGL](#3d-views-and-webgl));
  - the images.

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

**test_m7** (`python GUI/tests/test_m7.py [--openfoam]`, with --openfoam about 15 minutes): the block
optimiser of step 3b on new FFA_W3_211 and NACA0012 wings.
- Its own vertices equal mshGen's.
- The suggestion lowers the objective, passes the block check and has fewer cell faces above 40°.
- With --openfoam, both meshes are built: lower average non-orthogonality, and no failed check other
  than the aspect ratio.

**test_m6** (`python GUI/tests/test_m6.py [MegAWES|FFA|swept]`, about 15 minutes, needs OpenFOAM):
- Each mesh is optimised with all variable groups, then built.
- checkMesh before and after must show the same cells, non-orthogonality and skewness that are not
  worse, no additional failed checks, and unchanged y+.

**test_m5** (`python GUI/tests/test_m5.py`, about 5 minutes, needs OpenFOAM and ParaView):
- The full chain is driven through the app: new wing, planform, flow, split and an extra slice, the
  helpers for y+ 30, then the build with Fluent and images.
- It checks the cells, y+ measured per wall patch, the Fluent zones, the 13 images and the 3D view.

**test_m4** (`python GUI/tests/test_m4.py [--openfoam]`):
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
  - `sizing.py`: sizing helpers and predictions
  - `results.py`: built mesh: y+, patches, wall surfaces, Fluent zones
  - `views.py`: camera positions of the ParaView images (`images.json`)
  - `quality.py`: quality metrics in the slices, optimiser
- `tools/visualize_mesh.py`: ParaView images of any case
- `tools/tutorial/`: recorder of the tutorial video
  - `record.py`: scenario and video assembly
  - `browser.py`: headless Firefox via Marionette
  - `page.js`: widget helpers and pop-ups in the page
- `docs/`: tutorial video and its chapters, `block_optimiser.md`
- `ui/`:
  - `common.py`, `vertexpanel.py`, `view3d.py` (3D views: WebGL probe, server-rendered images)
  - `step1_geometry.py`, `step2_domain.py`, `step3_blocks.py` (3a slicing, 3b blocking)
  - `step4_meshing.py`, `helpers.py`
  - `build.py` (step 5)
- `templates/`: `generate_mesh.py`, `Allrun.pre`, OpenFOAM files
- `tests/`
