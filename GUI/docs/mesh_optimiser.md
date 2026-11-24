# Mesh optimiser (step 4: *Optimise mesh*)

The button **Optimise mesh** in step 4 computes the cells and gradings of all edges (e00–e10) and of the spanwise
layers from a few targets. It keeps the block vertices of step 3 as they are. The resulting edges and layers are
written to the tables of step 4, where they can still be changed by hand.

Code: [`core/meshopt.py`](../core/meshopt.py). Test: [`tests/test_m8.py`](../tests/test_m8.py).

## Using it

1. Choose the **mesh level**. It loads the presets of the targets (table below) into the inputs. The level also
   names the settings files (`..._COARSE.csv`), as before.
2. Change the targets if needed. They are stored in the case (`meshing.targets`), so a saved case keeps them.
   **Reset to the level presets** restores the presets.
3. Set the **flow conditions** (U, nu): they turn the target y+ into a first cell height.
4. Click **Optimise mesh**. It runs in the background for 0.5–2 minutes; the progress is shown next to the button.
5. The cells and gradings are written to the case (not saved yet). **Undo**, under the button, restores the previous
   cells and gradings. The result table, at the top of the detailed analysis, compares the mesh before and after
   with the targets; a target that is not met is marked *NOT MET*.
6. The tables (edges, layers), the top view with the spanwise cell faces, the preview and the detailed analysis
   follow every change, also a change by hand.

### Targets and presets

| Target | COARSE | MEDIUM | FINE |
|---|---|---|---|
| y+ at the first cell centre (median of the wing's first cells) | 100 | 30 | 1 |
| chordwise cells per side (LE → TE), min–max | 60–100 | 120–200 | 200–400 |
| spanwise cells on the wing (root → tip), min–max | 30–60 | 60–120 | 100–250 |
| max growth ratio between neighbouring cells | 1.25 | 1.2 | 1.15 |
| max number of cells | 1.5 M | 5 M | 30 M |

The presets are constants (`PRESETS`) at the top of `core/meshopt.py`.

## What is optimised

The optimiser works in three stages. The first is closed form (rules), the second a search on the exact mesh, the
third a correction.

### 1. Sizing: cells and airfoil-side gradings from the targets

**Wall-normal (e00).** The first cell height h follows from the target y+ at the cell centre (h/2), with the
flat-plate skin friction of the root chord: Cf = (2 log10 Re − 0.65)^−2.3, u_τ = U √(Cf/2), h = 2 y+ ν / u_τ.
The number of cells is the smallest one with a growth ratio ≤ the maximum over the median wall-normal edge length.

**Chordwise (e03–e07: cells and grad1).** A size field along each airfoil side (suction, pressure):
- the cell size is C × the local radius of curvature, at most C × one chord, so the cells are small where the wall
  turns fast (leading edge). C is then the turning angle per cell (radians) where the curvature limits the size;
- the trailing edge gets the size of the smallest (leading-edge) cell (`TE_RATIO = 1`): the TE has no curvature
  but needs fine cells for the wake and the pressure recovery;
- the field grows by at most 0.85 × (max growth − 1) per cell length (`GROWTH_MARGIN`), so the fitted series stay
  below the maximum;
- C is chosen so that the segments add up to the chordwise cells per side. Each segment gets the larger count of its
  two sides, because both sides share the same edge parameter.

Each segment is one geometric series (blockMesh), so its count and grading are then fitted to the field: least
squares of ln(cell size / target size) over the cells of both sides, with growth inside a segment limited to the
maximum (hard bound) and size jumps at the segment ends above the maximum penalised. Cells can move between
neighbouring segments while that improves the fit; the total stays the same.

**With slices beyond the tip** (flow layers), grad1 of e03 and e07 stays within 1/3 … 3
(`INTERFACE_GRADING`). Beyond the tip mshGen fills the airfoil contour with blocks and joins them with a
non-conformal interface (`mergePatchPairs`, LIMITATIONS.md item 9): blocks 1–2 are graded across the airfoil
thickness with e03, blocks 3–6 with e08 and e07. With strongly graded e03/e07, blockMesh can fail in OpenFOAM's face
cutting ("Duplicate point found in cut face"). Tests with blockMesh on an FFA_W3_211 wing (24.11.2026):
- failed: e03/e07 grad1 34.6 / 0.066 (the unlimited optimiser), and 1 / 0.066 with 10 / 45 cells; with the limit 10
  the optimised wing also failed;
- built: 10 / 0.1, 5 / 0.2, 1 / 0.066 with 20 / 40 cells, and all 24 combinations of grad1 1/3 or 3 for e03 and e07
  with 10–60 cells on e03 and 20–90 on e07;
- built with a graded e08 (10 or 40 cells, grading 1.2, 5 or 20) and e03/e07 at the limits: all 12 combinations.

The fit then reaches the TE and LE refinement more through the cell counts of these segments.

**Wake (e02, e01: cells and grad1).** The near wake (TE → vertex 22) starts with the TE cell and grows by at most
the maximum, up to the largest airfoil cell (the near wake keeps the airfoil resolution). The far wake (22 →
outlet) continues from the last near-wake cell with the same growth limit.

**Wake line (e10, e09).** e10, the vertical grading at the near-wake end, starts equal to e00: the cells at the
wake line are as thin as at the wall, and the near-wake blocks are not sheared between their TE edge (graded with
e00) and their downstream edge. e09 (outlet) starts with cells as high as the streamwise ones. Both are then
optimised (stage 2).

**e08** (only beyond the tip). The blocks inside the airfoil contour (3, 6, 7, 10, 11, 12) are graded across the
outer part of the airfoil thickness with e08, from the contour inwards; further inside, blocks 4, 5, 8, 9 use e07.
At the contour they meet the wall-normal cells of the blocks around the airfoil, at the other end the e07 cells. So
e08 is one geometric series from the wall-normal first cell (e00 at the tip slice) to the first e07 cell inside,
with the growth within the maximum (otherwise it grows by the maximum up to that size). After the y+ correction
both ends are matched on the exact mesh of the tip slice (medians over the blocks: first e08 cell = wall-normal
first cell outside, last e08 cell = first e07 cell inside); the quality search then refines the grading (stage 2).
mshGen's defaults use 4 uniform cells there (MegAWES: the first cell inside is 66 times the wall-normal cell
outside, LIMITATIONS.md item 23). The cells of e08 count in the cell budget. Without slices beyond the tip e08 is
not used and set to 1 cell.

**Spanwise layers (cells and grading).** A size field along the span: uniform H on the wing, the y+ cell at the
tip slice (first spanwise cell of the tip face, same y+ target with the tip chord), at most the outermost
wall-normal cell beyond the tip, growing by at most 0.85 × (max growth − 1). H is chosen so that the wing has the
spanwise cells of the target. The layers are fitted like the chordwise segments (one series per layer, growth inside
a layer ≤ the maximum, jumps at the slices penalised, cells moved between neighbouring layers on the same side of
the tip). The two layers next to the tip match the tip cell exactly; if that needs more growth than allowed, they
get more cells.

**Counts in the cell budget.** The counts use the same fraction f of both ranges: chordwise = min + f (max − min),
spanwise likewise. f = 1 if the cells stay ≤ the maximum, otherwise the largest f that fits (bisection). If even
the minimum counts exceed the budget, the minimum counts are used and the result table marks the cells as not met.

### 2. Quality on the exact block mesh

The outer-side gradings decide how the grid lines run from the wall to the outer boundary. They are optimised on
the **exact block mesh** of the root and tip slices (2D: the slice), computed with blockMesh's own interpolation
(`core/preview.py`), for the blocks around the airfoil (13–26). With slices beyond the tip, the tip slice also
includes the **blocks inside the airfoil contour** (0–12) and the faces across the contour.
mshGen is called once; for every evaluation the block resolutions and gradings are set from the edges in Python,
as `Mesh.generate_blocks` does it (checked in test_m8). One evaluation takes about 0.1–0.4 s.

**Variables** (as ln(grading)):
- grad2 of e01–e07 (outer side of the wake and wall blocks) and the gradings of e09, e10;
- beyond the tip also the grading of e08 and grad1 of e03–e07: the blocks inside the contour take their cells
  across the thickness from these edges (blocks 0–2 from e03 in both directions, 4, 5, 8, 9 from e07).

Bounds: the growth inside an edge stays ≤ the maximum, |ln g| ≤ (n − 1) ln(max growth); e03/e07 grad1 within
1/3 … 3 (merged interface); e08 keeps its first cell within the growth ratio of the wall-normal cell outside.

**Objective** (smaller is better), over all faces between cells, inside blocks and across block boundaries:

| Term | Weight | Definition |
|---|---|---|
| non-orthogonality, mean | 1 | Σ w (θ/45°)² / Σ w; θ: angle between the line joining the cell centres and the face normal (checkMesh's definition in the slice plane); w = 1 / (1 + d/c)², d: distance of the face from the airfoil, c: chord of the slice |
| non-orthogonality, tail | 0.25 | mean of (θ/45°)² over the worst 1 % of the faces |
| severe faces | 1 | Σ ((θ − 65°)/45°)² over the faces above 65° (checkMesh warns above 70°) |
| size jumps, mean | 1 | Σ w e² / Σ w; e = max(0, ln r − ln g_max) / ln g_max; r: size ratio of the two cells normal to the face |
| size jumps, tail | 0.05 | mean of e² over the worst 1 % of the faces |
| inside the contour | 0.5 | beyond the tip: the five terms above for the faces inside blocks 0–12 and across the contour (tip slice; all faces weighted equally) |
| chordwise sizes | 1 | with grad1 of e03–e07 as variables: their fit to the chordwise size field of stage 1 (the cost of the segment fit: ln deviations of the wall cells, jumps at the segment ends) |

The interior and the chordwise terms make the trade-off between the cells inside the contour beyond the tip and
the chordwise cells on the wing explicit. The result table shows the quality objective without the chordwise term.

**Starts:** five sets of outer gradings are evaluated and the best one is the start of the search:
- grad2 = grad1: the outer nodes at the same fractions as the wall nodes;
- normal projection: each wall (or wake-line) node is projected along its normal onto the outer edge of its block,
  and one geometric distribution is fitted to these points (only where at least 80 % of the normals reach the
  outer edge: with exterior vertices far downstream, as in MegAWES, they do not);
- smooth outer boundary: the cell size continuous along the outer boundary (least squares of the size ratios at
  the outer vertices);
- uniform outer edges;
- the case's current gradings.

**Search:** Hooke–Jeeves pattern search, at most 250 evaluations (350 with the variables beyond the tip): per variable a step up and down (kept if the
objective improves), then a pattern move along the last successful direction (it follows diagonal valleys, where
two gradings must change together), steps halved when nothing improves. Only improvements are kept.

### 3. y+ correction

e00 is regraded so that the median y+ of the first cells of all wing slices, measured on the exact block mesh with
the local chord of each slice, equals the target (first cell heights scale with the first-cell fraction of e00). If
the growth would exceed the maximum, e00 gets more cells. e09/e10 keep their first cells.

## Report

The result table (before / after / target) shows:
- **constraints:** y+ on the wing (median, min/max) and on the tip face, chordwise and spanwise counts, cells,
  largest growth ratio wall-normal, chordwise along the airfoil (also across the segment ends), along the wake line
  (from the TE cell), spanwise (inside layers and at the slices);
- **resolution:** LE and TE cell / chord, the largest turning angle between neighbouring wall faces (curvature
  resolution at the LE), the largest near-wake cell / chord, the vertical cell at the near-wake end / chord;
- **beyond the tip:** the first cell inside the airfoil contour (e08) over the wall-normal first cell outside it,
  within 1/max growth … max growth; inside the contour (tip slice): mean and largest non-orthogonality, faces with a
  size ratio above the max growth, the largest size ratio;
- **quality** (root and tip slices, blocks around the airfoil): mean and largest non-orthogonality, faces above 40°,
  faces with a size ratio above the max growth, the largest size ratio, the objective.

The **detailed analysis** below the preview lists every slice (y, layers, cells, y+, cell sizes, aspect ratio in
the slice and 3D, size ratios inside and across blocks, non-orthogonality, skewness, smallest angle, folded cells,
spanwise cell size and jump) with warnings: folded cells (error), checkMesh limits (non-orthogonality 70°, skewness
4, aspect ratio 1000), growth above the target, y+ more than 20 % off the target, spanwise jumps. Size jumps across
the blocks inside the airfoil contour and the non-orthogonality of the TE wedge beyond the tip are shown as
information (known mshGen limitations).

## Results

Default targets of the level, new wings (mshGen default vertices; slices 0, 4, 5 (tip), 6, 10 m) and MegAWES_wing
(its own slices), against mshGen's default cells (COARSE) of the same wing (test_m8, 24.11.2026). Non-orthogonality:
root and tip slices, blocks around the airfoil.

| Wing, level | Cells | y+ median | Mean non-orth. | Faces above 40° | Max non-orth. | e08 | Time |
|---|---|---|---|---|---|---|---|
| NACA0012, COARSE | 1,246,600 | 99.9 | 15.0° → 10.4° | 12.9 % → 0.0 % | 59.0° | 25 cells | 56 s |
| NACA0012, MEDIUM | 4,621,043 | 30.0 | 15.0° → 10.2° | 12.9 % → 0.0 % | 64.8° | 85 cells | 83 s |
| FFA_W3_211, COARSE | 1,331,420 | 100 | 22.2° → 16.7° | 27.8 % → 0.2 % | 41.8° | 27 cells | 53 s |
| MegAWES, COARSE | 1,426,932 | 100 | 23.8° → 19.4° | 24.5 % → 8.9 % | 65.3° | 51 cells | 53 s |

All targets are met: y+ within 0.1 %, counts in their ranges, cells below the maximum, growth ratios at most the
maximum in every direction, and beyond the tip the first cell inside the airfoil contour within the growth ratio of
the wall-normal first cell outside it (0.82–1.22). The counts are at the maximum of the ranges, except NACA0012
MEDIUM (52 %) and MegAWES (81 %), where the cell budget limits them (the e08 cells beyond the tip count as well).

**Inside the contour beyond the tip** (MegAWES, tip slice; its current settings → optimised): non-orthogonality
mean / max 11.5° / 66.3° → 10.9° / 66.4°, faces with a size ratio above 1.25: 8.4 % → 1.4 %, largest size ratio
98.9 → 6.7. The remaining jumps of 4–7 are at the trailing-edge wedge (blocks 0–2, and from them across the contour
to the wall-normal cells): these blocks take their cells in both directions from e03, the TE segment of the
chordwise cells, whose grading is limited to 3 by the merged interface. Between the e08 and e07 blocks the jump is
1.4; between the e08 blocks and the leading-edge blocks 11, 12 up to 3.0, because one e08 grading serves edges of
different lengths.

**checkMesh** of the built COARSE meshes (blockMesh, test_m8 --openfoam):

| Wing | Cells | Max non-orth. | Average non-orth. | Max skewness | y+ measured (wing / tip face) | Failed checks |
|---|---|---|---|---|---|---|
| NACA0012 | 1,246,600 | 68.5° | 13.83° | 1.37 | 98.8 / 100.0 | aspect ratio (10,516) |
| FFA_W3_211 | 1,331,420 | 83.3° | 17.85° | 1.85 | 96.5 / 100.0 | aspect ratio (9,339) |

The largest non-orthogonality of the FFA mesh is at the trailing-edge wedge beyond the tip (block 0, the TE angle
of the airfoil, LIMITATIONS.md item 24; mshGen's default cells give 82.9° there as well). The high aspect ratio
cells are the thin wall cells for y+ 100 against the spanwise cells, and the thin first spanwise cells at the tip
face against the chordwise cells.

**MegAWES_wing**, optimised with the COARSE targets and built: 1,426,932 cells (its current settings: 2,574,576
cells, y+ about 50). checkMesh: max non-orthogonality 76.0°, average 18.39°, max skewness 1.99; the
non-orthogonality check passes (9 faces above 70°, a warning); only the aspect ratio check fails (9,076). y+
measured on the mesh: wingUp 100.1, wingDown 94.4, wingTip 100.0.

The 9 faces above 70° are beyond the tip, in the last (longest, 0.46 m) spanwise cell of the first flow layer, on the
upper airfoil contour between two thin cells (the wall-normal cell outside and the first e08 cell inside, both about
1 mm). For these cells OpenFOAM's cell centres differ by about 2.6 mm in the spanwise direction, which gives the
angle; in the slice plane the faces are below 67°. With mshGen's 4 uniform e08 cells (one thick cell inside) the
maximum was 66.4°.

## Limitations

- **Aspect ratio.** With a small y+ the wall cells are very thin compared with the spanwise cells (and the first
  spanwise cell at the tip is very thin compared with the chordwise cells), so checkMesh reports high aspect ratio
  cells (> 1000). This is inherent to wall-resolved meshes of extruded blocks; the detailed analysis shows where.
- **Edge parameters are shared.** One edge parameter sets both the suction and the pressure side (and the
  top and bottom wake blocks), so the sizes cannot follow both sides exactly where their segment lengths differ.
- **The block vertices are not changed.** The layout of step 3b (or its block optimiser) decides how well the
  outer gradings can align the grid lines with the wall normals.
- **Only the root and tip slices** are evaluated in stage 2; the exterior vertices are the same in every slice, so
  for tapered wings the result is a compromise.
- **y+ is an estimate** (flat-plate skin friction); the real y+ follows from the flow solution.
- **Thin cells meet long spanwise cells beyond the tip.** e08 refines the inside of the contour like the wall
  outside it; where the spanwise cells beyond the tip are long, a few faces across the contour can exceed 70°
  (MegAWES: 8 faces, 74.7°).
- **Trailing-edge wedge beyond the tip.** Blocks 0–2 inside the contour take their cells in both directions from
  e03; with the merge limit on e03 their cells stay 4–7 times larger than the wall-normal cells next to them.
- **Merged interface beyond the tip.** The limit on e03/e07 is empirical: OpenFOAM's face cutting of the
  non-conformal interface can still fail for other combinations. Build the mesh (step 5) to be sure.

## Parameters (`core/meshopt.py`)

| Name | Value | Meaning |
|---|---|---|
| `PRESETS` | table above | targets per level |
| `RHO_CAP` | 1 chord | chordwise size follows the radius of curvature up to this radius |
| `TE_RATIO` | 1 | TE cell / smallest (LE) cell |
| `INTERFACE_GRADING` | 3 | with flow layers: e03/e07 grad1 within 1/3 … 3 |
| `GROWTH_MARGIN` | 0.85 | size fields grow by 0.85 × (max growth − 1) per cell |
| `W_CHAIN`, `W_PIN` | 50, 100 | weights of junction jumps and of the tip cell in the segment / layer fits |
| `W_ORTH`, `W_ORTH_MAX`, `W_SEVERE` | 1, 0.25, 1 | non-orthogonality terms |
| `W_GROWTH`, `W_GROWTH_MAX` | 1, 0.05 | size jump terms |
| `W_INTERIOR`, `W_WALL` | 0.5, 1 | beyond the tip: the blocks inside the contour; the fit of the chordwise grad1 |
| `TAIL`, `SEVERE` | 1 %, 65° | tail share, threshold of the severe faces |
| `FOCUS` | 1 chord | distance scale of the face weights |
| `STEP` | ln 2 | first step of the pattern search (ln of the grading) |
| `max_evals` | 250 (350 beyond the tip) | evaluations of the pattern search |
