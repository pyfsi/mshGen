# Block optimiser (step 3b: *Suggest block vertices*)

The button **Suggest block vertices** in step 3b moves the block vertices of a slice layout so that
the cells will be as orthogonal as possible, with the most weight close to the airfoil. It runs
before any cells are chosen: it judges the **block layout**, not a mesh. The cell counts and
gradings of step 4 are not changed. Optimising the real mesh (cells) is a separate, later step.

Code: [`core/blockopt.py`](../core/blockopt.py). Test: [`tests/test_m7.py`](../tests/test_m7.py).

## Using it

1. Choose what may move:
   - *airfoil and exterior vertices* (default),
   - *exterior vertices only*,
   - *airfoil vertices only*.
2. Choose the **priority close to the airfoil**. It sets how fast the weight of a block corner drops
   with its distance from the airfoil, and how much the far field counts. Settings: *high* (0.25 chord),
   *medium* (1 chord), *low* (4 chords).
3. Click **Suggest block vertices**. The search takes about 20–30 s in the background.
4. The suggested blocks are drawn dashed over the current ones in the block plot (any slice). A table
   compares the current and the suggested layout; smaller is better in every row (see *Report* below).
5. **Apply** writes the vertices to the case: they become *custom*. **Discard** drops the suggestion.
   A suggestion is also dropped when the case changes before it is applied.
6. After applying, check the chordwise cells in step 4. The segment lengths on the airfoil changed
   while the cell counts did not (see *Effects on the cells* below).

## What is optimised

### Variables

| Vertices | Moves along | Variable |
|---|---|---|
| 1, 3, 4, 6, 7, 9, 10, 11 (suction and pressure side) | their spline | position (0 = LE, 1 = TE) |
| 25–28 (top), 34–37 (bottom) | the straight top / bottom, z = ±R | x |
| 30, 31, 32 | the C-arc, radius R | angle |
| 22 (end of the near wake) | free | x, z |
| 23 (outlet, on the wake line) | the outlet, x = 2R | z |

Vertices that do not move:
- **0 (TE) and 13 (LE)** are fixed.
- **29 and 33** are put at the junction of the C-arc and the straight top / bottom (x = 0), so the outer
  boundary keeps its shape.
- **24 and 38** are corners of the domain.
- **Camber vertices 2, 5, 8, 12 and 14** follow their airfoil vertices: each keeps its offset (position
  along the camber line) from the mean of its pair (1–3, 4–6, 7–9, 10–11), as when a division is
  moved by hand.

Placing a camber vertex exactly between both sides is not used: at the TE it puts vertex 2 on the
line 1–3, and block 0 becomes degenerate (folded beyond the tip). The camber vertices only matter
beyond the tip and in aileron layers.

### Evaluation without cells

For every evaluation the vertices of a slice are computed directly from the splines, exactly as mshGen
places them:
- the airfoil vertices from their position along the spline, with the section's chord, twist, dx
  and dz;
- 16–21 halfway between their neighbours;
- the exterior vertices shifted by dz.

The result equals mshGen's vertices to about 1e-15 m (checked in test_m7). One evaluation takes about
5–8 ms for four slices, with no mshGen call.

**Slices evaluated:** the root up to the tip.
- Beyond the tip the section is the tip's, so those slices have the same vertices as the tip slice.
- The tip slice also includes the blocks inside the airfoil contour (0–12) when there are slices
  beyond the tip.
- Slices with the aileron cut are left out: mshGen places 4, 5 and 6 itself there. The final block
  check covers them.

### Objective

All angles are in degrees and normalised by 45°. The objective is averaged over the evaluated slices.

**1. Wall orthogonality**, weight 1. This is the main term.
- Each block next to the airfoil (15–24) has a wall edge a–b on the airfoil and straight edges to the
  outer boundary.
- blockMesh fills the block by transfinite interpolation. With straight side edges, the grid line that
  leaves the wall at fraction s of the wall edge runs straight to the point at the same fraction of
  the outer edge.
- The term compares that line with the local wall normal at 12 points per segment. The point at the
  TE is left out, because a sharp TE has no normal.
- Value: mean of the squared deviations + 0.25 × (largest deviation)².
- This is the angle the first cells at the wall will have, before any grading.

**2. Block corners**, weight 0.5.
- Every corner angle of the active blocks is compared with 90°.
- Weight w = 1 / (1 + d / (λ c))², where d is the corner's distance from the airfoil, c the chord of
  the slice, and λ the priority setting (0.25, 1 or 4 chords).
- Corners on the outer boundary are included: a corner of 90° there means the grid lines meet the
  boundary orthogonally.
- The two wall corners of blocks 15–24 are left out; the wall term measures them with the tangent.
- The blocks inside the contour (beyond the tip, no wall) count with weight 0.1.
- Value: weighted mean of the squared deviations.

**3. Wall spreading**, weight 0.5. Neighbouring wall segments on the same side may differ in length
by up to 3×; beyond that a penalty of (|ln ratio| − ln 3)².

**4. Smooth expansion**, weight 0.25 × λ, so the far field counts less with a higher priority close
to the airfoil.
- The expansion of a block is (outer edge length) / (inner edge length). The inner edge is the wall
  segment, or the wake line for the wake blocks 13, 14, 25 and 26.
- Both edges have the same cell count, so neighbouring blocks with very different expansions give
  jumps in cell size between them in the far field.
- The penalty is (|ln ratio| − ln 3)² from the outlet over the top, and over the bottom, towards the
  LE.
- The nose blocks 19 and 20 are left out. The LE segment is short on purpose (fine cells at the LE),
  so the nose blocks always expand much more than their neighbours.
- Without this term the far-field vertices crowd. For example, all top vertices of a NACA0012 wing
  ended up within 0.02 R, which gives very thin and very wide blocks next to each other.

**5. Wake line**, weight 1. The lines TE → 22 and 22 → 23 follow the flow direction (x). Penalty:
(angle / 15°)². Without it the wake line tilted by about 10° on MegAWES, and the wake cells would no
longer follow the wake.

### Constraints

These are hard limits: a layout that breaks one is not accepted.
- Order along every spline (TE > 1 > 4 > 7 > 10 > LE and the same on the pressure side; camber
  vertices too), with a minimum gap of 0.02 in position.
- Order along the outer boundary, with minimum gaps of 0.05 R on the straight top / bottom and
  0.15 rad (about 9°) on the C-arc. mshGen's defaults use 0.07–0.09 R.
- Near-wake length (TE → 22) between 0.2 and 2 root chords, 22 inside the domain, 23 inside the outlet.
- Every active block valid (no folded or inverted corner) in every evaluated slice.
- After the search, mshGen's own vertices of **all** slices (including aileron slices) must pass the
  block check.

### Search

**Pattern search.** It runs per variable:
- try a step up and a step down;
- keep the step if the objective improves, and grow it by 1.5;
- halve all steps after a sweep without improvement.

Only improvements are kept, and invalid layouts are rejected. A smooth optimiser (Powell) was tried
first, but it failed on the jump at invalid layouts.

**Two starts:**
- the current vertices;
- a layout where the exterior vertices sit where the wall normals of their airfoil vertices meet the
  boundary.

Each start runs up to 3,000 evaluations, and the best result is kept. Positions are rounded to 4
decimals, exterior vertices to 0.1 mm, unless rounding would break a minimum gap.

## Report

The table compares the current and the suggested layout.

**Block values**, over the evaluated slices:
- wall deviation from the normal (largest and mean, per side);
- worst block corner within 1 chord and overall;
- largest expansion ratio between neighbouring blocks;
- wake line angle;
- the objective.

**Cell values**, only for information: the optimiser does not use cells. They use the current cell
counts and gradings of step 4, in the root and tip slice. They come from the exact block mesh
(`core/preview.py`), with checkMesh's definition of non-orthogonality in the slice plane, over the
faces inside the blocks around the airfoil:
- the share of faces above 40°;
- the mean non-orthogonality;
- mean and largest non-orthogonality in the first five rows of cells at the wall.

## Results (priority *high*, 17–23.11.2026)

**Cell values** with mshGen's default cells, root slice:

| Wing | Faces above 40° | Mean non-orth. | First 5 rows at the wall: mean / largest |
|---|---|---|---|
| MegAWES (Mrev-v2) | 27.0 % → 1.2 % | 23.4° → 14.2° | 12.6° / 47.7° → 4.6° / 24.6° |
| FFA_W3_211, new wing | 29.6 % → 10.4 % | 22.5° → 16.8° | 10.3° / 40.5° → 5.3° / 19.0° |
| AP2, new wing | 29.7 % → 7.9 % | 22.3° → 15.8° | 8.8° / 43.3° → 5.5° / 17.7° |
| NACA0012, new wing | 13.3 % → 0.0 % | 14.4° → 13.5° | **1.0° / 5.7° → 5.2° / 14.4°** |

**checkMesh** (blockMesh with the same cells, test_m7):

| Wing | Cells | Average non-orth. | Max non-orth. | Max skewness |
|---|---|---|---|---|
| FFA_W3_211 | 922,880 | 23.2° → 18.1° | 82.9° → 83.3° | 2.08 → 1.76 |
| NACA0012 | 921,760 | 17.7° → 15.5° | 67.5° → 67.9° | 1.62 → 1.63 |

## Limitations and effects

- **The largest non-orthogonality does not change.** It sits in the TE wedge (block 0) beyond the tip
  (LIMITATIONS.md item 24), whose angle is the TE angle of the airfoil. No vertex position changes it.
  The optimiser therefore works on the distribution, not on the worst value.
- **Gradings are not seen.** The wall term assumes the same spacing along the wall edge and along the
  outer edge of a block. In blockMesh, the airfoil-side and outer-side gradings of step 4 (grad1 /
  grad2 of e03–e07) decide which outer point a wall point is joined to, so they also tilt the lines at
  the wall.
  - mshGen's NACA0012 gradings are tuned to its default vertices, which is why its near-wall rows are
    so good (1°).
  - With the suggested vertices and the same gradings they become worse (5°), although the layout
    itself is closer to orthogonal.
  - The cell values in the table show this before applying. The real-mesh optimisation (later, step 4)
    is where vertices and gradings should be tuned together.
- **Effects on the cells.** The cell counts stay the same while the wall segments change length, so
  cell sizes and aspect ratios at the wall change. On the NACA0012 wing, 10 cells went above
  checkMesh's aspect ratio limit of 1,000 (1,159). After applying, adapt the chordwise cells in step 4
  (chordwise helper).
- **The exterior vertices are the same in every slice** (LIMITATIONS.md item 7). For swept, tapered or
  twisted wings the result is a compromise over the slices.
- **The outer boundary keeps its C-shape:** 29 and 33 stay at x = 0.
- **The weights are tuned on four wings** (MegAWES, NACA0012, FFA_W3_211, AP2) and are constants at the
  top of `core/blockopt.py`.

## Parameters (`core/blockopt.py`)

| Name | Value | Meaning |
|---|---|---|
| `W_CORNER`, `W_SPREAD`, `W_EXPANSION`, `W_WAKE` | 0.5, 0.5, 0.25 (× λ), 1 | term weights (wall term: 1) |
| `W_INTERIOR` | 0.1 | weight of the blocks inside the contour (beyond the tip) |
| `FOCUS` | 0.25 / 1 / 4 chords | λ of the priority setting |
| `WALL_FREE`, `EXPANSION_FREE` | ln 3 | length / expansion ratios that are not penalised |
| `WAKE_SCALE` | 15° | wake line angle scale |
| `GAP`, `GAP_OUT`, `GAP_ARC` | 0.02, 0.05 R, 0.15 rad | minimum gaps |
| `NEAR_WAKE` | 0.2–2 root chords | near-wake length |
| `N_SAMPLES` | 12 | wall samples per segment |
| `STEP` | 0.03 (positions, x/R, rad), 0.05 / 0.02 (22: x / z per root chord), 0.03 (z23/R) | initial steps |
| `max_evals` | 3,000 per start | evaluation budget |
