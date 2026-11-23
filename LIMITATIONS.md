# mshGen: known limitations and suggested improvements

Collected while building the GUI (`GUI/`, 17.11.2026). mshGen itself has not been changed for these
points. Each item says where it is in the code and how it shows up. "Verified" means seen in the
code (and, where stated, in a run); "suspected" means read in the code but not reproduced.

## Geometry

1. **One airfoil along the whole span.** Every slice uses the same airfoil file
   (`AirfoilSection(self.airfoil_name, self.airfoil_file, ...)` in `mesh_class.py`); only chord,
   twist, `dx` and `dz` vary along the span.
   *Suggestion:* allow an airfoil per wing station and interpolate the coordinates between stations.
2. **Sharp (closed) trailing edge required.** The block layout has a single TE vertex (vertex 0,
   shared by the upper and lower blocks). A blunt TE is not supported.
3. **Behaviour depends on the airfoil name.** `airfoil_name` switches on hard-coded code in several places:
   - `aero_functions.read_airfoil_file`: the first half of the points is reversed for
     `NACA0012` and `Mrev-v2_aileron`;
   - *(fixed 2026-11-20)* `aero_functions.RawData` used to remove a "corrupted" point by name:
     **index 38** for `Mrev-v2` (a valid lower-surface point at x/c = 0.152) and the middle point
     for NACA0012/AP2/Mrev-v2_aileron (the duplicate leading edge after the reversal above). It now
     removes only points identical to the previous point, for every airfoil: NACA0012 and
     Mrev-v2_aileron are unchanged, Mrev-v2 keeps all its points;
   - `aero_functions.RawData`: camber-line control points exist only for Mrev-v2,
     FFA_W3_211 and AP2. Other airfoils get a **straight line from LE to TE**, which can lie
     outside the airfoil for strongly cambered thin sections;
   - `mesh_class.generate_default`: the default interior vertex positions and the manual
     corrections of the exterior vertices are set per airfoil name;
   - `cmesh_default_parameters.get_default_edge_parameters`: default resolutions exist for
     NACA0012, Mrev-v2 and FFA_W3_211. Other names print "Unknown airfoil" and get the generic
     values tuned for R = 10c, whatever `mesh_scale` is.

   *Suggestion:* compute the camber line from the coordinates (mean of upper and lower surface), clean
   the airfoil data in a separate step (duplicate points, ordering), and move the per-airfoil
   settings into data files instead of `if` branches.
4. **Aileron cut-out geometry depends on the names.** `Mesh.aileron_cut_parameters` selects the cut
   position by `wing_name` (AP2, AP2_H_tail) or `airfoil_name` (AP2_V_tail; there even by exact
   slice positions `{0.06, 0.07, 0.4, 0.48}`). *Suggestion:* make the cut position, angle and
   offset inputs.
5. **Straight spanwise edges.** Only edges inside a slice are curved (polyLines along the
   airfoil splines, 51 points each). Between slices blockMesh connects the vertices with straight
   lines, so the shape between two slices is interpolated linearly. Planform kinks need a slice;
   non-linear twist or curved leading edges need enough slices.

## Domain and blocks

6. **Fixed domain proportions.** The C-grid radius is `mesh_scale` × **root** chord, the domain
   reaches 2R downstream and ±R vertically, and the near-wake zone is 0.5 × root chord
   (`generate_default`, `generate_edges`). *Suggestion:* make the wake length and the near-wake
   length inputs.
7. **Exterior vertices are absolute coordinates, the same in every slice.** `default_exterior_points_*.csv`
   holds x, z in metres (the scaling with `c0` is commented out in `generate_vertices`), and they are
   the same in every slice apart from the `dz` shift. A new chord, COG or `mesh_scale` needs new
   exterior points. Because they do not follow the sections along the span, **sweep, twist and taper are
   limited**: the trailing edge of every slice must stay upstream of the near-wake vertex 22 at
   TE_root + 0.5 × root chord. Otherwise the near-wake blocks 14/25 fold and blockMesh stops with
   "inward-pointing faces" (verified with the GUI: a NACA0012 wing, root chord 1.5 m, half-span 8 m,
   25° LE sweep fails; 10° works). Slices beyond the tip use the tip section, so the tip trailing edge
   counts too. The GUI checks this before blockMesh (`GUI/core/blockcheck.py`).
   *Suggestion:* store the exterior vertices relative to the chord and R, and shift them with the
   section (`dx`, like `dz`) or along the swept trailing edge.
8. **Fixed block layout.** 27 blocks per layer (13 inside the airfoil, 14 around it); only vertex
   positions, cell counts and gradings can change.
9. **Non-conformal interface beyond the tip.** In flow layers the airfoil blocks are filled and
   joined by `mergePatchPairs (leftInterface rightInterface)`. Blocks 1–2 have 2·n3 cells on that
   face and blocks 3–6 have 2·(n7 + n8), so the interface is only conformal when
   **n3 = n7 + n8**. Otherwise blockMesh creates polyhedral cells (verified: 632 polyhedra in
   `cases/MegAWES_wing`). *Suggestion:* an option that derives n3 from n7 + n8, or a different
   split of the blocks.
10. **Tip detection by tolerance.** A slice counts as the tip when it is within 0.1 % of the
    half-span (`initialize_mesh`). A tip slice just outside that tolerance gives the wrong number of
    wing and flow layers.

## Settings and files

11. **`DEFAULT=True` overwrites shared files.** The settings CSVs are written to
    `meshGeneration/default_files/` under a name built from airfoil, dimension, wing name,
    `mesh_scale` and `mesh_level`. Two users or examples with the same names overwrite each other's
    tuned settings. (The GUI keeps every case's CSVs in `cases/<case>/settings/`.)
12. **`grad1` of e09/e10 is ignored (verified in code).** In `generate_blocks`,
    `h0, ..., h8, g9, g10 = grd2_list` overwrites `g9, g10` read from `grad1`, so only `grad2` of
    e09 and e10 is used. All current settings files use equal values, so nothing changes today.
    The resolution column of e09/e10 is not used at all.
13. **Missing input files are not reported as errors.** Missing CSVs print "Input file is
    missing..." and the code continues until it fails with a `NameError`. Unknown airfoil or wing
    files print an error and continue with `b = 1, c = 1`. *Suggestion:* raise exceptions with
    the file name.
14. **blockMeshDict is written by hand.** The `#include` lists in `system/blockMeshDict` must match
    the number of slices and layers and the patch names (`wing` vs `wingUp/wingDown`, `wingCut`,
    `extra`, interfaces). (The GUI's `generate_mesh.py` writes it from the lists mshGen actually produced.)
15. **Package name.** The examples import `mshGen.meshGeneration...`, so the repository folder must
    be named `mshGen`.

## Possible bugs (suspected, not reproduced)

16. **`symTip` of a wing section without flow layers.** With `N_flow_layers == 0` (3D),
    `generate_boundaries` writes `symTip` from the back faces of **layer 0**. With more than one
    layer these are internal faces; it should be the last layer. The same applies to the aileron
    blocks added to `symTip` in that branch.
17. **2D `frontAndBack` face list.** `list_of_face_index += [5] * len(list_of_blocks)` is evaluated
    after `list_of_blocks` was doubled, so the index list is longer than the block list. It works
    because `zip` stops at the shorter list.

## GUI (`GUI/`)

18. **Only mshGen's existing airfoils (decision 17.11.2026).** The GUI offers the airfoil name/file
    pairs used by the example scripts (`GUI/core/airfoils.py`): Mrev-v2, NACA0012, NACA2420,
    FFA_W3_211 and AP2 for new wings and in step 1 (AP2 only where `AP2_airfoil.txt` is present: the
    AP2 files are not in the repository); Mrev-v2_aileron only through its example cases. New
    airfoil files cannot be added through the GUI yet, because mshGen selects point clean-up, camber
    line, default vertices and default resolution by the airfoil *name* (item 3). Adding an airfoil
    needs, in mshGen: a file in `meshGeneration/airfoils/` (TE → upper → LE → lower → TE, closed TE, no
    duplicate points), camber control points in `RawData` (or a computed camber line), default
    interior/exterior vertex positions in `generate_default`, and default resolutions in
    `cmesh_default_parameters.py`. Then add it to `LIBRARY` in `GUI/core/airfoils.py`. With item 3
    solved, the GUI can get an import step (planned in `GUI/PLAN.md`, step 1).
19. **Mrev-v2_aileron has no working default vertices.** For this name `generate_default` has no
    corrections of the exterior vertices, and the generic ones cross (blocks 14 and 20–24 fold at
    the root; verified with the GUI block check and blockMesh). The aileron examples work because
    they use hand-made settings files. The GUI therefore does not offer it for new wings.
20. **Default resolutions exist only for some scale/level pairs.** For NACA0012 at R = 1c only
    COARSE exists; asking mshGen for another level there fails inside
    `get_default_edge_parameters` (undefined variables). The GUI checks the combination first
    (`core.geometry.check_level`).

## Code structure

21. **Plots by figure number.** Plotting uses `plt.figure(slice + 1)` and module-level pyplot state,
    which clashes with other figures in the same Python session (e.g. a GUI). *Suggestion:* create
    and return figure objects.
22. **`importlib.reload` at import.** `mesh_class.py` reloads `aero_functions` and `mesh_functions`
    on every import; this is only useful while developing.

## Mesh quality (found with the GUI's quality metrics, M6)

23. **Size jump around the blocks inside the airfoil contour beyond the tip.** In the flow layers mshGen
    fills the airfoil-shaped region behind the tip with blocks 0–12. Their cells across the thickness
    (e03/e08 counts, about 5 cm for MegAWES) sit next to the wall-normal cells of the surrounding blocks
    (e00, 0.5 mm at the wall): a size ratio of about 100 across the block boundary (verified on MegAWES).
    One geometric grading cannot refine towards both sides of these blocks. *Suggestion:* split the
    interior blocks (or use double grading, `multiGrading` in blockMesh) so they match e00 at the
    contour.
24. **Worst non-orthogonality at the trailing-edge wedge beyond the tip.** Block 0 (vertices 0-1-2-3) is
    a wedge whose corner angle at vertex 0 is the trailing-edge angle of the airfoil. Its cells set the
    maximum non-orthogonality of the whole mesh for thin trailing edges (FFA_W3_211 83°, MegAWES 66°,
    verified: the GUI's metric equals checkMesh). Moving vertices or changing gradings does not remove
    it. *Suggestion:* a different block layout at the trailing edge in the flow layers (e.g. an O-type
    cap or a blunt/cut trailing edge).
