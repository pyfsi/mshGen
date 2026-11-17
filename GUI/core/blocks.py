'''
blocks.py: slices, chordwise divisions and exterior vertices of a case (step 3), domain size (step 2)

Slices
  - add_slice: split a layer at the cell face nearest to y, so the cell distribution does not change
  - remove_slice: merge two layers (cell count summed, grading from the first and last cell sizes)
  - set_slice: move a slice (cell counts and gradings kept)
  - suggest_slices: root, planform kinks, refinement around the tip, tip, far field
Chordwise divisions (airfoil vertices 0-15, positions along the splines, 0 = LE, 1 = TE)
  - DIVISIONS: the four divisions of the airfoil surface, with their camber vertex
  - set_division: move a division on the suction/pressure side, the camber vertex follows
Exterior vertices 22-38 (x, z in m)
  - EXTERIOR_KIND: where each vertex lives (outer boundary, outlet, wake line)
  - move_exterior: move a vertex and keep it on its boundary
Domain
  - set_mesh_scale, set_far_field
Any edit of the vertices makes them "custom" (no automatic recompute of mshGen's defaults).
'''
import math

import numpy as np

from .estimate import graded_sizes, layer_types

# ---------------------------------------------------------------------------------------------- #
# Slices
# ---------------------------------------------------------------------------------------------- #


def tip_slice(case):
    '''Index of the slice at the wing tip (None in 2D or without a slice at the tip).'''
    b2 = case.half_span
    if b2 is None:
        return None
    for k, y in enumerate(case.blocks.slice_locations):
        if abs(y - b2) <= 1e-3 * b2:
            return k
    return None


def cell_faces(y0, y1, n, grading):
    '''Positions of the n+1 cell faces of a graded layer (grading = last/first cell).'''
    first, _ = graded_sizes(y1 - y0, n, grading)
    r = grading ** (1.0 / (n - 1)) if n > 1 else 1.0
    sizes = first * r ** np.arange(n)
    faces = y0 + np.r_[0.0, np.cumsum(sizes)]
    faces[-1] = y1
    return faces


def add_slice(case, y):
    '''
    Add a slice in the layer containing y, at the nearest cell face (the cells do not change).
    Returns the y of the new slice.
    '''
    s, res, grad = case.blocks.slice_locations, case.meshing.layer_resolution, case.meshing.layer_grading
    if not s[0] < y < s[-1]:
        raise ValueError("y must be between the root and the far field ({:g} ... {:g})".format(s[0], s[-1]))
    k = int(np.searchsorted(s, y) - 1)
    n, g = int(res[k]), float(grad[k])
    if n < 2:
        raise ValueError("layer {} has only one cell: give it more cells first".format(k))
    faces = cell_faces(s[k], s[k + 1], n, g)
    i = int(np.clip(np.argmin(np.abs(faces - y)), 1, n - 1))
    r = g ** (1.0 / (n - 1))
    n1, n2 = i, n - i
    g1 = r ** (n1 - 1) if n1 > 1 else 1.0
    g2 = r ** (n2 - 1) if n2 > 1 else 1.0
    ynew = float(faces[i])
    case.blocks.slice_locations = s[:k + 1] + [ynew] + s[k + 1:]
    case.meshing.layer_resolution = res[:k] + [n1, n2] + res[k + 1:]
    case.meshing.layer_grading = grad[:k] + [g1, g2] + grad[k + 1:]
    _shift_aileron_layers(case, k, +1)
    return ynew


def remove_slice(case, index):
    '''Remove slice index (not the root, not the last); the two layers are merged.'''
    s, res, grad = case.blocks.slice_locations, case.meshing.layer_resolution, case.meshing.layer_grading
    if index <= 0 or index >= len(s) - 1:
        raise ValueError("the root and the last slice cannot be removed")
    if index == tip_slice(case) and any(t == "flow" for t in layer_types(case)):
        raise ValueError("the tip slice cannot be removed while there are slices beyond the tip")
    a, b = index - 1, index
    first, _ = graded_sizes(s[a + 1] - s[a], res[a], grad[a])
    _, last = graded_sizes(s[b + 1] - s[b], res[b], grad[b])
    n = int(res[a] + res[b])
    case.blocks.slice_locations = s[:index] + s[index + 1:]
    case.meshing.layer_resolution = res[:a] + [n] + res[b + 1:]
    case.meshing.layer_grading = grad[:a] + [last / first if n > 1 else 1.0] + grad[b + 1:]
    _shift_aileron_layers(case, a, -1)


def _shift_aileron_layers(case, layer, delta):
    '''Keep the aileron layers on the same part of the span after a layer was split (+1) or merged (-1).'''
    a = case.options.aileron_layers
    if not a:
        return
    out = set()
    for k in a:
        if k < layer:
            out.add(k)
        elif k == layer:
            out.update([k, k + 1] if delta > 0 else [k])
        else:
            out.add(k + delta)
    case.options.aileron_layers = sorted(out)


def set_slice(case, index, y):
    '''Move slice index to y (cell counts and gradings kept). Root fixed; the tip slice stays at the tip.'''
    s = list(case.blocks.slice_locations)
    if index == 0:
        raise ValueError("the root slice stays at y = 0")
    if index == tip_slice(case) and abs(y - s[index]) > 1e-12:
        raise ValueError("the tip slice stays at the tip (change the half-span in step 1)")
    lo, hi = s[index - 1], (s[index + 1] if index + 1 < len(s) else math.inf)
    if not lo < y < hi:
        raise ValueError("slice {} must stay between {:g} and {:g}".format(index, lo, hi))
    s[index] = float(y)
    case.blocks.slice_locations = s


def planform_kinks(case, slope_change=0.02, twist_change_deg=3.0, merge=0.1):
    '''
    y of the planform kinks between root and tip: rows where the slope of chord, dx or dz changes by more
    than slope_change, or the twist slope by more than twist_change_deg per half-span. Kinks closer than
    merge x half-span are combined (mean position), so digitised planforms do not get a slice per row.
    '''
    st = np.asarray(case.wing_stations or [], dtype=float)
    if len(st) < 3:
        return []
    y = st[:, 1]
    b2 = y[-1]
    cand = []
    for i in range(1, len(st) - 1):
        for col in (2, 3, 4, 5):
            s0 = (st[i, col] - st[i - 1, col]) / (y[i] - y[i - 1])
            s1 = (st[i + 1, col] - st[i, col]) / (y[i + 1] - y[i])
            limit = twist_change_deg / b2 if col == 3 else slope_change
            if abs(s1 - s0) > limit:
                cand.append(float(y[i]))
                break
    out, group = [], []
    for yk in cand:
        if group and yk - group[0] > merge * b2:
            out.append(sum(group) / len(group))
            group = []
        group.append(yk)
    if group:
        out.append(sum(group) / len(group))
    return out


def suggest_slices(case, tip_fraction=0.1, cells_span=40, tip_ratio=0.02):
    '''
    Slices: root, planform kinks (only if clearly inside the wing), tip -/+ tip_fraction x half-span, tip,
    far field (kept). Cells: about half-span / cells_span in the wing, refined towards the tip to
    tip_ratio x that size, coarsening in the far field.
    '''
    b2 = case.half_span
    if b2 is None:
        raise ValueError("slices are only suggested for 3D wings")
    far = case.blocks.slice_locations[-1]
    if far <= b2 * (1 + tip_fraction) * 1.05:
        far = 2.0 * b2
    d = tip_fraction * b2
    inner = [y for y in planform_kinks(case) if 0.05 * b2 < y < b2 - 1.5 * d]
    s = sorted(set([0.0] + inner + [b2 - d, b2, b2 + d, far]))
    size = b2 / cells_span
    tip_size = tip_ratio * size
    res, grad = [], []
    for k in range(len(s) - 1):
        L = s[k + 1] - s[k]
        if s[k + 1] <= b2 - d:                                     # wing: uniform
            n, g = max(2, int(round(L / size))), 1.0
        elif s[k + 1] == b2:                                       # towards the tip
            n, g = _graded_layer(L, size, tip_size)
        elif s[k] == b2:                                           # away from the tip
            n, g = _graded_layer(L, tip_size, size)
        else:                                                      # far field: from size to ~10 x size
            n, g = _graded_layer(L, size, min(10 * size, L / 3))
        res.append(n)
        grad.append(g)
    case.blocks.slice_locations, case.meshing.layer_resolution, case.meshing.layer_grading = s, res, grad
    case.options.aileron_layers = None
    return s


def _graded_layer(L, first, last):
    '''Cells and grading of a geometric distribution with given first and last cell size (approximately).'''
    if abs(last - first) < 1e-12 * L:
        return max(2, int(round(L / first))), 1.0
    # sum of a geometric series: L = (last*r - first)/(r - 1) -> r
    r = (L - first) / (L - last)
    if r <= 0 or abs(r - 1) < 1e-9:
        return max(2, int(round(2 * L / (first + last)))), last / first
    n = max(2, int(round(1 + math.log(last / first) / math.log(r))))
    return n, last / first


# ---------------------------------------------------------------------------------------------- #
# Chordwise divisions (airfoil vertices)
# ---------------------------------------------------------------------------------------------- #

# name, suction-side vertex, pressure-side vertex, camber vertex
DIVISIONS = [("TE segment end (e03 | e04)", 1, 3, 2),
             ("aft segment end (e04 | e05)", 4, 6, 5),
             ("mid segment end (e05 | e06)", 7, 9, 8),
             ("nose segment start (e06 | e07)", 10, 11, None)]


def divisions(case):
    '''Rows: name, suction-side position, pressure-side position (positions along the splines).'''
    p = case.blocks.interior_points
    return [{"division": name, "suction side": p[s][0], "pressure side": p[q][0]} for name, s, q, _ in DIVISIONS]


def set_division(case, index, suction=None, pressure=None):
    '''Move division index; its camber vertex keeps its offset to the mean of both sides.'''
    name, s, q, cam = DIVISIONS[index]
    p = [list(v) for v in case.blocks.interior_points]
    old_mean = 0.5 * (p[s][0] + p[q][0])
    if suction is not None:
        p[s][0] = float(suction)
    if pressure is not None:
        p[q][0] = float(pressure)
    if cam is not None:
        p[cam][0] = float(p[cam][0] + 0.5 * (p[s][0] + p[q][0]) - old_mean)
    problems = division_problems(p)
    if problems:
        raise ValueError("; ".join(problems))
    case.blocks.interior_points = p
    case.blocks.vertex_source, case.blocks.defaults_key = "custom", None


def division_problems(points):
    '''Order along the splines: TE (1) > 1 > 4 > 7 > 10 > LE (0), same on the pressure side; camber too.'''
    pos = [v[0] for v in points]
    p = []
    for side, idx in (("suction", [0, 1, 4, 7, 10, 13]), ("pressure", [0, 3, 6, 9, 11, 13])):
        vals = [1.0] + [pos[i] for i in idx[1:-1]] + [0.0]
        if any(b >= a for a, b in zip(vals, vals[1:])):
            p.append(side + " side: divisions must decrease from the TE (1) to the LE (0)")
    cam = [pos[i] for i in (0, 2, 5, 8, 12, 13)]
    if any(b >= a for a, b in zip(cam, cam[1:])):
        p.append("camber line: vertices 0 > 2 > 5 > 8 > 12 > 13 required")
    if any(not 0.0 <= x <= 1.0 for x in pos):
        p.append("positions must be between 0 and 1")
    return p


def set_interior_point(case, vertex, position):
    p = [list(v) for v in case.blocks.interior_points]
    p[vertex][0] = float(position)
    problems = division_problems(p)
    if problems:
        raise ValueError("; ".join(problems))
    case.blocks.interior_points = p
    case.blocks.vertex_source, case.blocks.defaults_key = "custom", None


# ---------------------------------------------------------------------------------------------- #
# Exterior vertices
# ---------------------------------------------------------------------------------------------- #

# 22: near-wake end (on the wake line), 23: outlet centre, 24/38: outlet corners, 25-29 top, 30-32 C-arc,
# 33-37 bottom. "top"/"bottom": z = +-R, "arc": radius R, "outlet": x = 2R, "corner": fixed, "free".
EXTERIOR_KIND = {22: "free", 23: "outlet", 24: "corner", 25: "top", 26: "top", 27: "top", 28: "top",
                 29: "top", 30: "arc", 31: "arc", 32: "arc", 33: "bottom", 34: "bottom", 35: "bottom",
                 36: "bottom", 37: "bottom", 38: "corner"}


def radius(case):
    return float(case.domain.mesh_scale) * case.root_section()[0]


def project_exterior(case, vid, x, z):
    '''Point (x, z) moved onto the boundary of exterior vertex vid (root plane, without dz).'''
    R = radius(case)
    kind = EXTERIOR_KIND[vid]
    if kind == "top":
        return float(min(x, 2 * R)), R
    if kind == "bottom":
        return float(min(x, 2 * R)), -R
    if kind == "outlet":
        return 2 * R, float(np.clip(z, -R, R))
    if kind == "arc":
        a = math.atan2(z, x)
        if a < 0:
            a += 2 * math.pi
        a = float(np.clip(a, math.pi / 2 + 1e-3, 3 * math.pi / 2 - 1e-3))
        return R * math.cos(a), R * math.sin(a)
    if kind == "corner":
        return 2 * R, (R if vid == 24 else -R)
    return float(x), float(z)


def move_exterior(case, vid, x, z):
    '''Move exterior vertex vid (22-38) to (x, z), kept on its boundary. Returns the new position.'''
    i = vid - 22
    x, z = project_exterior(case, vid, x, z)
    p = [list(v) for v in case.blocks.exterior_points]
    p[i] = [x, z]
    case.blocks.exterior_points = p
    case.blocks.vertex_source, case.blocks.defaults_key = "custom", None
    return x, z


def exterior_problems(case, tol=1e-2):
    '''
    Exterior vertices that are not on their boundary for the current R (e.g. custom vertices after changing R).
    tol relative to R: mshGen's own defaults put some vertices slightly off (e.g. 35/36 for Mrev-v2).
    '''
    R = radius(case)
    out = []
    for vid, (x, z) in zip(range(22, 39), case.blocks.exterior_points):
        kind = EXTERIOR_KIND[vid]
        if kind in ("top", "bottom", "arc", "outlet", "corner"):
            px, pz = project_exterior(case, vid, x, z)
            if math.hypot(px - x, pz - z) > tol * R:
                out.append(vid)
    return out


# ---------------------------------------------------------------------------------------------- #
# Domain
# ---------------------------------------------------------------------------------------------- #

def set_mesh_scale(case, mesh_scale):
    if mesh_scale <= 0.5:
        raise ValueError("R/c must be larger than 0.5")
    case.domain.mesh_scale = float(mesh_scale)


def set_far_field(case, y):
    '''Move the last slice (symTip) to y; it must stay beyond the previous slice.'''
    s = list(case.blocks.slice_locations)
    if len(s) < 2 or y <= s[-2]:
        raise ValueError("the far field must lie beyond the previous slice (y > {:g})".format(s[-2]))
    s[-1] = float(y)
    case.blocks.slice_locations = s


def first_cell_estimate(case):
    '''
    Rough first wall-normal cell height from e00 (n0 cells, expansion g0 over about R) and the y+ it gives
    at the cell centre for the root chord (flat-plate Cf). Detailed y+ tools follow in M4.
    '''
    n0, g0 = int(case.meshing.edges["e00"][0]), float(case.meshing.edges["e00"][1])
    c0 = case.root_section()[0]
    first, _ = graded_sizes(radius(case) - 0.5 * c0, n0, g0)
    re = case.flow.U * c0 / case.flow.nu
    cf = (2 * math.log10(re) - 0.65) ** -2.3
    u_tau = case.flow.U * math.sqrt(cf / 2)
    return {"Re (root chord)": re, "Cf (flat plate)": cf, "u_tau (m/s)": u_tau,
            "first cell (m), approx.": first, "y+ at the first cell centre, approx.": 0.5 * first * u_tau / case.flow.nu}
