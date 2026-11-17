'''
quality.py: mesh quality from the slice previews, and the quality optimiser (M6)

Metrics (per slice, from the exact block mesh of core/preview.py):
  - non-orthogonality in the slice: angle between the line joining two cell centres and the normal of
    their common face, for all faces inside blocks and across block boundaries (checkMesh's definition,
    restricted to the slice plane)
  - size jumps: largest neighbour size ratio inside blocks and across block boundaries
  - smallest cell corner angle
Optimiser: see optimise().
'''
import math

import numpy as np

from . import preview


def _tri(p, q, r):
    a = 0.5 * ((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (r[..., 0] - p[..., 0]) * (q[..., 1] - p[..., 1]))
    return a, (p + q + r) / 3.0


def _centres(P):
    '''Area centroids of the quad cells (the cell centres of the extruded hexahedra).'''
    c00, c10, c11, c01 = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
    a1, g1 = _tri(c00, c10, c11)
    a2, g2 = _tri(c00, c11, c01)
    a = a1 + a2
    a = np.where(np.abs(a) < 1e-300, 1e-300, a)
    return (a1[..., None] * g1 + a2[..., None] * g2) / a[..., None]


def _face_metrics(c1, c2, a, b):
    '''
    Non-orthogonality (deg) and skewness of the faces with end points a, b between cells with centres c1, c2
    (2D arrays (..., 2)). Skewness as checkMesh (OpenFOAM primitiveMeshCheck::faceSkewness): distance from
    the face centre to where the line c1-c2 crosses the face, over max(0.2 |d|, face extent along it).
    '''
    d = c2 - c1
    e = b - a
    n = np.stack((e[..., 1], -e[..., 0]), axis=-1)
    dn = np.sum(d * n, axis=-1)
    cos = np.abs(dn) / np.maximum(np.linalg.norm(d, axis=-1) * np.linalg.norm(n, axis=-1), 1e-300)
    nonorth = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    fc = 0.5 * (a + b)
    cpf = fc - c1
    sv = cpf - (np.sum(n * cpf, axis=-1) / np.where(np.abs(dn) < 1e-300, 1e-300, dn))[..., None] * d
    msv = np.linalg.norm(sv, axis=-1)
    svh = sv / (msv[..., None] + 1e-300)
    fd = np.maximum(0.2 * np.linalg.norm(d, axis=-1), np.abs(np.sum(svh * (a - fc), axis=-1)))
    return nonorth, msv / np.maximum(fd, 1e-300)


def _angle(d, e):
    '''Angle (deg) between d and the normal of the face along e (2D vectors, arrays (..., 2)).'''
    n = np.stack((e[..., 1], -e[..., 0]), axis=-1)
    c = np.abs(np.sum(d * n, axis=-1)) / np.maximum(np.linalg.norm(d, axis=-1) * np.linalg.norm(n, axis=-1), 1e-300)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _sides(P):
    '''The 4 sides of a block face: (corner node ids order, node points along the side, adjacent cell centres,
    adjacent cell sizes along the side direction normal).'''
    C = _centres(P)
    # side: nodes, cell centres next to it (same order), cell size normal to the side
    dz = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)        # (nx+1, nz)
    dx = np.linalg.norm(P[1:, :] - P[:-1, :], axis=-1)        # (nx, nz+1)
    return {"bottom": (P[:, 0], C[:, 0], 0.5 * (dz[:-1, 0] + dz[1:, 0])),
            "top": (P[:, -1], C[:, -1], 0.5 * (dz[:-1, -1] + dz[1:, -1])),
            "left": (P[0, :], C[0, :], 0.5 * (dx[0, :-1] + dx[0, 1:])),
            "right": (P[-1, :], C[-1, :], 0.5 * (dx[-1, :-1] + dx[-1, 1:]))}


def slice_quality(blocks, dy=None):
    '''
    Non-orthogonality (max, and the block where it is), skewness, size ratios (max inside / across blocks),
    smallest cell angle and largest 3D aspect ratio (dy: spanwise cell size at the slice) of one slice.
    '''
    nonorth, skew, ratio_in, ratio_x, angle, ar3d = 0.0, 0.0, 1.0, 1.0, 180.0, 1.0
    where = ""
    sides = []
    for b in blocks:
        P = b["P"][..., [0, 2]]
        C = _centres(P)
        no = 0.0
        if C.shape[0] > 1:                                  # faces between cells i and i+1 (along z)
            o, s = _face_metrics(C[:-1], C[1:], P[1:-1, :-1], P[1:-1, 1:])
            no, skew = max(no, float(o.max())), max(skew, float(s.max()))
        if C.shape[1] > 1:                                  # faces between cells k and k+1 (along x)
            o, s = _face_metrics(C[:, :-1], C[:, 1:], P[:-1, 1:-1], P[1:, 1:-1])
            no, skew = max(no, float(o.max())), max(skew, float(s.max()))
        if no > nonorth:
            nonorth, where = no, "block {}".format(b["block"])
        m = preview.cell_metrics([b], dy)
        ratio_in = max(ratio_in, float(m["ratio"].max()))
        angle = min(angle, float(m["angle"].min()))
        ar3d = max(ar3d, float(m["ar3d"].max()))
        for name, (nodes, cen, size) in _sides(P).items():
            sides.append((nodes, cen, size, b["block"]))
    # faces across block boundaries: sides with the same end points (conformal: same nodes)
    index = {}
    for i, (nodes, cen, size, blk) in enumerate(sides):
        key = (tuple(np.round(nodes[0], 9)), tuple(np.round(nodes[-1], 9)))
        index.setdefault(key, []).append((i, False))
        index.setdefault(key[::-1], []).append((i, True))
    seen = set()
    for key, lst in index.items():
        if len(lst) < 2:
            continue
        (i, ri), (j, rj) = lst[0], lst[1]
        if (min(i, j), max(i, j)) in seen or i == j:
            continue
        seen.add((min(i, j), max(i, j)))
        ni, ci, si, bi = sides[i]
        nj, cj, sj, bj = sides[j]
        if ri:
            ni, ci, si = ni[::-1], ci[::-1], si[::-1]
        if rj:
            nj, cj, sj = nj[::-1], cj[::-1], sj[::-1]
        if len(ci) != len(cj) or not np.allclose(ni, nj, atol=1e-9 * (1 + np.abs(ni).max())):
            continue                                        # non-conformal (merged interface): skip
        o, s = _face_metrics(ci, cj, ni[:-1], ni[1:])
        no = float(o.max())
        skew = max(skew, float(s.max()))
        if no > nonorth:
            nonorth, where = no, "between blocks {} and {}".format(bi, bj)
        r = sj / np.maximum(si, 1e-300)
        ratio_x = max(ratio_x, float(np.maximum(r, 1 / r).max()))
    return {"non-orthogonality (deg)": nonorth, "skewness": skew, "size ratio inside blocks": ratio_in,
            "size ratio across blocks": ratio_x, "smallest angle (deg)": angle, "aspect ratio (3D)": ar3d,
            "where": where}


ACROSS_FLOW = "size ratio across blocks, beyond the tip"


def mesh_quality(case, airfoil_file, model=None, slices=None):
    '''
    Worst values over the given slices (default: all) and the largest spanwise jump at a slice.
    The size ratio across blocks is reported separately for the slices beyond the tip: there mshGen's
    blocks inside the airfoil contour are not graded towards the wall-normal cells around them (jumps of
    ~100 are normal; LIMITATIONS.md item 23); the optimiser does not use that value.
    '''
    from .estimate import layer_types
    from .model import mshgen_mesh
    model = model or mshgen_mesh(case, airfoil_file, upto="edges")
    types = layer_types(case)
    n = len(case.blocks.slice_locations)
    slices = range(n) if slices is None else slices
    out = {"non-orthogonality (deg)": 0.0, "skewness": 0.0, "aspect ratio (3D)": 1.0, "smallest angle (deg)": 180.0,
           "size ratio inside blocks": 1.0, "size ratio across blocks": 1.0, "chordwise jump on the airfoil": 1.0}
    per_slice = []
    for k in slices:
        blocks = preview.slice_mesh(case, airfoil_file, k, model)
        q = slice_quality(blocks, preview.spanwise_size(case, k))
        out["skewness"] = max(out["skewness"], q["skewness"])
        out["aspect ratio (3D)"] = max(out["aspect ratio (3D)"], q["aspect ratio (3D)"])
        per_slice.append(q["non-orthogonality (deg)"])
        if q["non-orthogonality (deg)"] > out["non-orthogonality (deg)"]:
            out["worst non-orthogonality at"] = "slice {}, {}".format(k, q["where"])
        out["non-orthogonality (deg)"] = max(out["non-orthogonality (deg)"], q["non-orthogonality (deg)"])
        out["smallest angle (deg)"] = min(out["smallest angle (deg)"], q["smallest angle (deg)"])
        out["size ratio inside blocks"] = max(out["size ratio inside blocks"], q["size ratio inside blocks"])
        flow = types[min(k, len(types) - 1)] == "flow"
        key = ACROSS_FLOW if flow else "size ratio across blocks"
        out[key] = max(out.get(key, 1.0), q["size ratio across blocks"])
        surf = preview.surface_cells(blocks)
        if surf:
            out["chordwise jump on the airfoil"] = max(out["chordwise jump on the airfoil"],
                                                       surf["upper"]["max_jump"], surf["lower"]["max_jump"])
    lj = preview.layer_jumps(case)
    out["spanwise jump at a slice"] = max([r["ratio"] for r in lj], default=1.0)
    out["mean of the slice maxima of non-orthogonality (deg)"] = float(np.mean(per_slice)) if per_slice else 0.0
    return out


# Metrics that must not get worse in an optimisation (with the tolerance), and their scale in the objective
GUARDED = {"non-orthogonality (deg)": (0.05, 45.0), "skewness": (0.005, 1.0), "size ratio inside blocks": (0.002, 0.5),
           "size ratio across blocks": (0.002, 0.5), "chordwise jump on the airfoil": (0.002, 0.5),
           "spanwise jump at a slice": (0.002, 0.5), "aspect ratio (3D)": ("1%", 0.0)}


# ---------------------------------------------------------------------------------------------- #
# Optimiser
# ---------------------------------------------------------------------------------------------- #
# Cell counts are never changed (fixed cell count, wall/tip first cells unchanged: e00 is not a variable,
# the gradings of the layers next to the tip are kept). Variables, by group:
GROUPS = {
    "chordwise gradings, airfoil side (e03-e07 grad1)": "grad1",
    "chordwise gradings, outer side (e03-e07 grad2)": "grad2",
    "wake gradings (e01, e02)": "wake",
    "spanwise gradings (layers not next to the tip)": "span",
    "chordwise divisions (airfoil vertices)": "divisions",
    "exterior vertices (outer boundary)": "exterior",
}
CHORD = ["e03", "e04", "e05", "e06", "e07"]
EXT_TOP, EXT_BOTTOM, EXT_ARC = [25, 26, 27, 28], [34, 35, 36, 37], [30, 32]


class _Problem:
    '''Variables of the selected groups <-> case values.'''

    def __init__(self, case, groups):
        from .blocks import radius, tip_slice
        self.case, self.groups = case, set(groups)
        self.R = radius(case)
        self.names, x0, lo, hi = [], [], [], []
        e = case.meshing.edges

        def add(name, value, a, b):
            self.names.append(name)
            x0.append(value)
            lo.append(a)
            hi.append(b)
        L = math.log(100.0)
        if "grad1" in groups:
            for k in CHORD:
                add(("edge", k, 1), math.log(e[k][1]), -L, L)
        if "grad2" in groups:
            for k in CHORD:
                add(("edge", k, 2), math.log(e[k][2]), -L, L)
        if "wake" in groups:
            for k in ("e02", "e01"):
                for side in (1, 2):
                    add(("edge", k, side), math.log(e[k][side]), -L, L)
        if "span" in groups and case.dim == "3D":
            t = tip_slice(case)
            fixed = {t - 1, t} if t is not None else set()
            for k, g in enumerate(case.meshing.layer_grading):
                if k not in fixed and case.meshing.layer_resolution[k] > 1:
                    add(("layer", k), math.log(g), -math.log(1e4), math.log(1e4))
        if "divisions" in groups:
            from .blocks import DIVISIONS
            for i, (_, s, p, _) in enumerate(DIVISIONS):
                add(("div", i, "s"), case.blocks.interior_points[s][0], 0.02, 0.98)
                add(("div", i, "p"), case.blocks.interior_points[p][0], 0.02, 0.98)
        if "exterior" in groups:
            for vid in EXT_TOP + EXT_BOTTOM:
                add(("ext", vid), case.blocks.exterior_points[vid - 22][0] / self.R, 0.0, 2.0)
            for vid in EXT_ARC:
                x, z = case.blocks.exterior_points[vid - 22]
                a = math.atan2(z, x) % (2 * math.pi)
                add(("arc", vid), a, math.pi / 2 + 0.05, 3 * math.pi / 2 - 0.05)
        self.x0, self.bounds = np.array(x0), list(zip(lo, hi))

    def apply(self, x, case):
        '''Set the values of x in case (a copy); returns False if the values are not admissible.'''
        from .blocks import DIVISIONS, division_problems, move_exterior
        e = {k: list(v) for k, v in case.meshing.edges.items()}
        grad = list(case.meshing.layer_grading)
        pts = [list(v) for v in case.blocks.interior_points]
        vert = False
        for name, v in zip(self.names, x):
            if name[0] == "edge":
                e[name[1]][name[2]] = float(math.exp(v))
            elif name[0] == "layer":
                grad[name[1]] = float(math.exp(v))
            elif name[0] == "div":
                _, s, p, cam = DIVISIONS[name[1]]
                pts[s if name[2] == "s" else p][0] = float(v)
                vert = True
        case.meshing.edges, case.meshing.layer_grading = e, grad
        if "divisions" in self.groups:
            for i, (_, s, p, cam) in enumerate(DIVISIONS):           # camber vertex: mean of both sides - offset
                if cam is not None:
                    old = case.blocks.interior_points
                    off = old[cam][0] - 0.5 * (old[s][0] + old[p][0])
                    pts[cam][0] = 0.5 * (pts[s][0] + pts[p][0]) + off
            if division_problems(pts):
                return False
            case.blocks.interior_points = pts
        for name, v in zip(self.names, x):
            if name[0] == "ext":
                kind_z = self.R if name[1] in EXT_TOP else -self.R
                move_exterior(case, name[1], float(v) * self.R, kind_z)
                vert = True
            elif name[0] == "arc":
                move_exterior(case, name[1], self.R * math.cos(v), self.R * math.sin(v))
                vert = True
        if vert:                    # vertices out of order fold blocks: caught by the block check in optimise()
            case.blocks.vertex_source, case.blocks.defaults_key = "custom", None
        return True


def objective_value(q, weight_orth, q0=None):
    '''
    Smaller is better. Non-orthogonality: the worst value and the mean of the slice maxima (so improving a
    slice counts even where the overall worst value is fixed by the block layout), in deg / 45. Size
    ratios: every ratio counts (ratio - 1). With q0 (the start), a guarded metric that gets worse than at
    the start is penalised heavily, so the result is never worse in any of them.
    '''
    orth = 0.5 * (q["non-orthogonality (deg)"] + q["mean of the slice maxima of non-orthogonality (deg)"]) / 45.0
    ratios = sum(q[k] - 1.0 for k in ("size ratio inside blocks", "size ratio across blocks",
                                       "chordwise jump on the airfoil", "spanwise jump at a slice"))
    j = weight_orth * orth + (1 - weight_orth) * ratios
    if q0 is not None:
        for k, (tol, scale) in GUARDED.items():
            if tol == "1%":                                 # relative tolerance (aspect ratio: thousands)
                tol, scale = 0.01 * q0[k], q0[k]
            worse = q[k] - q0[k] - tol
            if worse > 0:
                j += 20.0 * (1.0 + worse / scale)
    return j


INITIAL_STEP = {"edge": math.log(1.5), "layer": math.log(1.5), "div": 0.03, "ext": 0.04, "arc": 0.05}


def _pattern_search(f, prob, j0, max_evals, min_factor=1.0 / 64):
    '''
    Coordinate pattern search from the current mesh: per variable try +step and -step, keep a step that
    improves the objective, halve all steps after a sweep without improvement. Only accepted improvements
    are kept, so the result is never worse than the start.
    '''
    x, j = prob.x0.copy(), j0
    lo, hi = np.array([b[0] for b in prob.bounds]), np.array([b[1] for b in prob.bounds])
    step = np.array([INITIAL_STEP[n[0]] for n in prob.names])
    step0 = step.copy()
    evals = 0
    while evals < max_evals:
        improved = False
        for i in range(len(x)):
            for sgn in (1.0, -1.0):
                if evals >= max_evals:
                    break
                xt = x.copy()
                xt[i] = min(max(x[i] + sgn * step[i], lo[i]), hi[i])
                if xt[i] == x[i]:
                    continue
                jt = f(xt)
                evals += 1
                if jt < j - 1e-6:
                    x, j, improved = xt, jt, True
                    step[i] *= 1.5                      # moving in a good direction: larger steps
                    break
        if not improved:
            step *= 0.5
            if np.all(step < step0 * min_factor):
                break
    return x, j


def optimise(case, airfoil_file, groups, weight_orth=0.5, max_evals=200, progress=None, slices=None):
    '''
    Improve the mesh quality at a fixed cell count by changing the variables of the selected groups
    (keys of GROUPS values). Returns a proposal (core/sizing format, plus vertex positions) with "before"
    and "after" quality. weight_orth in [0, 1]: 1 = only non-orthogonality, 0 = only size ratios.
    '''
    import copy
    from .blockcheck import layout, problems_of
    from .model import mshgen_mesh
    prob = _Problem(case, groups)
    if not prob.names:
        raise ValueError("no variables in the selected groups")
    n = len(case.blocks.slice_locations)
    if slices is None:
        from .blocks import tip_slice
        t = tip_slice(case)
        slices = sorted({0, n - 1} | ({t - 1, t} if t is not None else set()) | ({n // 2} if n > 2 else set()))
        slices = [s for s in slices if 0 <= s < n]
    q0 = mesh_quality(case, airfoil_file, slices=slices)
    j0 = objective_value(q0, weight_orth, q0)
    best = {"j": j0, "x": prob.x0.copy(), "q": q0}
    count = [0]

    def f(x):
        count[0] += 1
        c = copy.deepcopy(case)
        if not prob.apply(x, c):
            return j0 * 10 + 10
        try:
            model = mshgen_mesh(c, airfoil_file, upto="edges")
            if prob.groups & {"divisions", "exterior"}:      # vertices moved: no folded blocks allowed
                v = np.array([vx.coord for vx in model.vertices], float).reshape(n, 39, 3)
                if problems_of(c, layout(c, airfoil_file, v)):
                    return j0 * 10 + 10
            q = mesh_quality(c, airfoil_file, model, slices)
        except Exception:  # noqa: BLE001
            return j0 * 10 + 10
        j = objective_value(q, weight_orth, q0)
        if j < best["j"]:
            best.update(j=j, x=np.array(x), q=q)
        if progress:
            progress(count[0], max_evals, best["j"], j0)
        return j
    _pattern_search(f, prob, j0, max_evals)
    c = copy.deepcopy(case)
    prob.apply(best["x"], c)
    prop = {"edges": {k: c.meshing.edges[k] for k in case.meshing.edges if c.meshing.edges[k] != case.meshing.edges[k]}}
    if c.meshing.layer_grading != case.meshing.layer_grading:
        prop["layers"] = (c.meshing.layer_resolution, c.meshing.layer_grading)
    if c.blocks.interior_points != case.blocks.interior_points:
        prop["interior_points"] = c.blocks.interior_points
    if c.blocks.exterior_points != case.blocks.exterior_points:
        prop["exterior_points"] = c.blocks.exterior_points
    prop["before"], prop["after"] = q0, best["q"]
    prop["note"] = "{} evaluations; objective {:.3f} -> {:.3f} (slices {})".format(count[0], j0, best["j"], slices)
    return prop
