'''
sizing.py: helpers that compute cell counts and gradings from target cell sizes (step 4, M4)

All helpers return a *proposal* (new values + predicted result); apply_proposal() writes it to the case.
  - wall:      target y+ (at the first cell centre) -> first cell height -> e00 cells/grading
               (e09/e10 gradings scaled with e00, so the wake blocks keep their relation to e00)
  - chordwise: cell size at the TE, at the LE, maximum size and growth -> e03-e07 cells and grad1
  - wake:      cell size at the TE and growth, maximum size -> e02 and e01
  - spanwise:  sizes at the root, tip and far field, growth -> cells and grading of every layer
  - budget:    total number of cells -> all counts scaled (optionally keeping the wall y+)
Geometric distributions as blockMesh: n cells, expansion ratio g = last/first, ratio r = g^(1/(n-1)).
Flow: flat-plate skin friction Cf = (2 log10 Re_c - 0.65)^-2.3 with the local chord, u_tau = U sqrt(Cf/2),
y+ = u_tau (h/2) / nu at the first cell centre.
'''
import copy
import math

import numpy as np

from . import estimate, preview
from .estimate import graded_sizes, layer_types
from .planform import section_params

# ---------------------------------------------------------------------------------------------- #
# Geometric series
# ---------------------------------------------------------------------------------------------- #


def first_fraction(n, g):
    '''First cell as a fraction of the edge length (n cells, expansion ratio g).'''
    n = int(n)
    if n == 1 or abs(g - 1) < 1e-14:
        return 1.0 / n
    a = math.log(g) / (n - 1)                   # r = exp(a); (r - 1)/(r^n - 1) without cancellation near g = 1
    return math.expm1(a) / math.expm1(n * a)


def grading_for_first_fraction(n, f):
    '''Expansion ratio g with first cell = f x edge length (n cells). f < 1/n: growing, f > 1/n: shrinking.'''
    n = int(n)
    if n == 1 or abs(f * n - 1.0) < 1e-13:
        return 1.0
    lo, hi = -60.0, 60.0                        # bisection on ln(g); first_fraction decreases with g
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if first_fraction(n, math.exp(mid)) > f:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def series_first_growth(L, first, r_max, last_max=None):
    '''Fewest cells from a first cell growing at most with r_max (optionally capped at last_max): (n, g).'''
    if first >= L:
        return 1, 1.0
    if last_max is not None and last_max <= first * (1 + 1e-9):
        return max(1, int(math.ceil(L / first))), 1.0
    # geometric part until last_max, then (if needed) constant cells of last_max are not possible in one
    # blockMesh edge: use one geometric series and let the ratio adapt
    for n in range(2, 100000):
        f = first / L
        if f * n >= 1.0 - 1e-12:
            return n, 1.0
        g = grading_for_first_fraction(n, f)
        r = g ** (1.0 / (n - 1))
        if r <= r_max * (1 + 1e-9) and (last_max is None or first * g <= last_max * (1 + 1e-9)):
            return n, g
    raise ValueError("no distribution found")


def series_first_last(L, first, last):
    '''Cells and grading of a geometric series with (approximately) given first and last cell.'''
    if L <= max(first, last):
        return 1, 1.0
    if first + last >= L:
        n = max(1, int(round(2 * L / (first + last))))
        return n, (last / first if n > 1 else 1.0)
    if abs(last - first) <= 1e-9 * L:
        return max(1, int(round(L / first))), 1.0
    r = (L - first) / (L - last)
    n = max(2, int(round(1 + math.log(last / first) / math.log(r))))
    return n, last / first


# ---------------------------------------------------------------------------------------------- #
# Flow
# ---------------------------------------------------------------------------------------------- #


def u_tau(case, chord):
    re = case.flow.U * chord / case.flow.nu
    cf = (2 * math.log10(re) - 0.65) ** -2.3
    return case.flow.U * math.sqrt(cf / 2)


def first_height_for_yplus(case, yplus, chord):
    '''First cell height h with y+ at the cell centre (h/2) = yplus.'''
    return 2.0 * yplus * case.flow.nu / u_tau(case, chord)


def wing_slices(case):
    '''Slices that cut the wing (the airfoil is in the slice).'''
    types = layer_types(case)
    n = len(case.blocks.slice_locations)
    return [k for k in range(n) if (k < len(types) and types[k] != "flow") or (k > 0 and types[k - 1] != "flow")]


def wall_yplus(case, airfoil_file, model):
    '''
    y+ of the first cells on the wing (cell centre) in every wing slice, from the actual block mesh:
    {"slices": [(slice, y, chord, u_tau, heights, yplus)], "all": yplus array, "tip": (h, yplus) or None}
    '''
    out, allv = [], []
    st = case.wing_stations or [[0, 0, 1.0, 0, 0, 0]]
    for k in wing_slices(case):
        blocks = preview.slice_mesh(case, airfoil_file, k, model)
        surf = preview.surface_cells(blocks)
        if not surf:
            continue
        y = case.blocks.slice_locations[k]
        c = section_params(st, y)[0]
        ut = u_tau(case, c)
        h = np.r_[surf["upper"]["first_height"], surf["lower"]["first_height"]]
        yp = ut * 0.5 * h / case.flow.nu
        out.append((k, y, c, ut, h, yp))
        allv.append(yp)
    tip = None
    b2 = case.half_span
    if b2 is not None:
        from .blocks import tip_slice
        t = tip_slice(case)
        if t is not None and t < len(case.blocks.slice_locations) - 1:
            h = preview.spanwise_size(case, t)
            ut = u_tau(case, section_params(st, b2)[0])
            tip = (h, ut * 0.5 * h / case.flow.nu)
    return {"slices": out, "all": np.concatenate(allv) if allv else np.array([]), "tip": tip}


def yplus_stats(w):
    a = w["all"]
    if not len(a):
        return {}
    s = {"y+ min": float(a.min()), "y+ median": float(np.median(a)), "y+ max": float(a.max())}
    if w["tip"]:
        s["y+ on the tip face (spanwise first cell)"] = float(w["tip"][1])
    return s


# ---------------------------------------------------------------------------------------------- #
# Proposals
# ---------------------------------------------------------------------------------------------- #


def predict(case, airfoil_file, proposal=None):
    '''
    Result of a proposal (or of the current settings), measured on the actual block mesh of a copy:
    cells, y+ on the wing, largest chordwise size jump, largest spanwise jump, largest 3D aspect ratio.
    '''
    from .model import mshgen_mesh
    c = copy.deepcopy(case)
    if proposal:
        apply_proposal(c, proposal)
    model = mshgen_mesh(c, airfoil_file, upto="edges")
    out = {"cells": estimate.cell_count(c)}
    out.update(yplus_stats(wall_yplus(c, airfoil_file, model)))
    jumps, ar3d, first_ratio = [], [], []
    for k in range(len(c.blocks.slice_locations)):
        blocks = preview.slice_mesh(c, airfoil_file, k, model)
        met = preview.cell_metrics(blocks, preview.spanwise_size(c, k))
        if len(met["ar3d"]):
            ar3d.append(float(met["ar3d"].max()))
            first_ratio.append(float(met["ratio"].max()))
        surf = preview.surface_cells(blocks)
        if surf:
            jumps.append(max(surf["upper"]["max_jump"], surf["lower"]["max_jump"]))
    if jumps:
        out["largest chordwise jump on the airfoil"] = max(jumps)
    if first_ratio:
        out["largest neighbour size ratio (in blocks)"] = max(first_ratio)
    lj = preview.layer_jumps(c)
    if lj:
        out["largest spanwise jump at a slice"] = max(r["ratio"] for r in lj)
    if ar3d:
        out["largest aspect ratio (3D)"] = max(ar3d)
    return out


def apply_proposal(case, proposal):
    '''Write a proposal to the case: edges {name: [n, g1, g2]}, layers (res, grad) if present.'''
    if "edges" in proposal:
        e = dict(case.meshing.edges)
        e.update({k: [int(v[0]), float(v[1]), float(v[2])] for k, v in proposal["edges"].items()})
        case.meshing.edges = e
    if "layers" in proposal:
        case.meshing.layer_resolution = [int(n) for n in proposal["layers"][0]]
        case.meshing.layer_grading = [float(g) for g in proposal["layers"][1]]
    if "interior_points" in proposal or "exterior_points" in proposal:
        if "interior_points" in proposal:
            case.blocks.interior_points = [list(v) for v in proposal["interior_points"]]
        if "exterior_points" in proposal:
            case.blocks.exterior_points = [list(v) for v in proposal["exterior_points"]]
        case.blocks.vertex_source, case.blocks.defaults_key = "custom", None


def changes(case, proposal):
    '''Rows old -> new of a proposal (for a table).'''
    rows = []
    for k, (n, g1, g2) in proposal.get("edges", {}).items():
        on, og1, og2 = case.meshing.edges[k]
        rows.append({"item": k, "old": "{} cells, {:.4g} / {:.4g}".format(on, og1, og2),
                     "new": "{} cells, {:.4g} / {:.4g}".format(n, g1, g2)})
    if "layers" in proposal:
        for k, (n, g) in enumerate(zip(*proposal["layers"])):
            on, og = case.meshing.layer_resolution[k], case.meshing.layer_grading[k]
            if on != n or abs(og - g) > 1e-12 * abs(og):
                rows.append({"item": "layer {}".format(k), "old": "{} cells, {:.4g}".format(on, og),
                             "new": "{} cells, {:.4g}".format(n, g)})
    for i, (new, old) in enumerate(zip(proposal.get("interior_points", []), case.blocks.interior_points)):
        if abs(new[0] - old[0]) > 1e-12:
            rows.append({"item": "airfoil vertex {}".format(i), "old": "{:.4f}".format(old[0]),
                         "new": "{:.4f}".format(new[0])})
    for vid, (new, old) in enumerate(zip(proposal.get("exterior_points", []), case.blocks.exterior_points), start=22):
        if abs(new[0] - old[0]) + abs(new[1] - old[1]) > 1e-12:
            rows.append({"item": "exterior vertex {}".format(vid), "old": "({:.4g}, {:.4g})".format(*old),
                         "new": "({:.4g}, {:.4g})".format(*new)})
    return rows


def wall_proposal(case, airfoil_file, model, yplus, criterion="median", n0=None, growth_max=None):
    '''
    e00 (and e09/e10 scaled with it) for the target y+. criterion: "median" or "max" of the wing's first
    cells. n0: keep this number of wall-normal cells (default: current); growth_max: instead choose the
    fewest cells with a cell-to-cell growth ratio below growth_max.
    '''
    w = wall_yplus(case, airfoil_file, model)
    if not len(w["all"]):
        raise ValueError("no wing slices: the airfoil is not in any slice")
    cur = float(np.median(w["all"]) if criterion == "median" else w["all"].max())
    n_old, g_old = int(case.meshing.edges["e00"][0]), float(case.meshing.edges["e00"][1])
    f_target = first_fraction(n_old, g_old) * yplus / cur          # heights scale with the first fraction
    if growth_max:
        n = None
        for cand in range(4, 2000):
            g = grading_for_first_fraction(cand, f_target)
            if g ** (1.0 / (cand - 1)) <= growth_max:
                n = cand
                break
        if n is None:
            raise ValueError("no cell count reaches this growth ratio")
    else:
        n = int(n0 or n_old)
        g = grading_for_first_fraction(n, f_target)
    s = g / g_old
    e = case.meshing.edges
    prop = {"edges": {"e00": [n, g, g], "e09": [e["e09"][0], e["e09"][1] * s, e["e09"][2] * s],
                      "e10": [e["e10"][0], e["e10"][1] * s, e["e10"][2] * s]}}
    prop["note"] = ("first cells scale by {:.3g}: y+ {} {:.3g} -> {:.3g}; growth ratio {:.3f}; e09/e10 gradings "
                    "scaled by {:.3g}").format(yplus / cur, criterion, cur, yplus, g ** (1.0 / (n - 1)) if n > 1 else 1, s)
    return prop


def surface_segments(case, model, slice_index=0):
    '''Lengths of the five airfoil segments TE -> LE (e03..e07), mean of suction and pressure side.'''
    from .model import active_edges
    edges = active_edges(model, slice_index)
    base = slice_index * 39
    up = [(0, 1), (1, 4), (4, 7), (7, 10), (10, 13)]
    lo = [(0, 3), (3, 6), (6, 9), (9, 11), (11, 13)]

    def length(a, b):
        key = (base + a, base + b)
        pts = edges.get(key)
        if pts is None:
            pts = edges[(base + b, base + a)]
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return [(length(*u), length(*l)) for u, l in zip(up, lo)]


def chordwise_proposal(case, model, d_te, d_le, d_max, growth, slice_index=0):
    '''
    e03-e07 (cells, grad1) for cell sizes d_te at the TE and d_le at the LE, growing by at most `growth`
    per cell up to d_max. Sizes follow size(s) = min(d_max, d_te + (growth-1) s, d_le + (growth-1)(S-s))
    along the airfoil arc s (TE -> LE); grad1 = size at the LE-side end / size at the TE-side end.
    '''
    seg = [0.5 * (u + l) for u, l in surface_segments(case, model, slice_index)]
    S = sum(seg)
    k = growth - 1.0

    def size(s):
        return min(d_max, d_te + k * s, d_le + k * (S - s))
    prop, s0 = {"edges": {}}, 0.0
    for name, L in zip(["e03", "e04", "e05", "e06", "e07"], seg):
        n, g = series_first_last(L, size(s0), size(s0 + L))
        old = case.meshing.edges[name]
        prop["edges"][name] = [n, g, old[2]]
        s0 += L
    prop["note"] = "airfoil arc {:.4g} m (mean of both sides); grad2 (outer side) kept".format(S)
    return prop


def wake_proposal(case, model, d_te, growth, d_max):
    '''e02 (TE -> TE + 0.5c) and e01 (-> outlet): cells growing from d_te at most by `growth`, up to d_max.'''
    v = np.array([vx.coord for vx in model.vertices[:39]], dtype=float)
    L2 = float(np.linalg.norm(v[22] - v[0]))
    L1 = float(np.linalg.norm(v[23] - v[22]))
    n2, g2 = series_first_growth(L2, d_te, growth, d_max)
    last2 = d_te * g2
    n1, g1 = series_first_growth(L1, last2, growth, d_max)
    e = case.meshing.edges
    return {"edges": {"e02": [n2, g2, e["e02"][2]], "e01": [n1, g1, e["e01"][2]]},
            "note": "near wake {:.4g} m, far wake {:.4g} m; grad2 (outer side) kept".format(L2, L1)}


def spanwise_proposal(case, d_root, d_tip, d_far, growth):
    '''
    Cells and grading of every layer: target size d_root on the wing, d_tip at the tip slice (spanwise first
    cell of the tip face), at most d_far beyond the tip; sizes change by at most `growth` per cell, so the
    targets are reduced where a layer is too short to reach them. Sizes are continuous at the slices.
    '''
    if case.dim != "3D":
        raise ValueError("spanwise sizes only for 3D wings")
    from .blocks import tip_slice
    s = list(case.blocks.slice_locations)
    m = len(s)
    t = tip_slice(case)
    want = [d_root] * m
    if t is not None:
        want[t] = d_tip
        for k in range(t + 1, m):
            want[k] = d_far
    k_ = growth - 1.0
    # sizes reachable with the growth limit: size can change by (growth-1) x distance (linear in y)
    size = list(want)
    for _ in range(3):
        for i in range(1, m):
            size[i] = min(size[i], size[i - 1] + k_ * (s[i] - s[i - 1]))
        for i in range(m - 2, -1, -1):
            size[i] = min(size[i], size[i + 1] + k_ * (s[i + 1] - s[i]))
    res, grad = [], []
    for i in range(m - 1):
        n, g = series_first_last(s[i + 1] - s[i], size[i], size[i + 1])
        res.append(n)
        grad.append(g if n > 1 else 1.0)
    return {"layers": (res, grad), "note": "slice sizes: " + ", ".join("{:.3g}".format(x) for x in size)}


def budget_proposal(case, airfoil_file, model, target, keep_yplus=True):
    '''Scale all cell counts (chordwise, wall-normal, wake, spanwise) towards `target` total cells.'''
    n_now = estimate.cell_count(case)
    if target < 1000:
        raise ValueError("target must be at least 1000 cells")
    w = wall_yplus(case, airfoil_file, model) if keep_yplus else None
    yp = float(np.median(w["all"])) if w is not None and len(w["all"]) else None

    def scaled(f):
        c = copy.deepcopy(case)
        e = dict(c.meshing.edges)
        for k in ("e00", "e01", "e02", "e03", "e04", "e05", "e06", "e07", "e08"):
            n, g1, g2 = e[k]
            e[k] = [max(1, int(round(n * f))), g1, g2]
        e["e09"] = [e["e00"][0], e["e09"][1], e["e09"][2]]
        e["e10"] = [e["e00"][0], e["e10"][1], e["e10"][2]]
        c.meshing.edges = e
        if c.dim == "3D":
            c.meshing.layer_resolution = [max(1, int(round(n * f))) for n in c.meshing.layer_resolution]
        return c
    lo, hi = 0.05, 20.0
    for _ in range(40):
        f = math.sqrt(lo * hi)
        if estimate.cell_count(scaled(f)) > target:
            hi = f
        else:
            lo = f
    f = lo if abs(estimate.cell_count(scaled(lo)) - target) <= abs(estimate.cell_count(scaled(hi)) - target) else hi
    c = scaled(f)
    prop = {"edges": {k: c.meshing.edges[k] for k in c.meshing.edges}}
    if c.dim == "3D":
        prop["layers"] = (c.meshing.layer_resolution, c.meshing.layer_grading)
    note = "counts x {:.3f}: {:,} -> {:,} cells".format(f, n_now, estimate.cell_count(c))
    if yp is not None:
        # keep the first wall cell: same first fraction as before for the new n0
        n_old, g_old = int(case.meshing.edges["e00"][0]), float(case.meshing.edges["e00"][1])
        n_new = int(prop["edges"]["e00"][0])
        g = grading_for_first_fraction(n_new, first_fraction(n_old, g_old))
        s = g / g_old
        e = prop["edges"]
        e["e00"] = [n_new, g, g]
        e["e09"] = [n_new, case.meshing.edges["e09"][1] * s, case.meshing.edges["e09"][2] * s]
        e["e10"] = [n_new, case.meshing.edges["e10"][1] * s, case.meshing.edges["e10"][2] * s]
        note += "; wall first cell kept (y+ about {:.3g})".format(yp)
        # keep the spanwise cell at the tip slice (y+ of the tip face): regrade the two layers next to it
        from .blocks import tip_slice
        t = tip_slice(case)
        if "layers" in prop and t is not None and 0 < t < len(case.blocks.slice_locations) - 1:
            s_ = case.blocks.slice_locations
            res, grad = list(prop["layers"][0]), list(prop["layers"][1])
            L0, L1 = s_[t] - s_[t - 1], s_[t + 1] - s_[t]
            last_old = graded_sizes(L0, case.meshing.layer_resolution[t - 1], case.meshing.layer_grading[t - 1])[1]
            first_old = graded_sizes(L1, case.meshing.layer_resolution[t], case.meshing.layer_grading[t])[0]
            grad[t - 1] = 1.0 / grading_for_first_fraction(res[t - 1], last_old / L0)
            grad[t] = grading_for_first_fraction(res[t], first_old / L1)
            prop["layers"] = (res, grad)
            note += ", tip spanwise cell kept"
    prop["note"] = note
    return prop
