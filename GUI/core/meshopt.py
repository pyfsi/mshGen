'''
meshopt.py: the mesh optimiser of step 4 - cells and gradings of all edges (e00-e10) and of the spanwise layers
from a few targets (docs/mesh_optimiser.md)

Targets (presets per mesh level, editable in the GUI): y+ of the first wall cell, chordwise cells per airfoil side
(min/max), spanwise cells on the wing (min/max), largest growth ratio between neighbouring cells, largest number of
cells. The block vertices (step 3) are not changed.

1. Sizing (rules, from the targets):
   - wall-normal (e00): first cell for the target y+, fewest cells with a growth ratio <= the maximum
   - chordwise (e03-e07: cells, grad1): cell size proportional to the local radius of curvature of the airfoil (at
     most one chord), the trailing edge as fine as the leading edge, growth <= the maximum, N cells per side; the
     segments are fitted together (counts and gradings, growth inside <= max, jumps at the segment ends penalised)
   - wake (e02, e01: cells, grad1): from the TE cell with growth <= the maximum, the near wake at most as coarse as
     the largest airfoil cell; e10 graded like e00 (thin cells at the wake line), e09 like the streamwise cells
   - spanwise: uniform on the wing, the y+ cell at the tip face, growth <= the maximum, N cells on the wing
   - counts: the largest counts in the ranges that the cell budget allows
2. Quality (Hooke-Jeeves pattern search on the exact block mesh of the root and tip slices, counts fixed): grad2 of
   e01-e07 and the gradings of e09/e10 (beyond the tip also e08 and grad1 of e03-e07), from the best of five starts;
   objective: non-orthogonality (mean weighted towards the airfoil, worst 1 %, faces above 65 deg) and size jumps above
   the max growth (mean, worst 1 %); beyond the tip also inside the airfoil contour (tip slice, weight W_INTERIOR),
   against the fit of the chordwise grad1 to the sizes of 1 (W_WALL).
3. y+: e00 is regraded so that the median y+ of the wing's first cells (exact mesh) equals the target.
Also: slice_table() - every slice of a mesh with warnings (detailed analysis of step 4).
'''
import copy
import math

import numpy as np
from scipy.interpolate import splev
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from . import estimate, preview, sizing
from .estimate import layer_types
from .planform import section_params
from .quality import _centres, _face_metrics, _sides
from .sizing import first_fraction, grading_for_first_fraction, series_first_growth

# ---------------------------------------------------------------------------------------------- #
# Targets
# ---------------------------------------------------------------------------------------------- #
LEVELS = ["COARSE", "MEDIUM", "FINE"]
PRESETS = {
    "COARSE": {"yplus": 100.0, "chord_min": 60, "chord_max": 100, "span_min": 30, "span_max": 60,
               "growth": 1.25, "max_cells": 1500000},
    "MEDIUM": {"yplus": 30.0, "chord_min": 120, "chord_max": 200, "span_min": 60, "span_max": 120,
               "growth": 1.2, "max_cells": 5000000},
    "FINE": {"yplus": 1.0, "chord_min": 200, "chord_max": 400, "span_min": 100, "span_max": 250,
             "growth": 1.15, "max_cells": 30000000},
}
def targets(case):
    '''The case's targets (meshing.targets), missing values from the preset of its mesh level.'''
    t = dict(PRESETS.get(case.domain.mesh_level, PRESETS["COARSE"]))
    t.update(case.meshing.targets or {})
    return t


def target_problems(t, dim="3D"):
    p = []
    if not t["yplus"] > 0:
        p.append("y+ must be positive")
    if not 1.0 < t["growth"] <= 2.0:
        p.append("the growth ratio must be in (1, 2]")
    if not 5 <= t["chord_min"] <= t["chord_max"]:
        p.append("chordwise cells: 5 <= min <= max")
    if dim == "3D" and not 1 <= t["span_min"] <= t["span_max"]:
        p.append("spanwise cells: 1 <= min <= max")
    if not t["max_cells"] >= 1000:
        p.append("max number of cells: at least 1000")
    return p


# Rules of the sizing (constants, documented)
RHO_CAP = 1.0           # chordwise size follows the radius of curvature up to this many chords
TE_RATIO = 1.0          # TE cell = TE_RATIO x LE cell
GROWTH_MARGIN = 0.85    # size fields grow by GROWTH_MARGIN x (growth - 1) per cell (room for fitting and rounding)
N_SPLINE = 20001        # spline samples per airfoil side
CHORD = ["e03", "e04", "e05", "e06", "e07"]       # TE -> LE
WAKE = ["e02", "e01"]

# ---------------------------------------------------------------------------------------------- #
# Edges -> blocks (Mesh.generate_blocks), so the exact mesh can be evaluated without calling mshGen again
# ---------------------------------------------------------------------------------------------- #


def layer_blocks(edges, ny, gy):
    '''(resolution, grading) of the 27 blocks of a layer from the edge parameters, as Mesh.generate_blocks.'''
    n0, n1, n2, n3, n4, n5, n6, n7, n8, _, _ = [int(edges["e{:02d}".format(i)][0]) for i in range(11)]
    g0, g1, g2, g3, g4, g5, g6, g7, g8, _, _ = [float(edges["e{:02d}".format(i)][1]) for i in range(11)]
    h0, h1, h2, h3, h4, h5, h6, h7, h8, g9, g10 = [float(edges["e{:02d}".format(i)][2]) for i in range(11)]
    res = [[n3, ny, n3]] + [[n4, ny, n3]] * 2
    res += [[n5, ny, n8], [n5, ny, n7], [n5, ny, n7], [n5, ny, n8]]
    res += [[n6, ny, n8], [n6, ny, n7], [n6, ny, n7], [n6, ny, n8]]
    res += [[n7, ny, n8]] * 2
    res += [[ni, ny, n0] for ni in [n1, n2, n3, n4, n5, n6, n7]]
    res += [[ni, ny, n0] for ni in [n7, n6, n5, n4, n3, n2, n1]]

    def four(a, b, c):
        return [a] * 4 + [b] * 4 + [c] * 4
    gr = [four(g3, gy, g3), four(g4, gy, g3), four(g4, gy, 1. / g3)]
    gr += [four(g5, gy, g8), four(g5, gy, g7), four(g5, gy, 1. / g7), four(g5, gy, 1. / g8)]
    gr += [four(g6, gy, g8), four(g6, gy, g7), four(g6, gy, 1. / g7), four(g6, gy, 1. / g8)]
    gr += [four(g7, gy, g8), four(1. / g7, gy, g8)]
    gr += [four(g1, gy, g0), four(g2, gy, g0)]
    gr += [four(1. / g, gy, g0) for g in (g3, g4, g5, g6, g7)]
    gr += [four(g, gy, g0) for g in (g7, g6, g5, g4, g3)]
    gr += [four(1. / g2, gy, g0), four(1. / g1, gy, g0)]
    gr[13][8:] = [g10, g9, g9, g10]
    gr[14][8:] = [g0, g10, g10, g0]
    gr[25][8:] = [g10, g0, g0, g10]
    gr[26][8:] = [g9, g10, g10, g9]
    gr[13][:4] = [g1, g1, h1, h1]
    gr[14][:4] = [g2, g2, h2, h2]
    for k, (g, h) in zip(range(15, 20), ((g3, h3), (g4, h4), (g5, h5), (g6, h6), (g7, h7))):
        gr[k][:4] = [1 / g, 1 / g, 1 / h, 1 / h]
    for k, (g, h) in zip(range(20, 25), ((g7, h7), (g6, h6), (g5, h5), (g4, h4), (g3, h3))):
        gr[k][:4] = [g, g, h, h]
    gr[25][:4] = [1 / g2, 1 / g2, 1 / h2, 1 / h2]
    gr[26][:4] = [1 / g1, 1 / g1, 1 / h1, 1 / h1]
    return list(zip(res, gr))


def set_model_edges(model, case):
    '''Set the resolution and grading of every block of the mshGen model from case.meshing (edges and layers).'''
    for layer in range(model.N_layers):
        rg = layer_blocks(case.meshing.edges, int(case.meshing.layer_resolution[layer]),
                          float(case.meshing.layer_grading[layer]))
        for k, (r, g) in enumerate(rg):
            blk = model.blocks[layer * 27 + k]
            blk.resolution, blk.grading = list(r), list(g)


# ---------------------------------------------------------------------------------------------- #
# Geometry of the root section
# ---------------------------------------------------------------------------------------------- #


class _Side:
    '''One airfoil side (suction 's' or pressure 'p') of the root section: arc length (m, from the LE), curvature,
    segment ends TE -> LE as arc positions.'''

    def __init__(self, tck, chord, div):
        u = np.linspace(0.0, 1.0, N_SPLINE)
        x1, z1 = splev(u, tck, der=1)
        x2, z2 = splev(u, tck, der=2)
        speed = np.hypot(x1, z1)
        kappa = np.abs(x1 * z2 - z1 * x2) / np.maximum(speed, 1e-300) ** 3
        s = np.r_[0.0, np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))]
        self.s = s * chord                      # m, from the LE
        self.kappa = kappa / chord              # 1/m
        self.S = float(self.s[-1])
        # segment ends: TE (u = 1), vertices (1|3), (4|6), (7|9), (10|11), LE (u = 0)
        ends = [1.0] + list(div) + [0.0]
        self.ends = [float(np.interp(e, u, self.s)) for e in ends]


E08_EDGES = [(4, 16), (7, 18), (10, 20), (6, 17), (9, 19), (11, 21)]
E07_INNER = [(16, 5), (18, 8), (17, 5), (19, 8)]


class _Geometry:
    def __init__(self, case, airfoil, model):
        self.case = case
        self.chord = case.root_section()[0]
        pts = case.blocks.interior_points
        self.sides = [_Side(airfoil.vertex_tck["s"], self.chord, [pts[i][0] for i in (1, 4, 7, 10)]),
                      _Side(airfoil.vertex_tck["p"], self.chord, [pts[i][0] for i in (3, 6, 9, 11)])]
        v = np.array([vx.coord for vx in model.vertices[:39]], dtype=float)[:, [0, 2]]
        self.v = v

        def dist(a, b):
            return float(np.linalg.norm(v[a] - v[b]))
        radial = [(1, 27), (4, 28), (7, 29), (10, 30), (13, 31), (11, 32), (9, 33), (6, 34), (3, 35), (0, 26), (0, 36)]
        self.radial = np.array([dist(a, b) for a, b in radial])
        self.near_wake, self.far_wake = dist(0, 22), dist(22, 23)
        self.vert_22 = 0.5 * (dist(22, 25) + dist(22, 37))
        self.vert_23 = 0.5 * (dist(23, 24) + dist(23, 38))
        # beyond the tip (flow layers): the blocks inside the airfoil contour at the tip slice
        from .blocks import tip_slice
        t = tip_slice(case)
        self.tip = t if interface_limited(case) and t is not None else None
        if self.tip is not None:
            vt = np.array([vx.coord for vx in model.vertices[39 * t:39 * (t + 1)]], dtype=float)[:, [0, 2]]

            def dt(a, b):
                return float(np.linalg.norm(vt[a] - vt[b]))
            self.radial_tip = np.array([dt(a, b) for a, b in radial])
            # e08: from the contour inwards (4 -> 16, ...); e07 inside them (16 -> 5, ...)
            self.e08_edges = np.array([dt(a, b) for a, b in E08_EDGES])
            self.e07_inner = np.array([dt(a, b) for a, b in E07_INNER])


def _graded(s, h0, slope):
    '''Smallest field h <= h0 with |dh/ds| <= slope (exact 1D gradation, two sweeps).'''
    a = np.minimum.accumulate(h0 - slope * s) + slope * s
    b = (np.minimum.accumulate((h0 + slope * s)[::-1]))[::-1] - slope * s
    return np.minimum(a, b)


def _cum_cells(s, h):
    inv = 1.0 / h
    return np.r_[0.0, np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s))]


END_WEIGHT = 10.0      # weight of the end cells in the fits: neighbouring segments / layers meet at the same size


def _fit_grading(n, sides):
    '''
    Expansion ratio (last/first, first cell at the TE-side end) of one geometric series of n cells that follows the
    target size field best on both sides: least squares of ln(cell size / target size) over the cells, the end cells
    weighted END_WEIGHT (continuity with the neighbouring segments).
    sides: [(length, s at the TE-side end, s array, h array)] (s decreasing towards the LE).
    '''
    if n == 1:
        return 1.0
    w = np.ones(n)
    w[0] = w[-1] = END_WEIGHT

    def cost(lg):
        g = math.exp(lg)
        lam = preview.graded_lambda(n, g)
        c = 0.0
        for L, s_te, s, h in sides:
            d = np.diff(lam) * L
            mid = s_te - 0.5 * (lam[1:] + lam[:-1]) * L
            tgt = np.interp(mid, s, h)
            tgt[0], tgt[-1] = np.interp([s_te, s_te - L], s, h)     # end cells: the size at the segment ends
            c += float(np.sum(w * np.log(d / tgt) ** 2))
        return c
    r = minimize_scalar(cost, bounds=(-math.log(1e4), math.log(1e4)), method="bounded", options={"xatol": 1e-5})
    return float(math.exp(r.x))


# ---------------------------------------------------------------------------------------------- #
# 1. Sizing
# ---------------------------------------------------------------------------------------------- #


def _chordwise(geo, n_side, growth, refine=True):
    '''
    Cells and grad1 of e03-e07 for n_side cells per side. Size h = C x min(radius of curvature, RHO_CAP chords),
    TE size = TE_RATIO x the smallest (LE) size, graded with slope GROWTH_MARGIN (growth - 1); C such that the
    segments (the larger count of both sides) add up to n_side.
    Returns (counts, gradings, field) with field = per side (s, h) of the size field.
    '''
    slope = GROWTH_MARGIN * (growth - 1.0)
    ell = []
    for sd in geo.sides:
        e = np.minimum(1.0 / np.maximum(sd.kappa, 1e-12), RHO_CAP * geo.chord)
        ell.append(e)
    ell_min = min(float(e.min()) for e in ell)
    for e in ell:
        e[-1] = min(e[-1], TE_RATIO * ell_min)

    def fields(C):
        return [_graded(sd.s, C * e, slope) for sd, e in zip(geo.sides, ell)]

    def frac_counts(C):
        out = []
        for sd, h in zip(geo.sides, fields(C)):
            cum = _cum_cells(sd.s, h)
            at = np.interp(sd.ends, sd.s, cum)
            out.append(-np.diff(at))                    # TE -> LE segments
        return np.maximum(out[0], out[1])
    lo, hi = math.log(1e-7), math.log(100.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if frac_counts(math.exp(mid)).sum() > n_side:
            lo = mid
        else:
            hi = mid
    C = math.exp(hi)
    f = frac_counts(C)
    f = f * n_side / f.sum()
    n = np.maximum(1, np.floor(f)).astype(int)
    while n.sum() < n_side:                             # largest remainder
        n[int(np.argmax(f - n))] += 1
    while n.sum() > n_side and (n > 1).any():
        i = int(np.argmin(np.where(n > 1, f - n, np.inf)))
        n[i] -= 1
    hs = fields(C)
    grads = []
    for i in range(5):
        sides = []
        for sd, h in zip(geo.sides, hs):
            L = sd.ends[i] - sd.ends[i + 1]
            sides.append((L, sd.ends[i], sd.s, h))
        grads.append(_fit_grading(int(n[i]), sides))
    n = [int(v) for v in n]
    if refine:
        n, grads = _chord_refine(geo, n, grads, hs, growth)
    return n, grads, [(sd.s, h) for sd, h in zip(geo.sides, hs)], C


W_CHAIN = 50.0          # weight of size jumps above the max growth at the junctions of a chain of segments / layers


W_PIN = 100.0           # weight of a pinned end cell (the y+ cell at the tip face)


def _chain_cost(chain, n, lg, growth, pins=()):
    """
    Fit of a chain of graded segments (chordwise segments TE -> LE, or spanwise layers) to a size field: mean
    (ln cell size / target size)^2 over all cells + W_CHAIN x (jump above the max growth at the junctions)^2.
    chain: per segment a list of sides (L, start, direction, s, h): cell centres at start + direction x position.
    """
    lgm = math.log(growth)
    dev, pen = [], 0.0
    ends = []
    for i, sides in enumerate(chain):
        lam = preview.graded_lambda(n[i], math.exp(lg[i]))
        mid_frac = 0.5 * (lam[1:] + lam[:-1])
        row = []
        for L, start, direction, sg, h in sides:
            d = np.diff(lam) * L
            dev.append(np.log(d / np.interp(start + direction * mid_frac * L, sg, h)) ** 2)
            row.append((d[0], d[-1]))
        ends.append(row)
    for i in range(len(chain) - 1):
        for (_, last), (first, _) in zip(ends[i], ends[i + 1]):
            pen += max(abs(math.log(last / first)) - lgm, 0.0) ** 2
    pin = sum(math.log(ends[i][0][0 if which == "first" else 1] / size) ** 2 for i, which, size in pins)
    return float(np.mean(np.concatenate(dev))) + W_CHAIN * pen / lgm ** 2 + W_PIN * pin


def _chain_refine(chain, n, grads, growth, groups, pins=(), limits=None):
    """
    Gradings (L-BFGS-B; inside a segment the growth is at most the maximum: |ln g| <= (n - 1) ln growth) and the
    split of the cells (one cell moved between neighbouring segments of the same group while it helps; the totals
    of the groups stay the same) that minimise _chain_cost.
    """
    from scipy.optimize import minimize
    lgm = math.log(growth)

    def solve(nn, lg0):
        b = [(-(k - 1) * lgm, (k - 1) * lgm) for k in nn]
        for i, lim in (limits or {}).items():           # extra bounds (e.g. e03/e07 with flow layers)
            b[i] = (max(b[i][0], -lim), min(b[i][1], lim))
        x0 = np.clip(lg0, [v[0] for v in b], [v[1] for v in b])
        r = minimize(lambda x: _chain_cost(chain, nn, x, growth, pins), x0, method="L-BFGS-B", bounds=b,
                     options={"maxiter": 200})
        return r.x, float(r.fun)
    lg, j = solve(n, np.log(grads))
    m = max(1, max(n) // 8)                             # cells moved at once; halved when no move helps
    for _ in range(300):
        best = None
        for grp in groups:
            for a in grp:
                for b in (a - 1, a + 1):
                    if b in grp and n[a] > m:
                        nn = list(n)
                        nn[a] -= m
                        nn[b] += m
                        x, jj = solve(nn, lg)
                        if jj < j - 1e-9 and (best is None or jj < best[2]):
                            best = (nn, x, jj)
        if best is None:
            if m == 1:
                break
            m //= 2
            continue
        n, lg, j = best
    return list(n), [float(math.exp(v)) for v in lg]


INTERFACE_GRADING = 3.0    # with flow layers: e03 and e07 grad1 within 1/3 ... 3 (see interface_limited)


def interface_limited(case):
    '''
    True if blockMesh merges the blocks inside the airfoil contour beyond the tip (flow layers) with a non-conformal
    interface (mergePatchPairs, LIMITATIONS.md item 9). Across the airfoil thickness those blocks are graded with
    e03 (blocks 1, 2) and e07/e08 (blocks 3-6); strongly graded e03 and e07 together made OpenFOAM's face cutting
    fail ("Duplicate point found in cut face"), so their grad1 is limited to 1/INTERFACE_GRADING ... INTERFACE_GRADING.
    '''
    return "flow" in layer_types(case)


def _chord_chain(geo, hs):
    '''The five chordwise segments (TE -> LE) of both sides with their size fields, for _chain_cost.'''
    return [[(sd.ends[i] - sd.ends[i + 1], sd.ends[i], -1.0, sd.s, h) for sd, h in zip(geo.sides, hs)]
            for i in range(5)]


def _chord_refine(geo, n, grads, hs, growth):
    chain = _chord_chain(geo, hs)
    limits = None
    if interface_limited(geo.case):
        limits = {0: math.log(INTERFACE_GRADING), 4: math.log(INTERFACE_GRADING)}
    return _chain_refine(chain, n, grads, growth, [list(range(5))], limits=limits)


def _segment_sizes(geo, n, g):
    '''First (TE-side) and last (LE-side) cell of segment i on both sides: [(first, last)] x 2 per segment.'''
    out = []
    for i in range(5):
        row = []
        for sd in geo.sides:
            L = sd.ends[i] - sd.ends[i + 1]
            first = L * first_fraction(n[i], g[i])
            row.append((first, first * g[i]))
        out.append(row)
    return out


def _wall_normal(case, geo, yplus, growth):
    h = sizing.first_height_for_yplus(case, yplus, geo.chord)
    L = float(np.median(geo.radial[:9]))
    n0, g0 = series_first_growth(L, h, growth)
    return max(n0, 2), g0, h


def interior_e08(geo, n0, g0, n7, g7, growth):
    '''
    e08 beyond the tip (None without flow layers). The blocks inside the airfoil contour (3, 6, 7, 10, 11, 12) are
    graded across the thickness with e08 from the contour inwards; at the contour they meet the wall-normal cells of
    the blocks around the airfoil; at the other end they meet the e07 cells further inside (blocks 4, 5, 8, 9). So e08
    starts with the wall-normal first cell (e00, at the tip slice) and ends with the first e07 cell inside (one
    geometric series between both sizes, on the median edge length), if that keeps the growth within the maximum;
    otherwise it grows by the maximum up to that size. The quality search then refines its grading (stage 2).
    '''
    if geo.tip is None:
        return None
    h_c = float(np.median(geo.radial_tip[:9])) * first_fraction(n0, g0)
    L = float(np.median(geo.e08_edges))
    h_in = float(np.median(geo.e07_inner)) * first_fraction(n7, g7)
    if h_c >= L:
        return [1, 1.0, 1.0]
    n8 = None
    if h_c * 1.0001 < h_in < L:                     # both ends: h_c at the contour, h_in at the e07 cells
        n, g = sizing.series_first_last(L, h_c, h_in)
        if n > 1 and g ** (1.0 / (n - 1)) <= growth * (1 + 1e-9):
            n8 = n
    if n8 is None:
        n8, _ = series_first_growth(L, h_c, growth, max(h_in, h_c * growth))
    n8 = max(n8, 2)
    while True:                                     # first cell exactly h_c, growth within the maximum
        g8 = grading_for_first_fraction(n8, h_c / L)
        if g8 ** (1.0 / (n8 - 1)) <= growth * (1 + 1e-9) or n8 > 5000:
            return [n8, g8, g8]
        n8 += 1


def _wake(geo, n0, h_wall, h_te, h_cap, growth):
    '''e02, e01 (cells, grad1) and the gradings of e10 (at vertex 22) and e09 (at the outlet).'''
    n2, g2 = series_first_growth(geo.near_wake, h_te, growth, h_cap)
    last2 = geo.near_wake * first_fraction(n2, g2) * g2
    n1, g1 = series_first_growth(geo.far_wake, last2, growth)
    last1 = geo.far_wake * first_fraction(n1, g1) * g1

    def vertical(L, v):
        v = min(max(v, h_wall), L / n0)
        return grading_for_first_fraction(n0, v / L)
    return (n2, g2), (n1, g1), vertical(geo.vert_22, last2), vertical(geo.vert_23, last1)


def spanwise_field(case, n_wing, h_tip, h_far, growth, n=20001):
    '''
    Spanwise sizes: uniform H on the wing, h_tip at the tip slice, at most h_far beyond the tip, graded with slope
    GROWTH_MARGIN (growth - 1); H such that the wing has n_wing cells. Returns (y, h, H).
    '''
    from .blocks import tip_slice
    s = case.blocks.slice_locations
    y = np.linspace(s[0], s[-1], n)
    t = tip_slice(case)
    b = s[t] if t is not None else s[-1]
    y = np.unique(np.r_[y, s])
    slope = GROWTH_MARGIN * (growth - 1.0)
    i_tip = int(np.argmin(np.abs(y - b)))
    wing = y <= b + 1e-12

    def field(H):
        h0 = np.where(wing, H, max(h_far, H))
        h0[i_tip] = min(h_tip, H)
        return _graded(y, h0, slope)

    def count(H):
        h = field(H)
        cum = _cum_cells(y, h)
        return cum[i_tip]
    lo, hi = math.log(h_tip), math.log(max(b, h_tip * 1.0001))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if count(math.exp(mid)) > n_wing:
            lo = mid
        else:
            hi = mid
    H = math.exp(hi)
    return y, field(H), H


def _layers(case, y, h, growth, refine=True):
    '''Cells and grading (last/first) of every layer from the spanwise size field.'''
    from .blocks import tip_slice
    s = case.blocks.slice_locations
    tip = tip_slice(case)
    cum = _cum_cells(y, h)
    res, grad = [], []
    for k in range(len(s) - 1):
        a, b = np.interp([s[k], s[k + 1]], y, cum)
        n = max(1, int(round(b - a)))
        L = s[k + 1] - s[k]
        if n == 1:
            res.append(1)
            grad.append(1.0)
            continue
        sel = (y >= s[k]) & (y <= s[k + 1])
        w = np.ones(n)
        w[0] = w[-1] = END_WEIGHT

        def cost(lg):
            lam = preview.graded_lambda(n, math.exp(lg))
            d = np.diff(lam) * L
            mid = s[k] + 0.5 * (lam[1:] + lam[:-1]) * L
            tgt = np.interp(mid, y[sel], h[sel])
            tgt[0], tgt[-1] = np.interp([s[k], s[k + 1]], y, h)
            return float(np.sum(w * np.log(d / tgt) ** 2))
        r = minimize_scalar(cost, bounds=(-math.log(1e5), math.log(1e5)), method="bounded", options={"xatol": 1e-5})
        lim = (n - 1) * math.log(growth)                   # cell-to-cell growth inside the layer <= the maximum
        g = float(math.exp(min(max(r.x, -lim), lim)))
        res.append(n)
        grad.append(g)
    # chain refinement: gradings and the split of the cells (wing layers and flow layers separately)
    chain = [[(s[k + 1] - s[k], s[k], 1.0, y, h)] for k in range(len(s) - 1)]
    cut = tip if tip is not None else len(s) - 1
    groups = [list(range(0, cut)), list(range(cut, len(s) - 1))]
    pins = []
    if tip is not None:
        h_tip = float(np.interp(s[tip], y, h))
        pins = ([(tip - 1, "last", h_tip)] if tip > 0 else []) + ([(tip, "first", h_tip)] if tip < len(s) - 1 else [])
    if refine:
        res, grad = _chain_refine(chain, res, grad, growth, [g_ for g_ in groups if g_], pins)
    for k in range(len(s) - 1):                         # exact y+ cell at the tip face (more cells if the growth
        if tip is None or k not in (tip - 1, tip):      # inside the layer would exceed the maximum)
            continue
        L = s[k + 1] - s[k]
        f = float(np.interp(s[tip], y, h)) / L
        while True:
            n = res[k]
            gk = grading_for_first_fraction(n, f) if n > 1 else 1.0
            if n == 1 or gk ** (1.0 / (n - 1)) <= growth * (1 + 1e-9) or n > 5000:
                break
            res[k] += 1
        grad[k] = gk if k == tip else 1.0 / gk
    return res, grad


def wing_cells(case):
    '''Spanwise cells on the wing (wing and aileron layers).'''
    return int(sum(n for n, t in zip(case.meshing.layer_resolution, layer_types(case)) if t != "flow"))


def size_design(case, geo, t, frac, refine=True):
    '''
    Edges and layers of stage 1 for the fraction frac in [0, 1] of the count ranges. Returns (edges, layers, info).
    '''
    g = float(t["growth"])
    n_side = int(round(t["chord_min"] + frac * (t["chord_max"] - t["chord_min"])))
    n0, g0, h_wall = _wall_normal(case, geo, float(t["yplus"]), g)
    nc, gc, field, C = _chordwise(geo, n_side, g, refine)
    seg = _segment_sizes(geo, nc, gc)
    h_te = 0.5 * (seg[0][0][0] + seg[0][1][0])
    h_cap = max(float(h.max()) for _, h in field)
    (n2, g2), (n1, g1), g10, g9 = _wake(geo, n0, h_wall, h_te, h_cap, g)
    e08 = interior_e08(geo, n0, g0, nc[4], gc[4], g) or [1, 1.0, 1.0]
    e = {"e00": [n0, g0, g0], "e01": [n1, g1, g1], "e02": [n2, g2, g2], "e08": e08,
         "e09": [n0, g9, g9], "e10": [n0, g10, g10]}
    for k, n, gg in zip(CHORD, nc, gc):
        e[k] = [n, gg, gg]                              # grad2 = grad1: same fractions on the outer side
    info = {"field": field, "C": C, "h_wall": h_wall, "h_te": h_te, "n_side": n_side}
    layers = None
    if case.dim == "3D":
        n_wing = int(round(t["span_min"] + frac * (t["span_max"] - t["span_min"])))
        b2 = case.half_span
        h_tip = sizing.first_height_for_yplus(case, float(t["yplus"]),
                                              section_params(case.wing_stations, b2)[0])
        L_out = float(np.median(geo.radial[:9]))
        h_far = L_out * first_fraction(n0, g0) * g0     # outermost wall-normal cell
        y, h, H = spanwise_field(case, n_wing, h_tip, h_far, g)
        layers = _layers(case, y, h, g, refine)
        info.update(span=(y, h), H=H, h_tip=h_tip, n_wing=n_wing)
    return e, layers, info


def _with(case, edges, layers):
    c = copy.deepcopy(case)
    c.meshing.edges = {k: list(v) for k, v in edges.items()}
    if layers is not None:
        c.meshing.layer_resolution, c.meshing.layer_grading = list(layers[0]), list(layers[1])
    return c


def fill_budget(case, geo, t):
    '''Largest fraction of the count ranges with cells <= max_cells (bisection). Returns (frac, edges, layers, info).'''
    def design(frac, refine=False):
        e, lay, info = size_design(case, geo, t, frac, refine)
        return e, lay, info, estimate.cell_count(_with(case, e, lay))
    if design(1.0)[3] <= t["max_cells"]:
        frac = 1.0
    elif design(0.0)[3] > t["max_cells"]:
        frac = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if design(mid)[3] <= t["max_cells"]:
                lo = mid
            else:
                hi = mid
        frac = lo
    for _ in range(20):                                 # the refinement can add a few cells: stay in the budget
        out = design(frac, refine=True)
        if out[3] <= t["max_cells"] or frac <= 0:
            return (frac,) + out[:3]
        frac = max(0.0, frac - 0.02)
    return (frac,) + out[:3]


# ---------------------------------------------------------------------------------------------- #
# 2. Quality on the exact block mesh
# ---------------------------------------------------------------------------------------------- #
W_ORTH, W_ORTH_MAX, W_GROWTH, W_GROWTH_MAX = 1.0, 0.25, 1.0, 0.05
TAIL = 0.01                 # the worst 1 % of the faces form the tail terms (smoother than the single worst face)
W_SEVERE, SEVERE = 1.0, 65.0    # every face above 65 deg (checkMesh: 70) adds ((angle - 65) / 45)^2
W_INTERIOR = 0.5            # beyond the tip: weight of the blocks inside the airfoil contour (same terms, unweighted)
W_WALL = 1.0                # with the chordwise grad1 as variables: weight of their fit to the sizes of stage 1
FOCUS = 1.0                 # chords: weight of a face 1 / (1 + d / (FOCUS c))^2 with d its distance from the airfoil
EXTERIOR = list(range(13, 27))      # blocks around the airfoil (inside the contour: only in the tip slice, see _Evaluator)
# outer-side grading (grad2) of an edge parameter: blocks and the exponent of grad2 in their outer edge (4 -> 5)
OUTER = {"e01": [(13, 1), (26, -1)], "e02": [(14, 1), (25, -1)], "e03": [(15, -1), (24, 1)],
         "e04": [(16, -1), (23, 1)], "e05": [(17, -1), (22, 1)], "e06": [(18, -1), (21, 1)],
         "e07": [(19, -1), (20, 1)]}
VARIABLES = [(k, 2) for k in WAKE + CHORD] + [("e10", 2), ("e09", 2)]     # e09/e10: grad2 only (LIMITATIONS 12)
STEP = math.log(2.0)
BOUND = math.log(1e6)
PROJECTION_HITS = 0.8       # projection only where at least this share of the normals reaches the outer edge


def _ray_fraction(o, d, poly):
    '''Arc-length fraction of the point where the ray o + t d (t > 0) meets the polyline poly (m x 2); if it misses,
    the fraction of the polyline point closest to the ray. Returns (fraction, hit).'''
    A, B = poly[:-1], poly[1:]
    E = B - A
    seg = np.linalg.norm(E, axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    AO = A - o
    den = d[0] * E[:, 1] - d[1] * E[:, 0]
    ok = np.abs(den) > 1e-300
    den = np.where(ok, den, 1.0)
    t = (AO[:, 0] * E[:, 1] - AO[:, 1] * E[:, 0]) / den
    u = (AO[:, 0] * d[1] - AO[:, 1] * d[0]) / den
    hit = ok & (t > 0) & (u >= 0) & (u <= 1)
    if hit.any():
        j = int(np.argmin(np.where(hit, t, np.inf)))
        return float((cum[j] + u[j] * seg[j]) / cum[-1]), True
    rel = poly - o
    along = np.maximum(rel @ d, 0.0)
    dist = np.linalg.norm(rel - along[:, None] * d[None, :], axis=1)
    return float(cum[int(np.argmin(dist))] / cum[-1]), False


UPPER_CHAIN = [19, 18, 17, 16, 15, 14, 13]      # outer edges (4 -> 5) from the LE to the outlet, upper side
LOWER_CHAIN = [26, 25, 24, 23, 22, 21, 20]      # from the outlet to the LE, lower side
PARAM_OF = {k: (name, sign) for name, members in OUTER.items() for k, sign in members}


def outer_smooth(c, model):
    '''
    grad2 of e01-e07 so that the cell size is continuous along the outer boundary (C-arc, top, bottom): least squares
    of ln(size ratio) at the outer vertices plus growth above the maximum inside the edges (counts fixed). Changes c.
    '''
    from scipy.optimize import minimize
    from .model import active_edges
    set_model_edges(model, c)
    curves = active_edges(model, 0)
    coords = np.array([v.coord for v in model.vertices], dtype=float)
    names = [k for k, _ in VARIABLES if k in OUTER]
    lgm = math.log(max(targets(c)["growth"], 1.01))
    info = {}
    for k in UPPER_CHAIN + LOWER_CHAIN:
        v = model.blocks[k].vertices
        poly = preview._curve(curves, v[4], v[5])
        poly = poly if poly is not None else coords[[v[4], v[5]]]
        info[k] = (float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1))), int(model.blocks[k].resolution[0]))

    def ends(x):
        out = {}
        for k, (L, n) in info.items():
            name, sign = PARAM_OF[k]
            g = math.exp(sign * x[names.index(name)])
            first = L * first_fraction(n, g)
            out[k] = (first, first * g, n, g)
        return out

    def cost(x):
        e = ends(x)
        j = 0.0
        for chain in (UPPER_CHAIN, LOWER_CHAIN):
            for a, b in zip(chain, chain[1:]):
                j += math.log(e[a][1] / e[b][0]) ** 2
        j += math.log(e[19][0] / e[20][1]) ** 2                     # at the LE outer vertex
        for first, last, n, g in e.values():
            if n > 1:
                j += (n - 1) * max(abs(math.log(g)) / (n - 1) - lgm, 0.0) ** 2
        return j
    lim = [max(int(c.meshing.edges[k][0]) - 1, 0) * lgm for k in names]
    x0 = np.clip([math.log(c.meshing.edges[k][2]) for k in names], [-v for v in lim], lim)
    r = minimize(cost, x0, method="L-BFGS-B", bounds=[(-v, v) for v in lim])
    e = {k: list(v) for k, v in c.meshing.edges.items()}
    for name, v in zip(names, r.x):
        e[name][2] = float(math.exp(v))
    c.meshing.edges = e


def outer_projection(c, airfoil_file, model):
    '''
    grad2 of e01-e07 so that the grid lines leave the wall and the wake line along their normal: the wall (or wake
    line) nodes are projected along the normal onto the outer edge of their block, and one geometric distribution
    is fitted to these points (least squares, both blocks of the edge parameter together). Changes c.
    '''
    from .model import active_edges
    set_model_edges(model, c)
    blocks = {b["block"]: b["P"][..., [0, 2]] for b in preview.slice_mesh(c, airfoil_file, 0, model)}
    curves = active_edges(model, 0)
    coords = np.array([v.coord for v in model.vertices], dtype=float)[:, [0, 2]]
    e = {k: list(v) for k, v in c.meshing.edges.items()}
    for name, members in OUTER.items():
        data = []
        for k, sign in members:
            if k not in blocks:
                continue
            P = blocks[k]
            v = model.blocks[k].vertices
            poly = preview._curve(curves, v[4], v[5])
            poly = poly[:, [0, 2]] if poly is not None else coords[[v[4], v[5]]]
            inner = P[:, 0]
            tang = np.gradient(inner, axis=0)
            nrm = np.column_stack((-tang[:, 1], tang[:, 0]))
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
            mid = poly[len(poly) // 2]
            target, hits = [], 0
            for i in range(1, len(inner) - 1):
                dd = nrm[i] if np.dot(nrm[i], mid - inner[i]) > 0 else -nrm[i]
                lam, hit = _ray_fraction(inner[i], dd, poly)
                target.append(lam)
                hits += hit
            if hits < PROJECTION_HITS * (len(inner) - 2):
                data = []                   # the normals miss the outer edge (e.g. far downstream): keep grad2
                break
            data.append((len(inner) - 1, sign, np.maximum.accumulate(np.clip(target, 0, 1))))
        if not data:
            continue

        def cost(lh):
            s = 0.0
            for n, sign, tgt in data:
                lam = preview.graded_lambda(n, math.exp(sign * lh))[1:-1]
                s += float(np.sum((lam - tgt) ** 2))
            return s
        r = minimize_scalar(cost, bounds=(-BOUND, BOUND), method="bounded", options={"xatol": 1e-5})
        e[name][2] = float(math.exp(r.x))
    c.meshing.edges = e


def eval_slices(case):
    '''Root and tip slice (2D: the slice).'''
    from .blocks import tip_slice
    if case.dim != "3D":
        return [0]
    t = tip_slice(case)
    wing = [k for k in sizing.wing_slices(case)]
    tip = t if t is not None else (wing[-1] if wing else len(case.blocks.slice_locations) - 1)
    return sorted({0, tip})


def face_data(blocks, with_mask=False):
    '''Faces between cells of a slice (inside blocks and across block boundaries): non-orthogonality (deg), size
    ratio of the two cells normal to the face (>= 1) and face centres (x, z); with_mask also a boolean array: the face
    touches a block inside the airfoil contour (0-12).'''
    no, ratio, cen, mask = [], [], [], []
    sides = []
    for b in blocks:
        inner = b["block"] < 13
        P = b["P"][..., [0, 2]]
        C = _centres(P)
        dx = 0.5 * (np.linalg.norm(P[1:, :-1] - P[:-1, :-1], axis=-1) + np.linalg.norm(P[1:, 1:] - P[:-1, 1:], axis=-1))
        dz = 0.5 * (np.linalg.norm(P[:-1, 1:] - P[:-1, :-1], axis=-1) + np.linalg.norm(P[1:, 1:] - P[1:, :-1], axis=-1))
        if C.shape[0] > 1:
            a, bb = P[1:-1, :-1], P[1:-1, 1:]
            o, _ = _face_metrics(C[:-1], C[1:], a, bb)
            r = dx[1:] / dx[:-1]
            no.append(o.ravel())
            ratio.append(np.maximum(r, 1 / r).ravel())
            cen.append((0.5 * (a + bb)).reshape(-1, 2))
            mask.append(np.full(o.size, inner))
        if C.shape[1] > 1:
            a, bb = P[:-1, 1:-1], P[1:, 1:-1]
            o, _ = _face_metrics(C[:, :-1], C[:, 1:], a, bb)
            r = dz[:, 1:] / dz[:, :-1]
            no.append(o.ravel())
            ratio.append(np.maximum(r, 1 / r).ravel())
            cen.append((0.5 * (a + bb)).reshape(-1, 2))
            mask.append(np.full(o.size, inner))
        for nodes, cc, size in _sides(P).values():
            sides.append((nodes, cc, size, inner))
    index = {}
    for i, (nodes, _, _, _) in enumerate(sides):
        key = (tuple(np.round(nodes[0], 9)), tuple(np.round(nodes[-1], 9)))
        index.setdefault(key, []).append((i, False))
        index.setdefault(key[::-1], []).append((i, True))
    seen = set()
    for lst in index.values():
        if len(lst) < 2:
            continue
        (i, ri), (j, rj) = lst[0], lst[1]
        if i == j or (min(i, j), max(i, j)) in seen:
            continue
        seen.add((min(i, j), max(i, j)))
        ni, ci, si, ii = sides[i]
        nj, cj, sj, ij = sides[j]
        if ri:
            ni, ci, si = ni[::-1], ci[::-1], si[::-1]
        if rj:
            nj, cj, sj = nj[::-1], cj[::-1], sj[::-1]
        if len(ci) != len(cj) or not np.allclose(ni, nj, atol=1e-9 * (1 + np.abs(ni).max())):
            continue
        o, _ = _face_metrics(ci, cj, ni[:-1], ni[1:])
        r = sj / np.maximum(si, 1e-300)
        no.append(o)
        ratio.append(np.maximum(r, 1 / r))
        cen.append(0.5 * (ni[:-1] + ni[1:]))
        mask.append(np.full(o.size, ii or ij))
    out = np.concatenate(no), np.concatenate(ratio), np.concatenate(cen)
    return out + (np.concatenate(mask),) if with_mask else out


def wall_points(blocks):
    by = {b["block"]: b["P"] for b in blocks}
    pts = [by[k][:, 0][:, [0, 2]] for k in range(15, 25) if k in by]
    return np.concatenate(pts) if pts else None


def _stats(o, r, w, growth):
    '''Quality terms of a set of faces (non-orthogonality o in deg, size ratios r, weights w).'''
    lg = math.log(growth)
    ex = np.maximum(np.log(r) - lg, 0.0) / lg               # growth above the maximum, in units of ln(max)
    k = max(1, int(TAIL * len(o)))
    o2, ex2 = (o / 45.0) ** 2, ex ** 2
    return {"orth": float(np.sum(w * o2) / np.sum(w)), "orth_max": float(np.mean(np.partition(o2, -k)[-k:])),
            "growth": float(np.sum(w * ex2) / np.sum(w)), "growth_max": float(np.mean(np.partition(ex2, -k)[-k:])),
            "severe": float(np.sum((np.maximum(o - SEVERE, 0.0) / 45.0) ** 2)),
            "nonorth_max": float(o.max()), "nonorth_mean": float(np.mean(o)),
            "above40": float(np.mean(o > 40.0) * 100.0), "ratio_max": float(r.max()),
            "above_growth": float(np.mean(r > growth * 1.02) * 100.0)}


def _terms_value(q):
    return (W_ORTH * q["orth"] + W_ORTH_MAX * q["orth_max"] + W_GROWTH * q["growth"]
            + W_GROWTH_MAX * q["growth_max"] + W_SEVERE * q["severe"])


class _Evaluator:
    '''
    Quality terms of edges/layers on the exact block mesh of the evaluated slices (model with the block resolution and
    grading replaced, no mshGen call): the blocks around the airfoil (faces weighted by their distance from the
    airfoil); beyond the tip also the blocks inside the airfoil contour at the tip slice, with the faces across the
    contour (weight W_INTERIOR); with wall = the chordwise chain of stage 1, the fit of the chordwise grad1 to it.
    '''

    def __init__(self, case, airfoil_file, model, growth, wall=None):
        from .blocks import tip_slice
        self.f, self.model, self.growth, self.wall = airfoil_file, model, growth, wall
        self.slices = eval_slices(case)
        t = tip_slice(case)
        self.tip = t if (interface_limited(case) and t in self.slices) else None
        self.trees = {}

    def terms(self, c):
        set_model_edges(self.model, c)
        w_all, o_all, r_all, io, ir = [], [], [], [], []
        for k in self.slices:
            blocks = preview.slice_mesh(c, self.f, k, self.model)
            if k != self.tip:
                blocks = [b for b in blocks if b["block"] in EXTERIOR]
            no, ratio, cen, inner = face_data(blocks, with_mask=True)
            if k not in self.trees:
                wp = wall_points(blocks)
                chord = section_params(c.wing_stations or [[0, 0, 1.0, 0, 0, 0]], c.blocks.slice_locations[k])[0]
                self.trees[k] = (cKDTree(wp) if wp is not None else None, chord)
            tree, chord = self.trees[k]
            ext = ~inner
            d = tree.query(cen[ext])[0] if tree is not None else np.zeros(int(ext.sum()))
            w_all.append(1.0 / (1.0 + d / (FOCUS * chord)) ** 2)
            o_all.append(no[ext])
            r_all.append(ratio[ext])
            if inner.any():
                io.append(no[inner])
                ir.append(ratio[inner])
        q = _stats(np.concatenate(o_all), np.concatenate(r_all), np.concatenate(w_all), self.growth)
        if io:
            o, r = np.concatenate(io), np.concatenate(ir)
            q["interior"] = _stats(o, r, np.ones(len(o)), self.growth)
        return q

    def wall_cost(self, c):
        e = c.meshing.edges
        return _chain_cost(self.wall, [int(e[k][0]) for k in CHORD], [math.log(e[k][1]) for k in CHORD], self.growth)

    def objective(self, c):
        q = self.terms(c)
        j = _terms_value(q)
        if "interior" in q:
            j += W_INTERIOR * _terms_value(q["interior"])
        q["objective"] = j                                  # quality only (the result table)
        if self.wall is not None:
            q["wall"] = self.wall_cost(c)
            j += W_WALL * q["wall"]
        return j, q


def hooke_jeeves(f, x0, step, lo, hi, max_evals, min_step):
    '''
    Hooke-Jeeves pattern search: exploratory moves (+/- step per variable, keep improvements), then a pattern move
    along the last successful direction (x + (x - x_previous)), which follows diagonal valleys; steps are halved
    when nothing improves. Only improvements are kept. Returns (x, f(x), evaluations).
    '''
    evals = [0]

    def F(x):
        evals[0] += 1
        return f(np.clip(x, lo, hi))

    def explore(base, jb):
        x, j = base.copy(), jb
        for i in range(len(x)):
            for sgn in (1.0, -1.0):
                if evals[0] >= max_evals:
                    return x, j
                xt = x.copy()
                xt[i] = min(max(x[i] + sgn * step[i], lo[i]), hi[i])
                if xt[i] == x[i]:
                    continue
                jt = F(xt)
                if jt < j - 1e-9:
                    x, j = xt, jt
                    break
        return x, j
    x = np.clip(np.array(x0, float), lo, hi)
    j = F(x)
    step = np.array(step, float)
    while evals[0] < max_evals and np.any(step >= min_step):
        xn, jn = explore(x, j)
        if jn < j - 1e-9:
            while evals[0] < max_evals:                 # pattern moves while they help
                xp = np.clip(xn + (xn - x), lo, hi)
                x, j = xn, jn
                jp = F(xp)
                xe, je = explore(xp, jp)
                if je < j - 1e-9:
                    xn, jn = xe, je
                else:
                    break
        else:
            step *= 0.5
    return x, j, evals[0]


def variables(c):
    '''The variables of stage 2: (edge, 1 = grad1 | 2 = grad2). Beyond the tip also e08 and grad1 of e03-e07 (the
    blocks inside the airfoil contour take their cells across the thickness from them).'''
    from .blocks import tip_slice
    v = list(VARIABLES)
    if interface_limited(c) and tip_slice(c) is not None:
        v += [("e08", 1)] + [(k, 1) for k in CHORD]
    return v


def variable_bounds(c, growth, var=None):
    '''Bounds of ln(grading) of the variables: cell-to-cell growth inside an edge at most the maximum, |ln g| <=
    (n - 1) ln(growth) (a hard limit; growth across block boundaries is part of the objective); e03/e07 grad1 with flow
    layers within 1/INTERFACE_GRADING ... INTERFACE_GRADING.'''
    var = var or VARIABLES
    lim = []
    for k, side in var:
        v = max(int(c.meshing.edges[k][0]) - 1, 0) * math.log(growth)
        if side == 1 and k in ("e03", "e07") and interface_limited(c):
            v = min(v, math.log(INTERFACE_GRADING))
        lim.append(v)
    lim = np.array(lim)
    return -lim, lim


def _pattern_search(f, x0, lo, hi, max_evals):
    n = len(x0)
    return hooke_jeeves(f, np.clip(x0, lo, hi), np.full(n, STEP), lo, hi, max_evals, STEP / 32)


# ---------------------------------------------------------------------------------------------- #
# 3. y+
# ---------------------------------------------------------------------------------------------- #


def correct_yplus(c, airfoil_file, model, geo, t):
    '''Regrade e00 (and e09/e10 with it) so that the median y+ of the wing's first cells equals the target.'''
    g = float(t["growth"])
    for _ in range(3):
        set_model_edges(model, c)
        w = sizing.wall_yplus(c, airfoil_file, model)
        if not len(w["all"]):
            return None
        cur = float(np.median(w["all"]))
        n0, g0 = int(c.meshing.edges["e00"][0]), float(c.meshing.edges["e00"][1])
        if abs(cur / t["yplus"] - 1.0) < 0.002:
            return cur
        f = first_fraction(n0, g0) * t["yplus"] / cur
        n = n0
        while True:
            gg = grading_for_first_fraction(n, f)
            if gg ** (1.0 / (n - 1)) <= g * (1 + 1e-9) or n > 5000:
                break
            n += 1
        e = c.meshing.edges
        e["e00"] = [n, gg, gg]
        for k, L in (("e10", geo.vert_22), ("e09", geo.vert_23)):      # same first cell at the wake line
            v = L * first_fraction(n0, e[k][2])
            gk = grading_for_first_fraction(n, min(v / L, 1.0 / n))
            e[k] = [n, gk, gk]
    set_model_edges(model, c)
    return float(np.median(sizing.wall_yplus(c, airfoil_file, model)["all"]))


# ---------------------------------------------------------------------------------------------- #
# Report and the optimiser
# ---------------------------------------------------------------------------------------------- #


def growth_ratios(c, airfoil_file, model):
    '''Largest growth ratios (neighbouring cells) by direction, from the edges and the exact root slice.'''
    e = c.meshing.edges

    def r(n, g):
        n = int(n)
        return max(g, 1 / g) ** (1.0 / (n - 1)) if n > 1 else 1.0
    set_model_edges(model, c)
    blocks = preview.slice_mesh(c, airfoil_file, 0, model)
    surf = preview.surface_cells(blocks)
    out = {"wall-normal": r(e["e00"][0], e["e00"][1])}
    if surf:
        out["chordwise on the airfoil"] = max(surf["upper"]["max_jump"], surf["lower"]["max_jump"])
    by = {b["block"]: b["P"] for b in blocks}
    if 14 in by and 13 in by:
        pts = np.vstack((by[14][:, 0], by[13][1:, 0]))[:, [0, 2]]
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        q = d[1:] / d[:-1]
        out["wake line"] = float(np.max(np.maximum(q, 1 / q)))
        if surf:
            te = 0.5 * (surf["upper"]["length"][-1] + surf["lower"]["length"][-1])
            out["wake line"] = max(out["wake line"], d[0] / te, te / d[0])
    if c.dim == "3D":
        rs = [r(n, g) for n, g in zip(c.meshing.layer_resolution, c.meshing.layer_grading)]
        rs += [j["ratio"] for j in preview.layer_jumps(c)]
        out["spanwise"] = max(rs)
    return out


def report(case, airfoil_file, model=None, t=None):
    '''
    Rows {"metric", "value", "target", "ok"} of a mesh (case edges and layers) against the targets: the constraints
    (y+, counts, cells, growth) and the quality terms. model: mshGen model of the case (its blocks are regraded).
    '''
    from .model import mshgen_mesh
    t = t or targets(case)
    model = model or mshgen_mesh(case, airfoil_file, upto="edges")
    set_model_edges(model, case)
    rows = []

    def row(metric, value, target="", ok=None):
        rows.append({"metric": metric, "value": value, "target": target, "ok": ok})
    w = sizing.wall_yplus(case, airfoil_file, model)
    if len(w["all"]):
        med = float(np.median(w["all"]))
        row("y+ on the wing, median", med, "{:g}".format(t["yplus"]), abs(med / t["yplus"] - 1) <= 0.1)
        row("y+ on the wing, min / max", "{:.3g} / {:.3g}".format(w["all"].min(), w["all"].max()))
    if w["tip"]:
        row("y+ on the tip face", float(w["tip"][1]), "{:g}".format(t["yplus"]), abs(w["tip"][1] / t["yplus"] - 1) <= 0.25)
    nc = sum(int(case.meshing.edges[k][0]) for k in CHORD)
    row("chordwise cells per side", nc, "{}-{}".format(t["chord_min"], t["chord_max"]),
        t["chord_min"] <= nc <= t["chord_max"])
    if case.dim == "3D":
        ns = wing_cells(case)
        row("spanwise cells on the wing", ns, "{}-{}".format(t["span_min"], t["span_max"]),
            t["span_min"] <= ns <= t["span_max"])
    n = estimate.cell_count(case)
    row("cells", n, "<= {:,}".format(int(t["max_cells"])), n <= t["max_cells"])
    for k, v in growth_ratios(case, airfoil_file, model).items():
        row("largest growth ratio, " + k, v, "<= {:g}".format(t["growth"]), v <= t["growth"] * 1.02)
    blocks = preview.slice_mesh(case, airfoil_file, 0, model)
    surf = preview.surface_cells(blocks)
    c0 = case.root_section()[0]
    if surf:
        le = min(surf["upper"]["length"][0], surf["lower"]["length"][0])
        te = max(surf["upper"]["length"][-1], surf["lower"]["length"][-1])
        row("LE cell / chord (root)", le / c0)
        row("TE cell / chord (root)", te / c0)
        row("largest turning angle between wall faces (deg; LE curvature)", turning_angle(blocks))
    by = {b["block"]: b["P"] for b in blocks}
    if 14 in by:
        d = np.linalg.norm(np.diff(by[14][:, 0][:, [0, 2]], axis=0), axis=1)
        row("near wake: largest streamwise cell / chord", float(d.max()) / c0)
        P = by[14][-1, :][:, [0, 2]]
        row("near-wake end: vertical cell at the wake line / chord", float(np.linalg.norm(P[1] - P[0])) / c0)
    ratio = contour_ratio(case, airfoil_file, model)
    if ratio is not None:
        g = float(t["growth"])
        row("beyond the tip: first cell inside the contour / wall-normal first cell (e08 / e00)", ratio,
            "{:.2g}-{:.2g}".format(1 / g, g), 1 / (g * 1.02) <= ratio <= g * 1.02)
    ev = _Evaluator(case, airfoil_file, model, float(t["growth"]))
    j, q = ev.objective(case)
    sl = ", ".join(str(k) for k in ev.slices)
    row("non-orthogonality, mean (deg; slices {})".format(sl), q["nonorth_mean"])
    row("non-orthogonality, max (deg)", q["nonorth_max"], "< 70", q["nonorth_max"] < 70)
    row("faces above 40 deg (%)", q["above40"])
    row("faces with a size ratio above the max growth (%)", q["above_growth"])
    row("largest size ratio between neighbouring cells", q["ratio_max"])
    if "interior" in q:
        qi = q["interior"]
        row("inside the contour beyond the tip (slice {}): non-orthogonality mean / max (deg)".format(ev.tip),
            "{:.3g} / {:.3g}".format(qi["nonorth_mean"], qi["nonorth_max"]))
        row("inside the contour beyond the tip: faces with a size ratio above the max growth (%)", qi["above_growth"])
        row("inside the contour beyond the tip: largest size ratio between neighbouring cells", qi["ratio_max"])
    row("quality objective (smaller is better)", j)
    return rows


CONTOUR_START = {3: True, 7: True, 11: True, 12: True, 6: False, 10: False}     # e08 blocks: contour at j = 0?


def contour_ratio(case, airfoil_file, model):
    '''
    Beyond the tip: median first cell across the thickness in the e08 blocks at the airfoil contour (blocks 3, 6, 7,
    10, 11, 12) over the median first wall-normal cell of the blocks around it, at the tip slice. None without
    flow layers.
    '''
    from .blocks import tip_slice
    k = tip_slice(case)
    if k is None or not interface_limited(case):
        return None
    blocks = preview.slice_mesh(case, airfoil_file, k, model)
    by = {b["block"]: b["P"][..., [0, 2]] for b in blocks}
    surf = preview.surface_cells(blocks)
    if not surf or not all(b in by for b in CONTOUR_START):
        return None
    inner = []
    for b, start in CONTOUR_START.items():
        P = by[b]
        d = P[:, 1] - P[:, 0] if start else P[:, -1] - P[:, -2]
        inner.append(np.linalg.norm(d, axis=1))
    outer = np.r_[surf["upper"]["first_height"], surf["lower"]["first_height"]]
    return float(np.median(np.concatenate(inner)) / np.median(outer))


def turning_angle(blocks):
    '''Largest angle (deg) between neighbouring wall faces along the airfoil, TE excluded (curvature resolution:
    the LE region needs small cells where the wall turns fast). None if the airfoil is not in the slice.'''
    by = {b["block"]: b["P"] for b in blocks}
    if not all(k in by for k in range(15, 25)):
        return None
    parts = [by[k][:, 0][:, [0, 2]] for k in (24, 23, 22, 21, 20)]       # lower side, TE -> LE
    parts += [by[k][:, 0][:, [0, 2]] for k in (19, 18, 17, 16, 15)]      # upper side, LE -> TE
    pts = np.vstack([parts[0]] + [p[1:] for p in parts[1:]])
    d = np.diff(pts, axis=0)
    a = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
    return float(np.degrees(turn.max())) if len(turn) else 0.0


E07_START = {4: True, 8: True, 5: False, 9: False}       # e07 blocks next to the e08 blocks: that side at j = 0?


def e08_end_ratios(case, airfoil_file, model):
    '''
    Beyond the tip, at the tip slice (exact mesh): (first e08 cell / wall-normal first cell outside the contour, last
    e08 cell / first e07 cell inside), medians over the blocks. None without flow layers.
    '''
    from .blocks import tip_slice
    k = tip_slice(case)
    if k is None or not interface_limited(case):
        return None
    blocks = preview.slice_mesh(case, airfoil_file, k, model)
    by = {b["block"]: b["P"][..., [0, 2]] for b in blocks}
    surf = preview.surface_cells(blocks)
    if not surf or not all(b in by for b in list(CONTOUR_START) + list(E07_START)):
        return None

    def cells(spec, first):
        out = []
        for b, start in spec.items():
            P = by[b]
            if start == first:
                d = P[:, 1] - P[:, 0]
            else:
                d = P[:, -1] - P[:, -2]
            out.append(np.linalg.norm(d, axis=1))
        return float(np.median(np.concatenate(out)))
    wall = float(np.median(np.r_[surf["upper"]["first_height"], surf["lower"]["first_height"]]))
    e08_first = cells(CONTOUR_START, True)          # at the contour
    e08_last = cells(CONTOUR_START, False)          # at the inner end
    e07_first = cells(E07_START, True)              # next to the e08 blocks
    return e08_first / wall, e08_last / e07_first


def _update_e08(c, geo, t, airfoil_file=None, model=None):
    '''
    e08 from the current e00/e07 (interior_e08); with the model then corrected on the exact mesh of the tip slice so
    that the first cell inside the contour equals the wall-normal first cell outside and the last one the first e07
    cell inside (medians; one geometric series between both, the growth within the maximum).
    '''
    e = c.meshing.edges
    g = float(t["growth"])
    v = interior_e08(geo, int(e["e00"][0]), float(e["e00"][1]), int(e["e07"][0]), float(e["e07"][1]), g)
    if v is None:
        return
    e["e08"] = v
    if model is None:
        return
    for _ in range(4):
        set_model_edges(model, c)
        ratios = e08_end_ratios(c, airfoil_file, model)
        if ratios is None or (abs(ratios[0] - 1) < 0.01 and abs(ratios[1] - 1) < 0.02):
            return
        n8, g8 = int(e["e08"][0]), float(e["e08"][1])
        f1 = first_fraction(n8, g8) / ratios[0]                 # wanted first and last cell (fractions)
        f2 = first_fraction(n8, g8) * g8 / ratios[1]
        n, gg = sizing.series_first_last(1.0, f1, max(f2, f1 * 1.0001))
        n = max(n, 2)
        while True:                                             # first cell exact, growth within the maximum
            gg = grading_for_first_fraction(n, f1)
            if gg ** (1.0 / (n - 1)) <= g * (1 + 1e-9) or n > 5000:
                break
            n += 1
        e["e08"] = [n, gg, gg]


def optimise(case, airfoil_file, airfoil, t=None, max_evals=250, progress=None):
    '''
    Edges and layers from the targets. Returns {"edges", "layers" (or None), "before", "after", "note", "frac"};
    raises ValueError for invalid targets.
    '''
    from .model import mshgen_mesh
    t = t or targets(case)
    probs = target_problems(t, case.dim)
    if probs:
        raise ValueError("; ".join(probs))
    model = mshgen_mesh(case, airfoil_file, upto="edges")
    before = report(case, airfoil_file, model, t)
    geo = _Geometry(case, airfoil, model)
    if progress:
        progress("sizing: cell counts for the budget")
    frac, edges, layers, info = fill_budget(case, geo, t)
    c = _with(case, edges, layers)
    correct_yplus(c, airfoil_file, model, geo, t)
    e = c.meshing.edges
    e["e10"] = [e["e00"][0], e["e00"][1], e["e00"][2]]    # near-wake end graded like the TE: fine at the wake line
    _update_e08(c, geo, t, airfoil_file, model)
    var = variables(c)
    wall = _chord_chain(geo, [h for _, h in info["field"]]) if len(var) > len(VARIABLES) else None
    ev = _Evaluator(c, airfoil_file, model, float(t["growth"]), wall)
    if wall is not None:
        max_evals = max(max_evals, 350)
    if progress:
        progress("outer gradings: normal projection")
    starts = [[math.log(c.meshing.edges[k][side]) for k, side in var]]      # grad2 = grad1

    def uniform(p):
        for k, side in VARIABLES:
            if k in OUTER:
                p.meshing.edges[k][side] = 1.0

    def current(p):
        for k, side in VARIABLES:
            p.meshing.edges[k][side] = float(case.meshing.edges[k][side])
    names = ["grad2 = grad1", "normal projection", "smooth outer boundary", "uniform outer edges", "current gradings"]
    for fn in (lambda p: outer_projection(p, airfoil_file, model), lambda p: outer_smooth(p, model), uniform, current):
        p = copy.deepcopy(c)
        p.meshing.edges = {k: list(v) for k, v in c.meshing.edges.items()}
        fn(p)
        starts.append([math.log(p.meshing.edges[k][side]) for k, side in var])
    count = [0]
    best = [None, None]

    def apply(x, cc):
        e = {k: list(v) for k, v in cc.meshing.edges.items()}
        for (k, side), v in zip(var, x):
            e[k][side] = float(math.exp(v))
            if k in ("e09", "e10"):
                e[k][1] = e[k][2]
            elif k == "e08":
                e[k][2] = e[k][1]
        cc.meshing.edges = e

    def f(x):
        apply(x, c)
        j, q = ev.objective(c)
        count[0] += 1
        if best[0] is None or j < best[0]:
            best[0], best[1] = j, q
        if progress and count[0] % 5 == 0:
            progress("quality: evaluation {} of at most {}, objective {:.4f}".format(count[0], max_evals, best[0]))
        return j
    lo, hi = variable_bounds(c, float(t["growth"]), var)
    if ("e08", 1) in var:                       # the first cell inside the contour stays within the growth ratio of
        i = var.index(("e08", 1))               # the wall-normal cell outside (stage 1 matched them exactly)
        n8, g8 = int(c.meshing.edges["e08"][0]), float(c.meshing.edges["e08"][1])
        if n8 > 1:
            f0, G = first_fraction(n8, g8), 0.98 * float(t["growth"])
            lo[i] = max(lo[i], math.log(grading_for_first_fraction(n8, min(f0 * G, 1.0))))
            hi[i] = min(hi[i], math.log(grading_for_first_fraction(n8, f0 / G)))
    starts = [np.clip(x, lo, hi) for x in starts]
    js = [f(x) for x in starts]
    x0 = starts[int(np.argmin(js))]
    start_name = names[int(np.argmin(js))]
    j_start = min(js)
    x, j, _ = _pattern_search(f, x0, lo, hi, max_evals)
    apply(x, c)
    if progress:
        progress("y+ correction")
    correct_yplus(c, airfoil_file, model, geo, t)
    for k in c.meshing.edges:                            # rounding for the table
        n, g1, g2 = c.meshing.edges[k]
        c.meshing.edges[k] = [int(n), float("{:.5g}".format(g1)), float("{:.5g}".format(g2))]
    c.meshing.layer_grading = [float("{:.5g}".format(g)) for g in c.meshing.layer_grading]
    after = report(c, airfoil_file, model, t)
    note = ("{} of the count ranges ({} chordwise cells per side{}); search objective {:.3f} -> {:.3f} in {} "
            "evaluations from the start '{}' (slices {}{})").format(
        "all" if frac >= 1 else "{:.0%}".format(frac), info["n_side"],
        ", {} spanwise on the wing".format(info["n_wing"]) if "n_wing" in info else "", j_start, j, count[0],
        start_name, ", ".join(str(k) for k in ev.slices),
        "; inside the contour beyond the tip in slice {}".format(ev.tip) if ev.tip is not None else "")
    if frac <= 0 and estimate.cell_count(c) > t["max_cells"]:
        note += "; the minimum counts already exceed the cell budget"
    if interface_limited(c):
        note += "; e03/e07 grad1 within 1/{0:g}-{0:g} (merged interface beyond the tip)".format(INTERFACE_GRADING)
    from .case import EDGE_NAMES
    return {"edges": {k: c.meshing.edges[k] for k in EDGE_NAMES},
            "layers": ((c.meshing.layer_resolution, c.meshing.layer_grading)
                                                 if case.dim == "3D" else None),
            "before": before, "after": after, "note": note, "frac": frac}


# ---------------------------------------------------------------------------------------------- #
# Detailed analysis: every slice
# ---------------------------------------------------------------------------------------------- #
CHECKMESH = {"non-orthogonality": 70.0, "skewness": 4.0, "aspect ratio": 1000.0}


def slice_table(case, airfoil_file, model, t=None):
    '''
    Quantities of every slice from the exact block mesh: (rows {"quantity", "slice 0", ...}, warnings [(kind, text)]).
    model: mshGen model of the case as it is (not regraded). Warnings: folded cells; checkMesh limits (non-
    orthogonality 70 deg, skewness 4, aspect ratio 1000); growth above the target; y+ off the target by more than
    20 %; spanwise jumps. Known mshGen limitations beyond the tip (LIMITATIONS.md 23, 24) are reported as info.
    '''
    from .quality import slice_quality
    t = t or targets(case)
    g = float(t["growth"]) * 1.02
    s = case.blocks.slice_locations
    types = layer_types(case)
    yp = {k: v for k, _, _, _, _, v in sizing.wall_yplus(case, airfoil_file, model)["slices"]}
    jumps = {r["slice"]: r["ratio"] for r in preview.layer_jumps(case)}
    names = ["y (m)", "layers below | above", "cells in the slice", "y+ median", "y+ max", "smallest cell (m)",
             "largest cell (m)", "max aspect ratio, slice", "max aspect ratio, 3D", "max size ratio inside blocks",
             "max size ratio across blocks", "max non-orthogonality (deg)", "max skewness",
             "smallest corner angle (deg)", "folded cells", "spanwise cell size (m)", "spanwise jump at the slice"]
    cols, warn = [], []
    for k in range(len(s)):
        blocks = preview.slice_mesh(case, airfoil_file, k, model)
        dy = preview.spanwise_size(case, k)
        met = preview.cell_metrics(blocks, dy)
        q = slice_quality(blocks, dy)
        below = types[k - 1] if k > 0 else "-"
        above = types[k] if k < len(types) else "-"
        flow = types[min(k, len(types) - 1)] == "flow"
        v = {"y (m)": s[k], "layers below | above": "{} | {}".format(below, above),
             "cells in the slice": int(len(met["size"])),
             "y+ median": float(np.median(yp[k])) if k in yp else None,
             "y+ max": float(np.max(yp[k])) if k in yp else None,
             "smallest cell (m)": float(met["size"].min()), "largest cell (m)": float(met["size"].max()),
             "max aspect ratio, slice": float(met["ar"].max()), "max aspect ratio, 3D": float(met["ar3d"].max()),
             "max size ratio inside blocks": q["size ratio inside blocks"],
             "max size ratio across blocks": q["size ratio across blocks"],
             "max non-orthogonality (deg)": q["non-orthogonality (deg)"], "max skewness": q["skewness"],
             "smallest corner angle (deg)": float(met["angle"].min()), "folded cells": int(met["folded"].sum()),
             "spanwise cell size (m)": dy, "spanwise jump at the slice": jumps.get(k)}
        cols.append(v)
        name = "slice {} (y = {:.4g} m)".format(k, s[k])
        if v["folded cells"]:
            warn.append(("danger", "{}: {} folded cells (blockMesh would fail)".format(name, v["folded cells"])))
        if v["max non-orthogonality (deg)"] >= CHECKMESH["non-orthogonality"]:
            known = flow and "between" not in q["where"] and q["where"] == "block 0"
            warn.append(("info" if known else "warning",
                         "{}: non-orthogonality {:.1f} deg in {} (checkMesh limit 70){}".format(
                             name, v["max non-orthogonality (deg)"], q["where"],
                             "; the TE wedge beyond the tip, LIMITATIONS.md item 24" if known else "")))
        if v["max skewness"] > CHECKMESH["skewness"]:
            warn.append(("warning", "{}: skewness {:.2f} (checkMesh limit 4)".format(name, v["max skewness"])))
        if v["max aspect ratio, 3D"] > CHECKMESH["aspect ratio"]:
            warn.append(("warning", "{}: 3D aspect ratio {:.0f} (checkMesh reports above 1000)".format(
                name, v["max aspect ratio, 3D"])))
        if v["max size ratio inside blocks"] > g:
            warn.append(("warning", "{}: size ratio {:.2f} inside a block, above the max growth {:g}".format(
                name, v["max size ratio inside blocks"], t["growth"])))
        if v["max size ratio across blocks"] > g:
            if flow:
                warn.append(("info", "{}: size ratio {:.2f} across blocks beyond the tip (blocks inside the airfoil "
                                     "contour, LIMITATIONS.md item 23)".format(name, v["max size ratio across blocks"])))
            else:
                warn.append(("warning", "{}: size ratio {:.2f} across block boundaries, above the max growth {:g}"
                             .format(name, v["max size ratio across blocks"], t["growth"])))
        if v["y+ median"] is not None and abs(v["y+ median"] / t["yplus"] - 1) > 0.2:
            warn.append(("warning", "{}: y+ median {:.3g}, target {:g}".format(name, v["y+ median"], t["yplus"])))
        if v["spanwise jump at the slice"] is not None and v["spanwise jump at the slice"] > g:
            warn.append(("warning", "{}: spanwise size jump {:.2f}, above the max growth {:g}".format(
                name, v["spanwise jump at the slice"], t["growth"])))
    rows = []
    for n in names:
        r = {"quantity": n}
        for k, v in enumerate(cols):
            r["slice {}".format(k)] = v[n]
        if all(r["slice {}".format(k)] is None for k in range(len(cols))):
            continue
        rows.append(r)
    return rows, warn
