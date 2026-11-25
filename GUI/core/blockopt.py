'''
blockopt.py: suggest block vertices (step 3b) for the expected cell quality, before any cells are chosen.
Documented in GUI/docs/block_optimiser.md.

The block layout itself is evaluated (no cells, no mshGen call per evaluation): the vertices of a slice are
computed directly from the splines (exactly as mshGen places them) and the exterior positions. Objective:
  - wall: along the airfoil, the grid lines leaving the wall (straight lines from a wall point to the matching
    point of the outer block edge, as blockMesh's transfinite interpolation with straight radial edges gives
    them) against the wall normal; sampled along every wall segment
  - corners: block corner angles against 90 deg, weighted by the distance from the airfoil (priority close to
    the airfoil), including the corners on the outer boundary
  - spreading: neighbouring segments on the wall and on the outer boundary of similar length
Variables: airfoil vertices 1, 3, 4, 6, 7, 9, 10, 11 (along their spline; the camber vertices follow them)
and exterior vertices 22 (x, z), 23 (z), 25-28, 34-37 (x on top / bottom), 30-32 (angle on the C-arc);
29 and 33 sit at the junction of the C-arc and the straight top / bottom (x = 0). Constraints: order along
every spline and boundary with a minimum gap, all blocks valid in every evaluated slice. Search: coordinate
pattern search (only improvements are kept) from the current vertices and from a layout following the wall
normals; the result is checked with mshGen's vertices (block check of all slices).
'''
import copy
import math

import numpy as np
from scipy.interpolate import splev

from .blockcheck import BLOCKS_2D
from .blocks import aileron_slices, division_problems, radius, tip_slice
from .estimate import layer_types
from .planform import section_params

MOVE = {"airfoil and exterior vertices": ("airfoil", "exterior"), "exterior vertices only": ("exterior",),
        "airfoil vertices only": ("airfoil",)}
FOCUS = {"high (0.25 chord)": 0.25, "medium (1 chord)": 1.0, "low (4 chords)": 4.0}
NO_STATIONS = [[0, 0, 1.0, 0, 0, 0]]

WALL_BLOCKS = list(range(15, 25))           # BLOCKS_2D[k] = [a, b, c, d]: a-b on the airfoil, c above b, d above a
ARC_EDGES = {(29, 30), (30, 31), (31, 32), (32, 33)}
AIRFOIL_SIDE = [1, 3, 4, 6, 7, 9, 10, 11]
TOP, BOTTOM, ARC = [25, 26, 27, 28], [34, 35, 36, 37], [30, 31, 32]
DERIVED = ((16, 4, 5), (17, 5, 6), (18, 7, 8), (19, 8, 9), (20, 10, 14), (21, 14, 11))
GAP = 0.02                                  # minimum gap between vertices along a spline (position)
GAP_OUT = 0.05                              # ... along the straight top / bottom (fraction of R)
GAP_ARC = 0.15                              # ... along the C-arc (rad, about 9 deg)
NEAR_WAKE = (0.2, 2.0)                      # length of the near wake (TE -> vertex 22), root chords
N_SAMPLES = 12                              # samples per wall segment
W_CORNER, W_SPREAD, W_EXPANSION, W_WAKE, W_INTERIOR = 0.5, 0.5, 0.25, 1.0, 0.1
WALL_FREE = math.log(3.0)                   # neighbouring wall segments: length ratios up to 3 are not penalised
EXPANSION_FREE = math.log(3.0)              # neighbouring blocks: expansion ratios up to 3 apart are not penalised
WAKE_SCALE = 15.0                           # deg: wake line direction against the flow (x)
# Blocks from the outlet over the top and over the bottom towards the LE, with their inner edge (wake line or wall)
# and outer edge; expansion = outer length / inner length. The nose blocks 19 and 20 are left out: the LE segment
# is short on purpose (fine cells at the LE), so the nose blocks always expand much more than their neighbours.
EXPANSION_TOP = [(13, (22, 23), (25, 24)), (14, (0, 22), (26, 25))] + [(k, None, None) for k in range(15, 19)]
EXPANSION_BOTTOM = [(26, (22, 23), (37, 38)), (25, (0, 22), (36, 37))] + [(k, None, None) for k in range(24, 20, -1)]
INFEASIBLE = 1e3
STEP = {"pos": 0.03, "x": 0.03, "arc": 0.03, "x22": 0.05, "z22": 0.02, "z23": 0.03}


# ---------------------------------------------------------------------------------------------- #
# Geometry of a slice
# ---------------------------------------------------------------------------------------------- #

def _transform(sec, cog):
    '''(A, shift): placed = normalised @ A.T + shift (rotate by -beta about the LE, scale, shift), as mshGen.'''
    c, beta, dx, dz = sec
    a = math.radians(-beta)
    A = c * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return A, np.array([-(cog[0] + dx), -cog[2] + dz])


class _Slice:
    '''A slice with its section, active blocks and the corner weights' bookkeeping.'''

    def __init__(self, case, airfoil, k, interior):
        self.k, self.interior = k, interior
        y = case.blocks.slice_locations[k]
        self.sec = section_params(case.wing_stations or NO_STATIONS, y)
        self.dz, self.chord = self.sec[3], self.sec[0]
        self.A, self.b = _transform(self.sec, case.cog)
        self.outline = airfoil.outline()[::4] @ self.A.T + self.b
        self.active = list(range(13, 27)) + (list(range(13)) if interior else [])
        self.Q = np.array([BLOCKS_2D[k] for k in self.active])
        mask = np.ones(self.Q.shape, bool)               # wall corners: in the wall term (with the tangent)
        for i, k in enumerate(self.active):
            if k in WALL_BLOCKS:
                mask[i, :2] = False
        self.mask = mask
        self.block_w = np.array([W_INTERIOR if k < 13 else 1.0 for k in self.active])


def slice_vertices(airfoil, pts, ext, sl):
    '''(39 x 2) x, z of one slice, as mshGen generates them (without the aileron cut).'''
    xz = np.zeros((39, 2))
    for line in ("s", "c", "p"):
        idx = [i for i, (_, l) in enumerate(pts) if l == line]
        if idx:
            u = np.array([pts[i][0] for i in idx])
            xz[idx] = np.array(splev(u, airfoil.vertex_tck[line])).T @ sl.A.T + sl.b
    for i, a, b in DERIVED:
        xz[i] = 0.5 * (xz[a] + xz[b])
    xz[22:] = np.asarray(ext, float) + np.array([0.0, sl.dz])
    return xz


def _corners(xz, Q):
    '''Corner angles (deg), turns (positive: counter-clockwise) and size^2 of the blocks with corners Q (n x 4).'''
    p = xz[Q]
    prev, nxt = np.roll(p, 1, axis=1) - p, np.roll(p, -1, axis=1) - p
    cos = np.sum(prev * nxt, -1) / np.maximum(np.linalg.norm(prev, axis=-1) * np.linalg.norm(nxt, axis=-1), 1e-300)
    turn = -prev[..., 0] * nxt[..., 1] + prev[..., 1] * nxt[..., 0]
    size2 = np.max(np.ptp(p, axis=1), axis=-1) ** 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1))), turn, size2


def _outer_edge(xz, d, c, s, dz):
    '''Points of the outer block edge from vertex d (s = 0) to c (s = 1): arc (about (0, dz)) or straight.'''
    if (min(d, c), max(d, c)) in ARC_EDGES:
        p0, p1 = xz[d] - [0, dz], xz[c] - [0, dz]
        a0, a1 = math.atan2(p0[1], p0[0]) % (2 * math.pi), math.atan2(p1[1], p1[0]) % (2 * math.pi)
        r0, r1 = math.hypot(*p0), math.hypot(*p1)
        a, r = a0 + s * (a1 - a0), r0 + s * (r1 - r0)
        return np.stack((r * np.cos(a), r * np.sin(a) + dz), axis=1)
    return xz[d] + s[:, None] * (xz[c] - xz[d])


def _wall(airfoil, pts, xz, sl, n=N_SAMPLES):
    '''
    Per wall block (15-24): deviation (deg) of the grid lines leaving the wall from the wall normal at n samples
    along the segment, and the length of the segment. The sample at the TE is left out (a sharp TE has no normal).
    '''
    s = np.linspace(0, 1, n)
    out = []
    for line, blocks in (("s", WALL_BLOCKS[:5]), ("p", WALL_BLOCKS[5:])):
        u = []
        for k in blocks:
            a, b = BLOCKS_2D[k][:2]
            pa = 1.0 if a == 0 else 0.0 if a == 13 else pts[a][0]
            pb = 1.0 if b == 0 else 0.0 if b == 13 else pts[b][0]
            u.append(pa + s * (pb - pa))
        u = np.concatenate(u)
        W = np.array(splev(u, airfoil.vertex_tck[line])).T @ sl.A.T + sl.b
        T = np.array(splev(u, airfoil.vertex_tck[line], der=1)).T @ sl.A.T
        for i, k in enumerate(blocks):
            a, b, c, d = BLOCKS_2D[k]
            w, t = W[i * n:(i + 1) * n], T[i * n:(i + 1) * n]
            r = _outer_edge(xz, d, c, s, sl.dz) - w
            cos = np.abs(np.sum(r * t, 1)) / np.maximum(np.linalg.norm(r, axis=1) * np.linalg.norm(t, axis=1), 1e-300)
            dev = np.degrees(np.arcsin(np.clip(cos, 0, 1)))
            keep = np.ones(n, bool)
            keep[0], keep[-1] = a != 0, b != 0
            out.append((k, dev[keep], float(np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1)))))
    return out


def _distances(xz, sl):
    '''Distance of every vertex from the airfoil contour.'''
    d = xz[:, None, :] - sl.outline[None, :, :]
    return np.sqrt(np.min(np.sum(d * d, -1), axis=1))


# ---------------------------------------------------------------------------------------------- #
# Objective
# ---------------------------------------------------------------------------------------------- #

def slice_objective(airfoil, pts, ext, sl, focus):
    '''(objective, valid, xz, wall) of one slice.'''
    xz = slice_vertices(airfoil, pts, ext, sl)
    ang, turn, size2 = _corners(xz, sl.Q)
    valid = bool(np.all(turn > 1e-9 * size2[:, None]))
    wall = _wall(airfoil, pts, xz, sl)
    e = np.concatenate([d for _, d, _ in wall]) / 45.0
    j_wall = float(np.mean(e ** 2) + 0.25 * np.max(e) ** 2)
    dist = _distances(xz, sl)[sl.Q]
    w = sl.block_w[:, None] / (1.0 + dist / (focus * sl.chord)) ** 2 * sl.mask
    j_corner = float(np.sum(w * ((ang - 90.0) / 45.0) ** 2) / max(np.sum(w), 1e-300))
    j_spread = 0.0
    lengths = {k: L for k, _, L in wall}
    for seq in ([lengths[k] for k in range(15, 20)], [lengths[k] for k in range(20, 25)]):
        for a, b in zip(seq, seq[1:]):
            j_spread += max(0.0, abs(math.log(a / b)) - WALL_FREE) ** 2
    j_exp = 0.0
    for seq in expansions(xz, lengths):
        for a, b in zip(seq, seq[1:]):
            j_exp += max(0.0, abs(math.log(a / b)) - EXPANSION_FREE) ** 2
    j_wake = 0.0
    for i, j in ((0, 22), (22, 23)):
        d = xz[j] - xz[i]
        j_wake += (math.degrees(math.atan2(abs(d[1]), max(d[0], 1e-300))) / WAKE_SCALE) ** 2
    return (j_wall + W_CORNER * j_corner + W_SPREAD * j_spread + W_EXPANSION * focus * j_exp + W_WAKE * j_wake,
            valid, xz, wall)


def expansions(xz, wall_lengths):
    '''Expansion (outer edge length / inner edge length) of the blocks over the top and over the bottom, from
    the outlet towards the LE (without the nose blocks). The cell counts along both edges are the same, so
    neighbouring blocks with very different expansions give cell size jumps between them in the far field.'''
    out = []
    for seq in (EXPANSION_TOP, EXPANSION_BOTTOM):
        row = []
        for k, inner, outer in seq:
            if inner is None:
                a, b, c, d = BLOCKS_2D[k]
                li, lo = wall_lengths[k], math.hypot(*(xz[c] - xz[d]))
            else:
                li, lo = math.hypot(*(xz[inner[1]] - xz[inner[0]])), math.hypot(*(xz[outer[1]] - xz[outer[0]]))
            row.append(lo / max(li, 1e-300))
        out.append(row)
    return out


def order_problems(pts, ext, R, move=("airfoil", "exterior")):
    '''Order along the splines and the outer boundary (with minimum gaps); empty: admissible. The minimum gaps are
    checked for the vertices that are moved (the others are the user's; mshGen's block check judges them).'''
    p = division_problems(pts)
    pos = [v[0] for v in pts]
    for idx in ([1, 4, 7, 10], [3, 6, 9, 11]) if "airfoil" in move else ():
        vals = [1.0] + [pos[i] for i in idx] + [0.0]
        if any(a - b < GAP for a, b in zip(vals, vals[1:])):
            p.append("airfoil vertices closer than {} along a spline".format(GAP))
    if "exterior" not in move:
        return p
    x = {vid: ext[vid - 22][0] for vid in range(22, 39)}
    for name, vals in (("top", [x[29]] + [x[v] for v in (28, 27, 26, 25)] + [2 * R]),
                       ("bottom", [x[33]] + [x[v] for v in (34, 35, 36, 37)] + [2 * R])):
        if any(b - a < GAP_OUT * R for a, b in zip(vals, vals[1:])):
            p.append("exterior vertices out of order on the " + name)
    ang = [math.pi / 2] + [math.atan2(ext[v - 22][1], ext[v - 22][0]) % (2 * math.pi) for v in ARC] + [1.5 * math.pi]
    if any(b - a < GAP_ARC for a, b in zip(ang, ang[1:])):
        p.append("exterior vertices out of order on the C-arc")
    if abs(ext[1][1]) > 0.95 * R:
        p.append("vertex 23 must stay inside the outlet")
    if not ext[0][0] < 1.95 * R or abs(ext[0][1]) > 0.9 * R:
        p.append("vertex 22 must stay inside the domain")
    return p


def eval_slices(case, airfoil):
    '''Slices with their own section (root to tip; beyond the tip the section is the tip's), no aileron cut.
    The tip slice includes the blocks inside the contour if there are slices beyond the tip.'''
    n = len(case.blocks.slice_locations)
    t = tip_slice(case)
    last = min(t if t is not None else 0, n - 1)
    cut = set(aileron_slices(case))
    flow = any(tp == "flow" for tp in layer_types(case))
    out = [_Slice(case, airfoil, k, flow and k == last) for k in range(last + 1) if k not in cut]
    return out or [_Slice(case, airfoil, 0, False)]    # every slice has the aileron cut: the root anyway


def camber_rule(pts, pts0):
    '''Camber vertices follow their airfoil vertices: 2, 5, 8 keep their offset (position along the camber line)
    from the mean of 1-3, 4-6, 7-9, and 12 and 14 from the mean of 10-11, as when a division is moved in the GUI.
    (Placing them exactly between both sides would make the TE wedge block 0 degenerate.)'''
    out = [list(v) for v in pts]
    for c, s, p in ((2, 1, 3), (5, 4, 6), (8, 7, 9), (12, 10, 11), (14, 10, 11)):
        shift = 0.5 * (pts[s][0] + pts[p][0]) - 0.5 * (pts0[s][0] + pts0[p][0])
        out[c][0] = float(np.clip(round(pts0[c][0] + shift, 4), 0.005, 0.995))
    return out


# ---------------------------------------------------------------------------------------------- #
# Problem: variables <-> vertices
# ---------------------------------------------------------------------------------------------- #

class Problem:

    def __init__(self, case, airfoil, move=("airfoil", "exterior"), focus=1.0):
        self.case, self.airfoil, self.move, self.focus = case, airfoil, set(move), focus
        self.R = radius(case)
        self.c0 = case.root_section()[0]
        self.slices = eval_slices(case, airfoil)
        self.pts0 = [list(v) for v in case.blocks.interior_points]
        self.ext0 = [list(v) for v in case.blocks.exterior_points]
        names, lo, hi = [], [], []
        if "airfoil" in self.move:
            for i in AIRFOIL_SIDE:
                names.append(("pos", i))
                lo.append(0.01)
                hi.append(0.99)
        if "exterior" in self.move:
            for v in TOP + BOTTOM:
                names.append(("x", v))
                lo.append(0.0)
                hi.append(2.0)
            for v in ARC:
                names.append(("arc", v))
                lo.append(math.pi / 2)
                hi.append(1.5 * math.pi)
            xte = max(float(np.max(sl.outline[:, 0])) for sl in self.slices)
            names += [("x22", 22), ("z22", 22), ("z23", 23)]
            lo += [xte / self.c0 + NEAR_WAKE[0], -0.9 * self.R / self.c0, -0.95]
            hi += [min(1.95 * self.R / self.c0, xte / self.c0 + NEAR_WAKE[1]), 0.9 * self.R / self.c0, 0.95]
        self.names = names
        self.lo, self.hi = np.array(lo), np.array(hi)

    def vector(self, pts, ext):
        x = []
        for kind, v in self.names:
            x.append({"pos": lambda: pts[v][0], "x": lambda: ext[v - 22][0] / self.R,
                      "arc": lambda: math.atan2(ext[v - 22][1], ext[v - 22][0]) % (2 * math.pi),
                      "x22": lambda: ext[0][0] / self.c0, "z22": lambda: ext[0][1] / self.c0,
                      "z23": lambda: ext[1][1] / self.R}[kind]())
        return np.clip(np.array(x, float), self.lo, self.hi)

    def vertices(self, x):
        R = self.R
        pts = [list(v) for v in self.pts0]
        ext = [list(v) for v in self.ext0]
        for (kind, v), val in zip(self.names, x):
            val = float(val)
            if kind == "pos":
                pts[v][0] = val
            elif kind == "x":
                ext[v - 22] = [val * R, R if v in TOP else -R]
            elif kind == "arc":
                ext[v - 22] = [R * math.cos(val), R * math.sin(val)]
            elif kind == "x22":
                ext[0][0] = val * self.c0
            elif kind == "z22":
                ext[0][1] = val * self.c0
            else:
                ext[1] = [2 * R, val * R]
        if "exterior" in self.move:
            ext[29 - 22], ext[33 - 22] = [0.0, R], [0.0, -R]
        if "airfoil" in self.move:
            pts = camber_rule(pts, self.pts0)
        return pts, ext

    def objective(self, pts, ext, check=True):
        if check and order_problems(pts, ext, self.R, self.move):
            return INFEASIBLE
        total = 0.0
        for sl in self.slices:
            j, valid, _, _ = slice_objective(self.airfoil, pts, ext, sl, self.focus)
            if check and not valid:
                return INFEASIBLE
            total += j
        return total / len(self.slices)

    def __call__(self, x):
        return self.objective(*self.vertices(x))

    def spread(self, pts, ext):
        '''
        A start that keeps the minimum gaps of order_problems: the moved vertices along the splines, the top, the
        bottom and the C-arc moved as little as possible (least squares) to their gaps, the ends fixed (TE / LE,
        x = 0 / the outlet, the arc ends). mshGen's defaults are closer than the gaps for some airfoils (NACA2420,
        bottom: 0.04 R and 0.02 R). None if the vertices do not fit.
        '''
        R = self.R
        pts, ext = [list(v) for v in pts], [list(v) for v in ext]
        if "airfoil" in self.move:
            for idx in ([1, 4, 7, 10], [3, 6, 9, 11]):          # positions fall from the TE (1) to the LE (0)
                v = _spread([-1.0] + [-pts[i][0] for i in idx] + [0.0], GAP)
                if v is None:
                    return None
                for i, val in zip(idx, v):
                    pts[i][0] = -val
        if "exterior" in self.move:
            for vids, z in (([28, 27, 26, 25], R), ([34, 35, 36, 37], -R)):
                v = _spread([0.0] + [ext[i - 22][0] for i in vids] + [2 * R], GAP_OUT * R)
                if v is None:
                    return None
                for i, val in zip(vids, v):
                    ext[i - 22] = [val, z]
            ang = [math.atan2(ext[i - 22][1], ext[i - 22][0]) % (2 * math.pi) for i in ARC]
            v = _spread([math.pi / 2] + ang + [1.5 * math.pi], GAP_ARC)
            if v is None:
                return None
            for i, a in zip(ARC, v):
                ext[i - 22] = [R * math.cos(a), R * math.sin(a)]
        return pts, ext

    def normal_start(self, pts, ext):
        '''Exterior vertices where the wall normals of their airfoil vertices meet the boundary (root slice);
        26 and 36 across the wake line from the TE, 31 upstream of the LE.'''
        sl, R = self.slices[0], self.R
        xz = slice_vertices(self.airfoil, pts, ext, sl)
        ext = [list(v) for v in ext]
        for i, vid in {1: 27, 4: 28, 10: 30, 11: 32, 6: 34, 3: 35}.items():
            line = pts[i][1]
            t = np.array(splev(pts[i][0], self.airfoil.vertex_tck[line], der=1)) @ sl.A.T
            nrm = np.array([-t[1], t[0]]) if line == "s" else np.array([t[1], -t[0]])
            hit = _ray_to_boundary(xz[i], nrm / np.linalg.norm(nrm), R, sl.dz)
            if hit is None:
                continue
            if vid in TOP or vid in BOTTOM:
                ext[vid - 22] = [float(np.clip(hit[0], 0.0, 2 * R)), R if vid in TOP else -R]
            else:
                a = math.atan2(hit[1] - sl.dz, hit[0]) % (2 * math.pi)
                ext[vid - 22] = [R * math.cos(a), R * math.sin(a)]
        ext[26 - 22], ext[36 - 22] = [float(xz[0][0]), R], [float(xz[0][0]), -R]
        ext[31 - 22] = [-R, 0.0]
        ext[29 - 22], ext[33 - 22] = [0.0, R], [0.0, -R]
        return ext


def _spread(vals, gap):
    '''
    vals increasing with at least gap between neighbours, the first and the last fixed, the others moved as little
    as possible (least squares): pool adjacent violators on u_i = v_i - i gap, clipped to the ends. None if they do
    not fit between the ends.
    '''
    v = np.asarray(vals, dtype=float)
    gap *= 1.0 + 1e-9                        # still at least the gap after rounding
    i = np.arange(len(v))
    u = v - i * gap
    if u[-1] < u[0]:
        return None
    pools = []                                # [sum, count]
    for val in u[1:-1]:
        pools.append([val, 1])
        while len(pools) > 1 and pools[-2][0] / pools[-2][1] > pools[-1][0] / pools[-1][1]:
            s, n = pools.pop()
            pools[-1][0] += s
            pools[-1][1] += n
    fit = np.concatenate([np.full(n, s / n) for s, n in pools]) if pools else np.zeros(0)
    return [float(x) for x in np.clip(fit, u[0], u[-1]) + i[1:-1] * gap]


def _ray_to_boundary(p, d, R, dz):
    '''First point where the ray p + t d (t > 0) meets the C-domain boundary (arc for x < 0, z = +-R for x >= 0).'''
    q = np.array([p[0], p[1] - dz])
    hits = []
    for zb in (R, -R):
        if abs(d[1]) > 1e-12:
            t = (zb - q[1]) / d[1]
            if t > 0 and q[0] + t * d[0] >= 0:
                hits.append(t)
    bq, cq = 2 * q @ d, q @ q - R * R
    disc = bq * bq - 4 * cq
    if disc >= 0:
        t = (-bq + math.sqrt(disc)) / 2
        if t > 0 and q[0] + t * d[0] < 0:
            hits.append(t)
    return None if not hits else np.array([p[0] + min(hits) * d[0], p[1] + min(hits) * d[1]])


def pattern_search(f, x0, lo, hi, steps, max_evals, min_factor=1.0 / 32):
    '''
    Coordinate pattern search: per variable try +step and -step, keep a step that improves the objective (and
    grow it), halve all steps after a sweep without improvement. Only improvements are kept.
    '''
    x, j = x0.copy(), f(x0)
    step, evals = steps.copy(), 1
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
                if jt < j - 1e-9:
                    x, j, improved = xt, jt, True
                    step[i] *= 1.5
                    break
        if not improved:
            step *= 0.5
            if np.all(step < steps * min_factor):
                break
    return x, j, evals


# ---------------------------------------------------------------------------------------------- #
# Metrics and the suggestion
# ---------------------------------------------------------------------------------------------- #

def metrics(case, airfoil, pts, ext, focus=1.0):
    '''Quality of a block layout over the evaluated slices (for the before / after table).'''
    prob = Problem(case, airfoil, focus=focus)
    dev = {"suction": [], "pressure": []}
    near, allc, jumps, wake = [], [], [], []
    for sl in prob.slices:
        xz = slice_vertices(airfoil, pts, ext, sl)
        wall = _wall(airfoil, pts, xz, sl, 41)
        for k, d, _ in wall:
            dev["suction" if k <= 19 else "pressure"].append(d)
        for seq in expansions(xz, {k: L for k, _, L in wall}):
            jumps += [max(a / b, b / a) for a, b in zip(seq, seq[1:])]
        for i, j in ((0, 22), (22, 23)):
            d = xz[j] - xz[i]
            wake.append(abs(math.degrees(math.atan2(d[1], d[0]))))
        ext_blocks = [i for i, k in enumerate(sl.active) if k >= 13]
        ang = _corners(xz, sl.Q[ext_blocks])[0]
        dist = _distances(xz, sl)[sl.Q[ext_blocks]]
        m = sl.mask[ext_blocks]
        allc += list(np.abs(ang[m] - 90))
        near += list(np.abs(ang[m & (dist < sl.chord)] - 90))
    out = {}
    for side in ("suction", "pressure"):
        d = np.concatenate(dev[side])
        out["wall, {} side: largest deviation from normal (deg)".format(side)] = float(d.max())
        out["wall, {} side: mean deviation from normal (deg)".format(side)] = float(d.mean())
    out["block corners within 1 chord: largest deviation from 90 deg"] = float(max(near, default=0.0))
    out["block corners, all: largest deviation from 90 deg"] = float(max(allc, default=0.0))
    out["expansion of neighbouring blocks: largest ratio"] = float(max(jumps, default=1.0))
    out["wake line against the flow direction (deg)"] = float(max(wake))
    out["objective (smaller is better)"] = prob.objective(pts, ext, check=False)
    return out


def cell_quality(case, airfoil_file, slices, rows=5):
    '''
    Non-orthogonality of the cells (current cell counts and gradings of step 4) in the blocks around the airfoil
    (13-26) of the given slices, from the exact block mesh (core/preview.py, checkMesh's definition in the slice
    plane; faces inside blocks): share of faces above 40 deg, mean, and mean / largest in the first rows of cells
    at the wall. Only for information: the optimiser itself does not use cells.
    '''
    from . import preview, quality
    from .model import mshgen_mesh
    model = mshgen_mesh(case, airfoil_file, upto="edges")
    allv, wall = [], []
    for k in slices:
        for b in preview.slice_mesh(case, airfoil_file, k, model):
            if b["block"] < 13:
                continue
            P = b["P"][..., [0, 2]]
            C = quality._centres(P)
            o1 = quality._face_metrics(C[:-1], C[1:], P[1:-1, :-1], P[1:-1, 1:])[0]
            o2 = quality._face_metrics(C[:, :-1], C[:, 1:], P[:-1, 1:-1], P[1:, 1:-1])[0]
            allv += [o1.ravel(), o2.ravel()]
            if b["block"] in WALL_BLOCKS:                   # second index: away from the wall
                wall += [o1[:, :rows].ravel(), o2[:, :rows].ravel()]
    a, w = np.concatenate(allv), np.concatenate(wall)
    return {"cells (current counts and gradings): faces above 40 deg (%)": float(100 * np.mean(a > 40)),
            "cells: mean non-orthogonality (deg)": float(a.mean()),
            "cells, first {} rows at the wall: mean non-orthogonality (deg)".format(rows): float(w.mean()),
            "cells, first {} rows at the wall: largest non-orthogonality (deg)".format(rows): float(w.max())}


def _rounded(prob, pts, ext):
    '''Positions to 4 decimals, exterior vertices to 0.1 mm (on their boundary); unrounded if rounding would break
    a minimum gap (the optimum often lies on one).'''
    R = prob.R
    rp = [[round(p, 4), l] for p, l in pts]
    re = [[round(x, 4), round(z, 4)] for x, z in ext]
    if "exterior" in prob.move:
        for v in TOP + [29]:
            re[v - 22][1] = R
        for v in BOTTOM + [33]:
            re[v - 22][1] = -R
        re[1][0] = 2 * R
        for v in ARC:
            a = math.atan2(re[v - 22][1], re[v - 22][0])
            re[v - 22] = [R * math.cos(a), R * math.sin(a)]
    return (rp, re) if prob.objective(rp, re) < INFEASIBLE else (pts, ext)


def suggest(case, airfoil, airfoil_file, move="airfoil and exterior vertices", focus="medium (1 chord)",
            max_evals=3000, progress=None):
    '''
    Suggested block vertices: {"interior_points", "exterior_points", "before", "after", "note"}.
    Raises ValueError if no valid layout better than the current one was found.
    '''
    from .blockcheck import check
    prob = Problem(case, airfoil, MOVE.get(move, move), FOCUS.get(focus, focus))
    j_now = prob.objective(prob.pts0, prob.ext0)
    starts = [("current vertices", prob.pts0, prob.ext0)]
    if "exterior" in prob.move:
        pts = prob.pts0
        starts.append(("wall normals", pts, prob.normal_start(pts, prob.ext0)))
    steps = np.array([STEP[kind] for kind, _ in prob.names])
    best, evals = None, 0
    for n, (name, pts, ext) in enumerate(starts):
        x0 = prob.vector(pts, ext)
        if prob(x0) >= INFEASIBLE:                          # closer than the minimum gaps: spread them first
            fixed = prob.spread(*prob.vertices(x0))
            if fixed is None:
                continue
            x0, name = prob.vector(*fixed), name + " (spread to the minimum gaps)"
            if prob(x0) >= INFEASIBLE:
                continue
        x, j, ne = pattern_search(prob, x0, prob.lo, prob.hi, steps, max_evals)
        evals += ne
        if progress:
            progress(n + 1, len(starts))
        if j < INFEASIBLE and (best is None or j < best[1]):
            best = (x, j, name)
    if best is None:
        raise ValueError("no admissible start: the current block vertices are invalid or out of order")
    pts, ext = _rounded(prob, *prob.vertices(best[0]))
    j_new = prob.objective(pts, ext)
    if j_now < INFEASIBLE and j_new >= j_now - 1e-9:
        raise ValueError("the current block vertices are already the best found for these settings")
    trial = copy.deepcopy(case)
    trial.blocks.interior_points, trial.blocks.exterior_points = pts, ext
    problems = check(trial, airfoil_file)
    if problems:
        raise ValueError("the suggested vertices fail mshGen's block check: " + problems[0])
    cell_slices = sorted({prob.slices[0].k, prob.slices[-1].k})
    before = metrics(case, airfoil, prob.pts0, prob.ext0, prob.focus)
    after = metrics(trial, airfoil, pts, ext, prob.focus)
    before.update(cell_quality(case, airfoil_file, cell_slices))
    after.update(cell_quality(trial, airfoil_file, cell_slices))
    return {"interior_points": pts, "exterior_points": ext, "before": before, "after": after,
            "note": "{} evaluations, best start: {}; slices evaluated: {}".format(
                evals, best[2], ", ".join(str(sl.k) for sl in prob.slices))}
