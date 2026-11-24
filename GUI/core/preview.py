'''
preview.py: 2D mesh of one slice as blockMesh builds it, and cell metrics (no OpenFOAM needed)

For every active block the face in the slice is meshed from exactly what mshGen gives blockMesh:
corner vertices, cell counts, the 12 edge gradings and the curved edges (polyLines along the airfoil,
C-front arcs), with blockMesh's own point interpolation (OpenFOAM-v2112 block::createPoints). Edge points
follow the graded distribution by arc length. tests/test_m3.py compares the preview with blockMesh.

Metrics per cell: size (sqrt of the area), aspect ratio in the slice and in 3D (with the spanwise cell
size at the slice), size ratio to neighbouring cells, smallest corner angle, folded cells (negative area).
'''
import numpy as np

from .blockcheck import active_blocks
from .estimate import graded_sizes, layer_types
from .model import active_edges, mshgen_mesh

N_BLOCKS = 27
# exterior blocks touching the airfoil: chordwise edge parameter of their airfoil side
SURFACE_EDGE = {15: "e03", 16: "e04", 17: "e05", 18: "e06", 19: "e07",
                20: "e07", 21: "e06", 22: "e05", 23: "e04", 24: "e03"}
UPPER_LE_TO_TE = [19, 18, 17, 16, 15]          # i increases towards the TE in these blocks
LOWER_LE_TO_TE = [20, 21, 22, 23, 24]          # i increases towards the LE in these blocks


def graded_lambda(n, g):
    '''n+1 parameters in [0, 1] of n cells with expansion ratio g (last/first), as blockMesh.'''
    n = int(n)
    if n == 1 or abs(g - 1.0) < 1e-12:
        return np.linspace(0.0, 1.0, n + 1)
    r = g ** (1.0 / (n - 1))
    return (r ** np.arange(n + 1) - 1.0) / (r ** n - 1.0)


def curve_at(pts, t):
    '''Points at arc-length fractions t (any shape) on the polyline pts (m x 3).'''
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    s /= s[-1]
    tt = np.clip(np.ravel(t), 0.0, 1.0)
    out = np.column_stack([np.interp(tt, s, pts[:, i]) for i in range(3)])
    return out.reshape(np.shape(t) + (3,))


# The 12 edges of a hex block (local vertex numbers), in blockMesh's order: x (0-3), y (4-7), z (8-11)
HEX_EDGES = [(0, 1), (3, 2), (7, 6), (4, 5), (0, 3), (1, 2), (5, 6), (4, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def block_face_points(corners, curves, resolution, grading, top):
    '''
    Points (nx+1, nz+1, 3) of the bottom (j = 0) or top (j = ny) face of a hex block, with blockMesh's
    interpolation (OpenFOAM-v2112 src/mesh/blockMesh/blocks/block/blockCreate.C, block::createPoints):
    weighted straight-edge points of all 12 edges plus the curved-edge corrections.
    corners: (8, 3); curves: 12 point arrays (or None for straight edges); grading: 12 expansion ratios.
    '''
    nx, ny, nz = (int(n) for n in resolution)
    cells = [nx] * 4 + [ny] * 4 + [nz] * 4
    j = ny if top else 0
    shape = (nx + 1, nz + 1)
    I, K = np.arange(nx + 1)[:, None], np.arange(nz + 1)[None, :]
    w = []
    for e in range(12):
        lam = graded_lambda(cells[e], grading[e])
        idx = I if e < 4 else (j if e < 8 else K)
        w.append(np.broadcast_to(lam[idx], shape).astype(float))
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11 = w
    wx = [(1 - w0) * (1 - w4) * (1 - w8) + w0 * (1 - w5) * (1 - w9), (1 - w1) * w4 * (1 - w11) + w1 * w5 * (1 - w10),
          (1 - w2) * w7 * w11 + w2 * w6 * w10, (1 - w3) * (1 - w7) * w8 + w3 * (1 - w6) * w9]
    wy = [(1 - w4) * (1 - w0) * (1 - w8) + w4 * (1 - w1) * (1 - w11), (1 - w5) * w0 * (1 - w9) + w5 * w1 * (1 - w10),
          (1 - w6) * w3 * w9 + w6 * w2 * w10, (1 - w7) * (1 - w3) * w8 + w7 * (1 - w2) * w11]
    wz = [(1 - w8) * (1 - w0) * (1 - w4) + w8 * (1 - w3) * (1 - w7), (1 - w9) * w0 * (1 - w5) + w9 * w3 * (1 - w6),
          (1 - w10) * w1 * w5 + w10 * w2 * w6, (1 - w11) * (1 - w1) * w4 + w11 * (1 - w2) * w7]
    imp = []
    for grp in (wx, wy, wz):
        s = grp[0] + grp[1] + grp[2] + grp[3]
        imp += [g / s for g in grp]
    pos = np.zeros(shape + (3,))
    cor = np.zeros(shape + (3,))
    for e, (a, b) in enumerate(HEX_EDGES):
        straight = corners[a] + (corners[b] - corners[a]) * w[e][..., None]
        pos += imp[e][..., None] * straight
        if curves[e] is not None:
            cor += imp[e][..., None] * (curve_at(curves[e], w[e]) - straight)
    return pos / 3.0 + cor


def _curve(edges, a, b):
    if (a, b) in edges:
        return edges[(a, b)]
    if (b, a) in edges:
        return edges[(b, a)][::-1]
    return None


def slice_mesh(case, airfoil_file, slice_index, mesh=None, blocks=None):
    '''
    Blocks of the slice: list of {"block", "P" (nx+1, nz+1, 3)} for the blocks of the adjacent layer (blocks: these
    block numbers instead of the active ones, e.g. all 27 to draw the layout; inactive blocks get straight edges).
    '''
    mesh = mesh or mshgen_mesh(case, airfoil_file, upto="edges")
    n_layers = mesh.N_layers
    layer = min(slice_index, n_layers - 1)
    top = slice_index == layer + 1
    coords = np.array([v.coord for v in mesh.vertices], dtype=float)
    edges = active_edges(mesh, slice_index)
    out = []
    for k in (active_blocks(case)[layer] if blocks is None else blocks):
        blk = mesh.blocks[layer * N_BLOCKS + k]
        v = blk.vertices
        # curved edges of this slice (edges of the other face do not contribute to this face)
        curves = [_curve(edges, v[a], v[b]) for a, b in HEX_EDGES]
        P = block_face_points(coords[v], curves, blk.resolution, blk.grading, top)
        out.append({"block": k, "P": P})
    return out


def spanwise_size(case, slice_index):
    '''Spanwise cell size at a slice (smaller of the two adjacent cells); 2D: None.'''
    if case.dim != "3D":
        return None
    s, res, grad = case.blocks.slice_locations, case.meshing.layer_resolution, case.meshing.layer_grading
    sizes = []
    if slice_index > 0:
        k = slice_index - 1
        sizes.append(graded_sizes(s[k + 1] - s[k], res[k], grad[k])[1])
    if slice_index < len(s) - 1:
        k = slice_index
        sizes.append(graded_sizes(s[k + 1] - s[k], res[k], grad[k])[0])
    return min(sizes)


def cell_metrics(blocks, dy=None):
    '''Per cell (flat arrays over all blocks): corners, block, size, aspect ratios, neighbour ratio, angle.'''
    out = {k: [] for k in ("xs", "zs", "block", "size", "ar", "ar3d", "ratio", "angle", "dx", "dz", "folded")}
    for blk in blocks:
        P = blk["P"][..., [0, 2]]
        c00, c10, c11, c01 = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
        ex0, ex1 = np.linalg.norm(c10 - c00, axis=-1), np.linalg.norm(c11 - c01, axis=-1)
        ez0, ez1 = np.linalg.norm(c01 - c00, axis=-1), np.linalg.norm(c11 - c10, axis=-1)
        dx, dz = 0.5 * (ex0 + ex1), 0.5 * (ez0 + ez1)
        area = 0.5 * ((c00[..., 0] * c10[..., 1] - c10[..., 0] * c00[..., 1])
                      + (c10[..., 0] * c11[..., 1] - c11[..., 0] * c10[..., 1])
                      + (c11[..., 0] * c01[..., 1] - c01[..., 0] * c11[..., 1])
                      + (c01[..., 0] * c00[..., 1] - c00[..., 0] * c01[..., 1]))
        lo, hi = np.minimum(dx, dz), np.maximum(dx, dz)
        ar = hi / np.maximum(lo, 1e-300)
        if dy is not None:
            ar3d = np.maximum(hi, dy) / np.maximum(np.minimum(lo, dy), 1e-300)
        else:
            ar3d = ar
        # size ratio to the neighbours inside the block (x: dx, z: dz)
        ratio = np.ones_like(dx)
        if dx.shape[0] > 1:
            rx = dx[1:] / np.maximum(dx[:-1], 1e-300)
            rx = np.maximum(rx, 1 / rx)
            ratio[1:] = np.maximum(ratio[1:], rx)
            ratio[:-1] = np.maximum(ratio[:-1], rx)
        if dx.shape[1] > 1:
            rz = dz[:, 1:] / np.maximum(dz[:, :-1], 1e-300)
            rz = np.maximum(rz, 1 / rz)
            ratio[:, 1:] = np.maximum(ratio[:, 1:], rz)
            ratio[:, :-1] = np.maximum(ratio[:, :-1], rz)
        ang = np.full(dx.shape, 180.0)
        corners = [c00, c10, c11, c01]
        for i in range(4):
            p, q, r = corners[i - 1], corners[i], corners[(i + 1) % 4]
            u, w = p - q, r - q
            cos = np.sum(u * w, axis=-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1), 1e-300)
            ang = np.minimum(ang, np.degrees(np.arccos(np.clip(cos, -1, 1))))
        nxc, nzc = dx.shape
        xs = np.stack([c00[..., 0], c10[..., 0], c11[..., 0], c01[..., 0]], axis=-1).reshape(-1, 4)
        zs = np.stack([c00[..., 1], c10[..., 1], c11[..., 1], c01[..., 1]], axis=-1).reshape(-1, 4)
        out["xs"].append(xs)
        out["zs"].append(zs)
        out["block"].append(np.full(nxc * nzc, blk["block"]))
        out["size"].append(np.sqrt(np.abs(area)).ravel())
        out["ar"].append(ar.ravel())
        out["ar3d"].append(ar3d.ravel())
        out["ratio"].append(ratio.ravel())
        out["angle"].append(ang.ravel())
        out["dx"].append(dx.ravel())
        out["dz"].append(dz.ravel())
        out["folded"].append((area <= 0).ravel())
    return {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}


def surface_cells(blocks):
    '''
    Cells along the airfoil surface of the slice: per side (upper, lower) from LE to TE arrays of
    x (cell centre), chordwise length, first wall-normal cell height, edge parameter; plus the largest
    size jump between neighbours. Empty if the airfoil is not in the slice (flow layers).
    '''
    by = {b["block"]: b["P"] for b in blocks}
    if not all(k in by for k in UPPER_LE_TO_TE + LOWER_LE_TO_TE):
        return None
    out = {}
    for side, order, flip in (("upper", UPPER_LE_TO_TE, False), ("lower", LOWER_LE_TO_TE, True)):
        xs, ds, hs, es = [], [], [], []
        for k in order:
            P = by[k][..., [0, 2]]
            wall, row1 = P[:, 0], P[:, 1]
            if flip:
                wall, row1 = wall[::-1], row1[::-1]
            d = np.linalg.norm(np.diff(wall, axis=0), axis=1)
            h = 0.5 * (np.linalg.norm(row1[:-1] - wall[:-1], axis=1) + np.linalg.norm(row1[1:] - wall[1:], axis=1))
            xs.append(0.5 * (wall[:-1, 0] + wall[1:, 0]))
            ds.append(d)
            hs.append(h)
            es += [SURFACE_EDGE[k]] * len(d)
        x, d, h = np.concatenate(xs), np.concatenate(ds), np.concatenate(hs)
        r = d[1:] / d[:-1]
        r = np.maximum(r, 1 / r)
        i = int(np.argmax(r)) if len(r) else 0
        out[side] = {"x": x, "length": d, "first_height": h, "edge": es,
                     "max_jump": float(r[i]) if len(r) else 1.0, "jump_x": float(0.5 * (x[i] + x[i + 1])) if len(r) else 0.0}
    return out


def edge_stats(blocks):
    '''Measured cell sizes per edge parameter in the slice (rows for a table).'''
    by = {b["block"]: b["P"][..., [0, 2]] for b in blocks}
    rows = []

    def add(name, where, lengths):
        if len(lengths):
            rows.append({"edge": name, "where": where, "smallest cell (m)": float(np.min(lengths)),
                         "largest cell (m)": float(np.max(lengths)), "cells": int(len(lengths))})

    surf = surface_cells(blocks)
    if surf:
        h = np.r_[surf["upper"]["first_height"], surf["lower"]["first_height"]]
        rows.append({"edge": "e00", "where": "first cell at the wall (height)", "smallest cell (m)": float(h.min()),
                     "largest cell (m)": float(h.max()), "cells": int(len(h))})
        for k in UPPER_LE_TO_TE[::-1] + LOWER_LE_TO_TE:
            P = by[k]
            add(SURFACE_EDGE[k], "along the airfoil, " + ("upper" if k <= 19 else "lower") + " side",
                np.linalg.norm(np.diff(P[:, 0], axis=0), axis=1))
    for k, name, where in ((14, "e02", "near wake, along the wake line"), (13, "e01", "far wake, along the wake line")):
        if k in by:
            add(name, where, np.linalg.norm(np.diff(by[k][:, 0], axis=0), axis=1))
    if 17 in by:
        P = by[17]
        add("e00", "outermost cell (at the C-boundary)", [float(np.linalg.norm(P[0, -1] - P[0, -2]))])
    return rows


def layer_jumps(case, limit=1.3):
    '''Spanwise size ratio between the last cell of a layer and the first of the next (rows).'''
    if case.dim != "3D":
        return []
    s, res, grad = case.blocks.slice_locations, case.meshing.layer_resolution, case.meshing.layer_grading
    rows = []
    for k in range(len(s) - 2):
        _, last = graded_sizes(s[k + 1] - s[k], res[k], grad[k])
        first, _ = graded_sizes(s[k + 2] - s[k + 1], res[k + 1], grad[k + 1])
        r = max(first / last, last / first)
        rows.append({"slice": k + 1, "y": s[k + 1], "ratio": r, "jump": r > limit})
    return rows


def summary(metrics):
    if not len(metrics["size"]):
        return {}
    return {"cells in the slice": int(len(metrics["size"])), "smallest cell (m)": float(metrics["size"].min()),
            "largest cell (m)": float(metrics["size"].max()), "max aspect ratio (slice)": float(metrics["ar"].max()),
            "max aspect ratio (3D)": float(metrics["ar3d"].max()),
            "max neighbour size ratio": float(metrics["ratio"].max()),
            "smallest corner angle (deg)": float(metrics["angle"].min()),
            "folded cells": int(metrics["folded"].sum())}
