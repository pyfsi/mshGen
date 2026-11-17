'''
blockcheck.py: are all blocks valid (not inverted or folded) in every slice? Checked before blockMesh.

mshGen generates the vertices of every slice (no files written). A block with 2D corners [a, b, c, d]
becomes the hex (a b b' a' d c c' d') with b' = b in the next slice (mshGen_functions.block), so the
hex is right-handed when the quadrilateral a-b-c-d runs counter-clockwise in the (x, z) plane. Every
corner of every active block must turn counter-clockwise in both slices of its layer; otherwise blockMesh
stops with "inward-pointing faces" or creates folded cells.

Typical cause: the exterior vertices are the same in every slice (LIMITATIONS.md item 7), so sweep,
twist or taper that moves the trailing edge past the near-wake vertex (TE_root + 0.5 c_root) inverts the
near-wake blocks 14/25.
'''
import numpy as np

from .estimate import layer_types

# 2D block corners per layer, as in Mesh.generate_blocks
BLOCKS_2D = [[0, 1, 2, 3], [1, 4, 5, 2], [2, 5, 6, 3],
             [4, 7, 18, 16], [16, 18, 8, 5], [5, 8, 19, 17], [17, 19, 9, 6],
             [7, 10, 20, 18], [18, 20, 12, 8], [8, 12, 21, 19], [19, 21, 11, 9],
             [10, 13, 12, 20], [13, 11, 21, 12],
             [22, 23, 24, 25], [0, 22, 25, 26],
             [1, 0, 26, 27], [4, 1, 27, 28], [7, 4, 28, 29],
             [10, 7, 29, 30], [13, 10, 30, 31], [11, 13, 31, 32], [9, 11, 32, 33],
             [6, 9, 33, 34], [3, 6, 34, 35], [0, 3, 35, 36],
             [22, 0, 36, 37], [23, 22, 37, 38]]
N_PER_SLICE = 39


def block_name(k):
    if k < 13:
        return "block {} (inside the airfoil contour)".format(k)
    names = {13: "far wake, upper", 14: "near wake, upper", 25: "near wake, lower", 26: "far wake, lower"}
    if k in names:
        return "block {} ({})".format(k, names[k])
    if 15 <= k <= 19:
        return "block {} (upper side, segment {} from the TE)".format(k, k - 14)
    return "block {} (lower side, segment {} from the LE)".format(k, k - 19)


def slice_vertices(case, airfoil_file, mesh=None):
    '''Vertex coordinates (N_slices x 39 x 3) as mshGen generates them for this case.'''
    from .model import mshgen_mesh
    mesh = mesh or mshgen_mesh(case, airfoil_file)
    v = np.array([vx.coord for vx in mesh.vertices], dtype=float)
    return v.reshape(len(case.blocks.slice_locations), N_PER_SLICE, 3)


def _corner_turns(xz, quad):
    '''z-component of the cross product at each corner of the quadrilateral (positive: counter-clockwise).'''
    p = xz[quad]
    e1 = np.roll(p, -1, axis=0) - p           # edge to the next corner
    e0 = p - np.roll(p, 1, axis=0)            # edge from the previous corner
    return e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]


def active_blocks(case):
    '''Per layer: indices of the blocks blockMesh builds (as Mesh.generate_blocks comments them out).'''
    out = []
    for t in layer_types(case):
        if t == "flow":
            out.append(list(range(27)))
        elif t == "aileron":
            out.append([0, 1, 2] + list(range(13, 27)))
        else:
            out.append(list(range(13, 27)))
    return out


SKEW_MIN_ANGLE = 10.0     # deg: block corners below/above these are reported as skewed (warning)
SKEW_MAX_ANGLE = 170.0


def corner_angles(xz, quad):
    '''Interior corner angles (deg) of the quadrilateral; meaningful for counter-clockwise, convex quads.'''
    p = xz[quad]
    a = np.roll(p, 1, axis=0) - p
    b = np.roll(p, -1, axis=0) - p
    cos = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def block_status(xz, k):
    '''("ok" | "skewed" | "folded" | "inverted", min corner angle, max corner angle) of block k in one slice.'''
    quad = BLOCKS_2D[k]
    turns = _corner_turns(xz, quad)
    scale = max(np.ptp(xz[quad], axis=0).max(), 1e-12) ** 2
    ang = corner_angles(xz, quad)
    if np.any(turns <= 1e-9 * scale):
        return ("inverted" if np.all(turns <= 0) else "folded"), float(ang.min()), float(ang.max())
    if ang.min() < SKEW_MIN_ANGLE or ang.max() > SKEW_MAX_ANGLE:
        return "skewed", float(ang.min()), float(ang.max())
    return "ok", float(ang.min()), float(ang.max())


def layout(case, airfoil_file, vertices=None):
    '''
    Vertices of all slices and the status of every active block in both slices of its layer:
    {"vertices": array, "active": [[blocks] per layer], "status": {(layer, block, slice): (status, min, max)}}
    '''
    v = slice_vertices(case, airfoil_file) if vertices is None else vertices
    active = active_blocks(case)
    status = {}
    for layer, blocks in enumerate(active):
        for k in blocks:
            for sl in (layer, layer + 1):
                status[(layer, k, sl)] = block_status(v[sl][:, [0, 2]], k)
    return {"vertices": v, "active": active, "status": status}


def _messages(case, lay, kinds):
    s = case.blocks.slice_locations
    out = []
    for (layer, k, sl), (st, amin, amax) in lay["status"].items():
        if st in kinds:
            extra = "" if st in ("folded", "inverted") else " (corner angles {:.0f}-{:.0f} deg)".format(amin, amax)
            msg = "layer {}: {} is {} at slice {} (y = {:.4g} m){}".format(layer, block_name(k), st, sl, s[sl], extra)
            if msg not in out:
                out.append(msg)
    return out


def problems_of(case, lay):
    '''Inverted/folded blocks (blockMesh fails or creates folded cells).'''
    return _messages(case, lay, ("folded", "inverted"))


def warnings_of(case, lay):
    '''Skewed blocks (valid, but poor cells).'''
    return _messages(case, lay, ("skewed",))


def check(case, airfoil_file):
    '''List of problems (empty: all active blocks valid in all slices).'''
    return problems_of(case, layout(case, airfoil_file))


def near_wake_x(case):
    '''x of the near-wake vertex 22 (same in every slice): trailing edges must stay upstream of it.'''
    return float(case.blocks.exterior_points[0][0]) if case.blocks.exterior_points else None
