'''
model.py: run mshGen in memory (no files written) and return its Mesh object

The Mesh object holds exactly what blockMesh gets: vertices (all slices), blocks (vertices, resolution,
12 edge gradings) and curved edges (polyLines along the airfoil splines, C-front arcs). Used by the block
check (vertices only) and the mesh preview (vertices, blocks, edges).
'''
import os
import shutil
import tempfile

from . import mshgen_io, paths

paths.enable_mshgen_import()


def mshgen_mesh(case, airfoil_file, upto="vertices"):
    '''mshGen Mesh object after generate_vertices ("vertices") or also generate_blocks/generate_edges ("edges").'''
    from mshGen.meshGeneration.mesh_class import Mesh
    tmp = tempfile.mkdtemp(prefix="mshgen_model_")
    try:
        settings = os.path.join(tmp, "settings")
        mshgen_io.write_settings(case, settings)
        wing_file = None
        if case.wing_stations:
            wing_file = os.path.join(tmp, "wing.txt")
            mshgen_io.write_wing(case.wing_stations, wing_file)
        names = [case.casename, case.airfoil.name, case.wing.name]
        files = [airfoil_file, wing_file, settings, os.path.join(tmp, "output_files")]
        params = [case.dim, [float(v) for v in case.cog], list(case.blocks.slice_locations),
                  list(case.meshing.layer_resolution), list(case.meshing.layer_grading),
                  case.options.aileron_layers or None, float(case.domain.mesh_scale), case.domain.mesh_level,
                  int(case.options.skip_layer)]
        flags = [False, False, False, False, bool(case.options.split), bool(case.options.artificious)]
        mesh = Mesh(names, files, params, flags)
        mesh.initialize_mesh()
        mesh.generate_vertices()
        if upto == "edges":
            mesh.generate_blocks()
            mesh.generate_edges()
        return mesh
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# Edges per slice in Mesh.generate_edges and the ones written commented out (not used by blockMesh):
# 0-4 suction, 5-8 camber, 9-13 pressure, 14-15 top, 16-17 bottom, 18-19 wake mid-line, 20-23 C-arcs,
# 24-29 face matching. Wing slices: camber, top/bottom, wake mid-line and face matching are commented;
# flow slices: only the wake mid-line.
EDGES_PER_SLICE = 30
COMMENTED_WING = set(range(5, 9)) | set(range(14, 18)) | {18, 19} | set(range(24, 30))
COMMENTED_FLOW = {18, 19}


def active_edges(mesh, slice_index):
    '''Edges blockMesh uses in a slice: {(v0, v1): points (n x 3)} with global vertex indices.'''
    import numpy as np
    n = len(mesh.edges) // mesh.N_slices
    assert n == EDGES_PER_SLICE, "unexpected number of edges per slice: {}".format(n)
    commented = COMMENTED_WING if slice_index <= mesh.N_wing_layers else COMMENTED_FLOW
    out = {}
    for k, e in enumerate(mesh.edges[slice_index * n:(slice_index + 1) * n]):
        if k in commented:
            continue
        v0, v1 = e.vertices[0].index, e.vertices[1].index
        if e.type == "polyLine":
            pts = np.array([np.asarray(e.coord[i], dtype=float) for i in range(3)]).T
        else:                                           # arc through v0, mid point, v1
            pts = _arc_points(np.asarray(e.vertices[0].coord, float), np.asarray(e.coord, float),
                              np.asarray(e.vertices[1].coord, float))
        out[(v0, v1)] = pts
    return out


def _arc_points(p0, pm, p1, n=2049):
    '''Points on the circle through p0, pm, p1 (all at the same y), from p0 over pm to p1.'''
    import numpy as np
    a, b, c = p0[[0, 2]], pm[[0, 2]], p1[[0, 2]]
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uz = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    ctr = np.array([ux, uz])
    r = np.linalg.norm(a - ctr)
    t0, tm, t1 = (np.arctan2(p[1] - ctr[1], p[0] - ctr[0]) for p in (a, b, c))
    # unwrap so that the arc from t0 to t1 passes through tm
    t1u = t0 + (t1 - t0) % (2 * np.pi)
    tmu = t0 + (tm - t0) % (2 * np.pi)
    if tmu > t1u:
        t1u -= 2 * np.pi
    t = np.linspace(t0, t1u, n)
    pts = np.column_stack((ctr[0] + r * np.cos(t), np.full(n, p0[1]), ctr[1] + r * np.sin(t)))
    pts[0], pts[-1] = p0, p1
    return pts
