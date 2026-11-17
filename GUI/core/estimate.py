'''
estimate.py: quick numbers of a case without running blockMesh (cell count, spanwise cell sizes, geometry)
'''
import numpy as np


def edge_cells(case):
    '''Cell counts n0..n10 of the edge parameters e00..e10.'''
    return [int(case.meshing.edges["e{:02d}".format(i)][0]) for i in range(11)]


def cells_per_layer_slice(case):
    '''Cells in one slice of a layer (per spanwise cell): (exterior blocks, interior blocks, blocks 0-2 only).'''
    n0, n1, n2, n3, n4, n5, n6, n7, n8, _, _ = edge_cells(case)
    exterior = 2 * n0 * (n1 + n2 + n3 + n4 + n5 + n6 + n7)        # blocks 13-26 (mesh_class.generate_blocks)
    interior_0_2 = n3 * n3 + 2 * n4 * n3                            # blocks 0-2 (aileron cut-out)
    interior = interior_0_2 + 2 * n5 * (n8 + n7) + 2 * n6 * (n8 + n7) + 2 * n7 * n8  # blocks 0-12
    return exterior, interior, interior_0_2


def layer_types(case):
    '''Per layer: "wing", "aileron" or "flow" (same rule as Mesh.initialize_mesh).'''
    s = case.blocks.slice_locations
    b2 = case.half_span if case.dim == "3D" else None
    aileron = case.options.aileron_layers or []
    types = []
    for k in range(len(s) - 1):
        if b2 is not None and s[k + 1] > b2 * (1 + 1e-3):
            types.append("flow")
        else:
            types.append("aileron" if k in aileron else "wing")
    return types


def cell_count(case):
    '''Total number of cells (as blockMesh will create them).'''
    exterior, interior, interior_0_2 = cells_per_layer_slice(case)
    total = 0
    for ny, t in zip(case.meshing.layer_resolution, layer_types(case)):
        total += ny * (exterior + {"wing": 0, "aileron": interior_0_2, "flow": interior}[t])
    return int(total)


def graded_sizes(length, n, grading):
    '''First and last cell size of an edge of given length, n cells, expansion ratio grading (last/first).'''
    n, g = int(n), float(grading)
    if n == 1 or abs(g - 1) < 1e-12:
        return length / n, length / n
    r = g ** (1.0 / (n - 1))
    first = length * (r - 1) / (r ** n - 1)
    return first, first * g


def spanwise_table(case):
    '''Rows per layer: layer, type, y0, y1, cells, grading, first and last cell size (m).'''
    s = case.blocks.slice_locations
    rows = []
    for k, t in enumerate(layer_types(case)):
        n, g = case.meshing.layer_resolution[k], case.meshing.layer_grading[k]
        first, last = graded_sizes(s[k + 1] - s[k], n, g)
        rows.append({"layer": k, "type": t, "y0": s[k], "y1": s[k + 1], "cells": n, "grading": g,
                     "first cell (m)": first, "last cell (m)": last})
    return rows


def section(case, airfoil_xz, y):
    '''Airfoil (normalised x, z arrays) placed at spanwise station y as mshGen does it (x, z in m).'''
    from .planform import section_params, place
    st = case.wing_stations or [[0, 0, 1.0, 0, 0, 0]]
    x, z = airfoil_xz
    return place(np.column_stack((x, z)), *section_params(st, y), case.cog)
