'''
mshgen_io.py: read and write mshGen's input files (settings CSVs, wing file) and obtain mshGen's defaults

Reading and writing use mshGen's own functions (meshGeneration/aero_functions.py), so the files have
exactly the format mshGen expects.
'''
import os
import shutil
import tempfile

from . import paths

paths.enable_mshgen_import()
import mshGen.meshGeneration.aero_functions as af  # noqa: E402

from .case import EDGE_NAMES, EXTERIOR_POINT_IDS  # noqa: E402

SETTINGS_KINDS = ("interior_points", "exterior_points", "resolution_grading")


def settings_file(folder, kind, casename):
    return os.path.join(folder, "default_" + kind + "_" + casename + ".csv")


# ---------------- settings CSVs ---------------- #
def read_settings(folder, casename):
    '''Read the three settings CSVs of casename from folder -> (interior_points, exterior_points, edges).'''
    missing = [settings_file(folder, k, casename) for k in SETTINGS_KINDS
               if not os.path.isfile(settings_file(folder, k, casename))]
    if missing:
        raise FileNotFoundError("Settings file(s) missing: " + ", ".join(missing))
    pos, spl = af.read_lists_from_file(settings_file(folder, "interior_points", casename), STR_DATA=True)
    interior = [[p, s] for p, s in zip(pos, spl)]
    x, z = af.read_lists_from_file(settings_file(folder, "exterior_points", casename))
    exterior = [[a, b] for a, b in zip(x, z)]
    n, g1, g2 = af.read_more_lists_from_file(settings_file(folder, "resolution_grading", casename))
    if len(n) != len(EDGE_NAMES):
        raise ValueError("Expected " + str(len(EDGE_NAMES)) + " edges in the resolution/grading file, got " + str(len(n)))
    edges = {e: [int(round(a)), b, c] for e, a, b, c in zip(EDGE_NAMES, n, g1, g2)}
    return interior, exterior, edges


def write_settings(case, folder):
    '''Write the three settings CSVs of case to folder (old default_*.csv files in folder are removed).'''
    os.makedirs(folder, exist_ok=True)
    for f in os.listdir(folder):
        if f.startswith("default_") and f.endswith(".csv"):
            os.remove(os.path.join(folder, f))
    b, m, cn = case.blocks, case.meshing, case.casename
    af.write_lists_to_file([p for p, _ in b.interior_points], [s for _, s in b.interior_points],
                           [str(i) for i in range(len(b.interior_points))], ['position', 'spline', 'vertex'],
                           settings_file(folder, "interior_points", cn), STR_DATA=True)
    af.write_lists_to_file([x for x, _ in b.exterior_points], [z for _, z in b.exterior_points],
                           [str(i) for i in EXTERIOR_POINT_IDS], ['x-coord', 'y-coord', 'vertex'],
                           settings_file(folder, "exterior_points", cn), STR_DATA=False)
    af.write_more_lists_to_file([float(m.edges[e][0]) for e in EDGE_NAMES], [m.edges[e][1] for e in EDGE_NAMES],
                                [m.edges[e][2] for e in EDGE_NAMES], EDGE_NAMES,
                                ['resolution', 'grad1', 'grad2', 'edge'],
                                settings_file(folder, "resolution_grading", cn), STR_DATA=False)


# ---------------- wing file ---------------- #
def read_wing(filename):
    '''Wing file -> rows [station, y, chord, beta, dx, dz] (dz = 0 for 5-column files).'''
    data = af.read_wing_file(filename)
    if isinstance(data, int):
        raise FileNotFoundError(filename)
    return [[float(v) for v in row] for row in data]


def write_wing(stations, filename):
    '''Rows [station, y, chord, beta, dx, dz] -> wing file (full precision, read back exactly).'''
    with open(filename, "w") as f:
        for row in stations:
            f.write(" ".join(repr(float(v)) for v in row) + "\n")


# ---------------- mshGen defaults ---------------- #
def generate_defaults(airfoil_name, airfoil_file, wing_name, wing_file, dim, cog, slice_locations,
                      layer_resolution, layer_grading, aileron_layers, mesh_scale, mesh_level, skip_layer=1):
    '''
    Run mshGen's generate_default (DEFAULT=True) in a temporary folder and return its settings
    (interior_points, exterior_points, edges). Nothing is written to meshGeneration/default_files.
    '''
    from mshGen.meshGeneration.mesh_class import Mesh
    tmp = tempfile.mkdtemp(prefix="mshgen_defaults_")
    try:
        casename = (airfoil_name + "_" + dim + "_" + wing_name + "_R" + "{:05.0f}".format(1e2 * mesh_scale)
                    + "C_" + mesh_level)
        names = [casename, airfoil_name, wing_name]
        files = [airfoil_file, wing_file, tmp, os.path.join(tmp, "output_files")]
        params = [dim, cog, slice_locations, layer_resolution, layer_grading, aileron_layers,
                  mesh_scale, mesh_level, skip_layer]
        flags = [True, False, False, True, False, False]   # WRITE, PRINT, PLOT, DEFAULT, SPLIT, ARTIFICIOUS
        mesh = Mesh(names, files, params, flags)
        mesh.initialize_mesh()
        mesh.generate_default()
        return read_settings(tmp, casename)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
