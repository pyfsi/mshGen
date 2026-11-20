'''
airfoils.py: the airfoils the GUI offers, and what mshGen does for each of them

mshGen selects built-in behaviour by airfoil *name* (LIMITATIONS.md item 3), so the GUI only offers the
name/file pairs that the example scripts use. New airfoils need changes in mshGen (LIMITATIONS.md item 18).
'''
import hashlib
import os

import numpy as np
from scipy.interpolate import splev

from . import paths

paths.enable_mshgen_import()
import mshGen.meshGeneration.aero_functions as af  # noqa: E402

FMC = ["FINE", "MEDIUM", "COARSE"]

# name -> file in meshGeneration/airfoils, description, examples using it
LIBRARY = {
    "Mrev-v2": dict(file="Mrev-v2.txt", description="MegAWES wing airfoil",
                    examples=["MegAWES_WING*", "MREV2_AIRFOIL"]),
    "Mrev-v2_aileron": dict(file="Mrev-v2_aileron.txt", description="MegAWES airfoil, aileron sections",
                            examples=["MegAWES_AILERON_IN", "MegAWES_AILERON_OUT"],
                            # mshGen's generic default exterior vertices cross for this airfoil (blockMesh:
                            # inward-pointing faces); its examples use hand-made settings files
                            defaults=False),
    "NACA0012": dict(file="NACA0012.txt", description="symmetric, 12 % thick (tail surfaces)",
                     examples=["NACA0012_AIRFOIL", "NACA0012_ELEVATOR", "NACA0012_RUDDER_*"]),
    "NACA2420": dict(file="NACA2420.txt", description="2 % camber, 20 % thick",
                     examples=["NACA2420_RIGHT_WING"]),
    "FFA_W3_211": dict(file="FFA_W3_211.txt", description="wind-turbine airfoil, 21 % thick",
                       examples=["FFAW3_AIRFOIL"]),
}

# Tabulated default resolution/grading (cmesh_default_parameters.py): scale R/c -> levels.
# Other airfoils use one generic set (tuned for R = 10c) for every scale.
DEFAULT_TABLES = {
    "NACA0012": {20.0: FMC, 10.0: FMC, 5.0: FMC, 3.0: FMC, 1.5: FMC, 1.0: ["COARSE"]},
    "Mrev-v2": {10.0: FMC, 5.0: FMC, 3.0: FMC},
}


def names():
    return list(LIBRARY)


def has_defaults(name):
    '''False if mshGen's default block vertices are known not to work for this airfoil.'''
    return LIBRARY.get(name, {}).get("defaults", True)


def names_with_defaults():
    '''Airfoils that can be used for new wings or chosen in step 1 (mshGen's defaults work).'''
    return [n for n in LIBRARY if has_defaults(n)]


def library_file(name):
    return os.path.join(paths.AIRFOILS_DIR, LIBRARY[name]["file"])


def file_hash(filename):
    with open(filename, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def default_levels(name, mesh_scale):
    '''
    (scale used by mshGen's defaults, available levels, note). For NACA0012 and Mrev-v2 mshGen takes
    the nearest tabulated scale; other airfoils use the generic set whatever the scale.
    '''
    table = DEFAULT_TABLES.get(name)
    if table is None:
        return None, FMC, "generic defaults (tuned for R = 10c), used for any scale"
    scales = np.array(sorted(table))
    used = float(scales[np.argmin(np.abs(scales - (mesh_scale + 1e-6)))])
    note = "tabulated scales " + ", ".join("{:g}".format(s) for s in sorted(table))
    if abs(used - mesh_scale) > 1e-9:
        note += "; mshGen uses the nearest one, R = {:g}c".format(used)
    return used, table[used], note


def default_mesh_scale(name):
    return 5.0 if name in DEFAULT_TABLES else 10.0


def support(name):
    '''Rows (topic, what mshGen does) for the airfoil name.'''
    removal = {"Mrev-v2_aileron": "duplicate leading edge point is removed",
               "NACA0012": "duplicate leading edge point is removed"}.get(name, "none (duplicate points only)")
    order = "first half of the points is reversed" if name in ("NACA0012", "Mrev-v2_aileron") else "as in the file"
    camber = ("spline through built-in control points" if name in ("Mrev-v2", "FFA_W3_211", "AP2")
              else "straight line from LE to TE")
    interior = ("built-in positions for " + name
                if name in ("NACA0012", "Mrev-v2", "NACA2420", "NACA0020", "AP2", "AP2_aileron", "AP2_H_tail",
                            "AP2_V_tail", "AP2_elevator", "AP2_rudder") else "generic positions")
    exterior = ("built-in corrections for " + name
                if name in ("NACA0012", "NACA0020", "NACA2420", "Mrev-v2", "FFA_W3_211", "AP2", "AP2_aileron",
                            "AP2_elevator", "AP2_rudder", "AP2_H_tail", "AP2_V_tail")
                else "none (normals of the airfoil to the outer boundary)")
    table = DEFAULT_TABLES.get(name)
    resolution = ("tabulated for R/c = " + ", ".join("{:g}".format(s) for s in sorted(table)) if table
                  else "generic set tuned for R = 10c")
    return [("point clean-up", removal), ("point order", order), ("camber line", camber),
            ("default airfoil vertices", interior), ("default exterior vertices", exterior),
            ("default resolution/grading", resolution)]


class AirfoilData:
    '''Airfoil as mshGen reads it: raw points, removed point, splines (normalised chord).'''

    def __init__(self, name, filename, n=401):
        raw = af.RawData(name, filename)
        self.name, self.filename = name, filename
        self.file_points = np.loadtxt(filename)
        self.points = np.asarray(raw.coordinates)
        self.removed = raw.corrupted_point          # removed duplicate point (None if there was none)
        u = np.linspace(0, 1, n)
        self.upper = np.array(splev(u, raw.tck_upper)).T
        self.lower = np.array(splev(u, raw.tck_lower)).T
        self.camber = np.array(splev(u, raw.tck_camber)).T
        self.le = self.points[raw.n]
        self.te = self.points[raw.m]
        thick = np.interp(self.upper[:, 0], self.lower[:, 0], self.lower[:, 1])
        self.max_thickness = float(np.max(self.upper[:, 1] - thick))
        cam = 0.5 * (self.upper[:, 1] + thick)
        self.max_camber = float(np.max(np.abs(cam)))

    def outline(self):
        '''Closed outline TE -> upper -> LE -> lower -> TE (normalised), for plotting/sections.'''
        return np.vstack((self.upper[::-1], self.lower[1:]))
