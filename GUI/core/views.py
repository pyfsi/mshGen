'''
views.py: camera positions and zoom windows for the ParaView images of a case (images.json)

The generic ParaView script GUI/tools/visualize_mesh.py reads images.json (written into the case folder by
writer.py) and renders the images of any case. Everything case-specific is computed here from case.json:
patch names, the root section (LE, TE, chord, first wall cell), R, the tip section and the far field.
'''
import json
import os

import numpy as np

from . import airfoils, preview
from .blocks import radius, tip_slice
from .model import mshgen_mesh
from .planform import place, section_params


def wall_patch_names(case):
    o = case.options
    up, down = (["wingUp"], ["wingDown"]) if o.split else (["wing"], [])
    other = (["wingCut"] if o.aileron_layers else []) + (["extra"] if (o.split and o.artificious) else [])
    tip = ["wingTip"] if case.dim == "3D" and case.half_span and case.blocks.slice_locations[-1] > case.half_span * 1.001 else []
    return {"up": up, "down": down, "tip": tip, "other": other}


def image_spec(case, airfoil_file):
    '''Dictionary for images.json.'''
    st = case.wing_stations or [[0, 0, 1.0, 0, 0, 0]]
    a = airfoils.AirfoilData(case.airfoil.name, airfoil_file) if case.airfoil.name in airfoils.LIBRARY else None
    outline = a.outline() if a else np.loadtxt(airfoil_file)
    x, z = place(outline, *section_params(st, 0.0), case.cog)
    i_le, i_te = int(np.argmin(x)), int(np.argmax(x))
    c0 = float(x.max() - x.min())
    model = mshgen_mesh(case, airfoil_file, upto="edges")
    surf = preview.surface_cells(preview.slice_mesh(case, airfoil_file, 0, model))
    h = float(np.median(np.r_[surf["upper"]["first_height"], surf["lower"]["first_height"]])) if surf else 1e-3 * c0
    spec = {
        "foam": case.name + ".foam", "dim": case.dim, "walls": wall_patch_names(case),
        "root_plane": "symRoot" if case.dim == "3D" else "frontAndBack",
        "R": radius(case), "dz0": section_params(st, 0.0)[3],
        "root": {"le": [float(x[i_le]), float(z[i_le])], "te": [float(x[i_te]), float(z[i_te])],
                 "chord": c0, "first_cell": h},
    }
    if case.dim == "3D":
        b2 = case.half_span
        xt, zt = place(outline, *section_params(st, b2), case.cog)
        ct = float(xt.max() - xt.min())
        xcut = float(xt.min() + 0.45 * ct)
        # upper surface z of the tip section at xcut (largest z of the outline near xcut)
        near = np.abs(xt - xcut) < 0.02 * ct
        z_up = float(zt[near].max()) if near.any() else float(zt.max())
        t = tip_slice(case)
        dy_tip = preview.spanwise_size(case, t) if t is not None else 0.01 * ct
        ys = np.linspace(0, b2, 41)
        le = [place(np.array([[0.0, 0.0]]), *section_params(st, y), case.cog)[0][0] for y in ys]
        te = [place(np.array([[1.0, 0.0]]), *section_params(st, y), case.cog)[0][0] for y in ys]
        spec["span"] = {"half_span": float(b2), "far": float(case.blocks.slice_locations[-1]), "xcut": xcut,
                        "tip_chord": ct, "tip_z_upper": z_up, "tip_z_mid": float(0.5 * (zt.max() + zt.min())),
                        "tip_first_cell": float(dy_tip),
                        "planform_x": [float(min(le)), float(max(te))]}
    return spec


def write_spec(case, folder, airfoil_file=None):
    airfoil_file = airfoil_file or os.path.join(folder, case.airfoil.file)
    path = os.path.join(folder, "images.json")
    with open(path, "w") as f:
        json.dump(image_spec(case, airfoil_file), f, indent=2)
    return path
