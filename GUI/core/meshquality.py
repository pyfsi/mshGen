'''
meshquality.py: where the built mesh fails checkMesh or has lower-quality cells (step 5)

Reads from the OpenFOAM case (written by Allrun.pre, or by run_quality for an older build):
  - the sets checkMesh writes for flagged cells/faces (constant/polyMesh/sets: highAspectRatioCells, nonOrthoFaces,
    skewFaces, wrongOrientedFaces, ...), and from log.checkMesh whether that check failed (***) or warned (*);
  - the cell fields of checkMesh -writeFields (nonOrthoAngle, skewness: per cell the worst of its faces);
  - the cell centres of postProcess -func writeCellCentres (constant/C).
Groups: every set, plus the lower-quality cells non-orthogonality 50-70 deg and skewness 2-4 (below checkMesh's
limits 70 and 4). For plotting, at most MAX_POINTS points per group are kept (every k-th).
'''
import os
import re

import numpy as np

from .results import _numbers, _points

MAX_POINTS = 4000
LOWER = [("nonOrthoAngle", 50.0, 70.0, "non-orthogonality 50-70 deg", "deg"),
         ("skewness", 2.0, 4.0, "skewness 2-4", "")]
SET_TEXT = {
    "highAspectRatioCells": "aspect ratio above 1000",
    "nonOrthoFaces": "non-orthogonality above 70 deg",
    "skewFaces": "skewness above 4",
    "wrongOrientedFaces": "faces pointing the wrong way",
    "wrongOrientedTriangleFaces": "triangles pointing the wrong way",
    "concaveCells": "concave cells",
    "concaveFaces": "concave faces",
    "zeroAreaFaces": "faces with zero area",
    "zeroVolumeCells": "cells with zero or negative volume",
    "underdeterminedCells": "under-determined cells",
    "lowQualityTetFaces": "faces with a low-quality tetrahedral decomposition",
    "lowWeightFaces": "faces with a low interpolation weight",
    "lowVolRatioFaces": "faces between cells of very different volume",
    "warpedFaces": "warped faces",
    "nonClosedCells": "cells that are not closed",
    "highAspectRatioFaces": "faces with a high aspect ratio",
}


def available(folder):
    '''True if the cell centres and the quality fields of the last build are there.'''
    c = os.path.join(folder, "constant")
    return all(os.path.isfile(os.path.join(c, f)) for f in ("C", "nonOrthoAngle", "skewness"))


def stamp(folder):
    '''Changes when the mesh or its analysis changes (cache key).'''
    out = []
    for f in ("log.checkMesh", os.path.join("constant", "C"), os.path.join("constant", "nonOrthoAngle")):
        p = os.path.join(folder, f)
        out.append(os.path.getmtime(p) if os.path.isfile(p) else 0.0)
    return tuple(out)


def _vectors(path):
    '''The internalField of a volVectorField (constant/C) as an (n x 3) array.'''
    return _points(path)


def _set_status(log):
    '''{set name: "failed" | "warning"} from log.checkMesh: a set written after a *** line failed the check.'''
    out = {}
    if not os.path.isfile(log):
        return out
    lines = open(log).read().splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"to set (\w+)", line)
        if not m:
            continue
        status = "warning"
        for prev in reversed(lines[max(0, i - 4):i]):
            s = prev.strip()
            if s.startswith("***"):
                status = "failed"
                break
            if s.startswith("*"):
                break
        out[m.group(1)] = status
    return out


def _ranges(pos):
    return {k + "_range": (float(pos[:, i].min()), float(pos[:, i].max())) for i, k in enumerate("xyz")}


def _thin(idx):
    k = max(1, int(np.ceil(len(idx) / MAX_POINTS)))
    return idx[::k]


def analyse(folder):
    '''
    Groups of flagged and lower-quality cells: list of dicts {name, criterion, status ("failed" | "warning" | "lower
    quality"), kind ("cells" | "faces"), count, x, y, z (plotted points), value (or None), y_range}. None if the cell
    centres or fields are missing (build again, or run_quality).
    '''
    if not available(folder):
        return None
    const = os.path.join(folder, "constant")
    C = _vectors(os.path.join(const, "C"))
    groups = []
    status = _set_status(os.path.join(folder, "log.checkMesh"))
    sets_dir = os.path.join(const, "polyMesh", "sets")
    owner = neighbour = None
    for name in sorted(os.listdir(sets_dir)) if os.path.isdir(sets_dir) else []:
        path = os.path.join(sets_dir, name)
        head = open(path).read(2000)
        kind = "cells" if "cellSet" in head else "faces" if "faceSet" in head else None
        if kind is None:
            continue
        try:
            ids = _numbers(path).astype(np.int64)
        except (ValueError, AttributeError):
            ids = np.zeros(0, dtype=np.int64)
        if not len(ids):
            continue
        if kind == "cells":
            pos = C[ids]
        else:
            if owner is None:
                owner = _numbers(os.path.join(const, "polyMesh", "owner")).astype(np.int64)
                neighbour = _numbers(os.path.join(const, "polyMesh", "neighbour")).astype(np.int64)
            o = C[owner[ids]]
            inner = ids < len(neighbour)
            n = o.copy()
            n[inner] = C[neighbour[ids[inner]]]
            pos = 0.5 * (o + n)
        sel = _thin(np.arange(len(ids)))
        groups.append({"name": name, "criterion": SET_TEXT.get(name, name), "status": status.get(name, "warning"),
                       "kind": kind, "count": int(len(ids)), "x": pos[sel, 0], "y": pos[sel, 1], "z": pos[sel, 2],
                       "value": None, **_ranges(pos)})
    for field, lo, hi, text, unit in LOWER:
        v = _numbers(os.path.join(const, field), "internalField")
        idx = np.where((v >= lo) & (v < hi))[0]
        if not len(idx):
            continue
        order = idx[np.argsort(-v[idx])]                   # the worst first, then every k-th
        sel = _thin(order)
        groups.append({"name": field, "criterion": text, "status": "lower quality", "kind": "cells",
                       "count": int(len(idx)), "x": C[sel, 0], "y": C[sel, 1], "z": C[sel, 2], "value": v[sel],
                       "unit": unit, **_ranges(C[idx])})
    return groups


def summary_rows(groups, xyz=False):
    '''Rows for a table; xyz: the x, y and z ranges (the AI summary), else the y range only.'''
    rows = []
    for g in groups:
        r = {"group": g["name"], "criterion": g["criterion"], "status": g["status"],
             "count": "{:,} {}".format(g["count"], g["kind"])}
        for k in ("xyz" if xyz else "y"):
            r[k + " range (m)"] = "{:.3g} - {:.3g}".format(*g[k + "_range"])
        rows.append(r)
    return rows
