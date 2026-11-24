'''
results.py: results of a built mesh read from the OpenFOAM case (ASCII polyMesh, constant/wallDistance)

measured_yplus: y+ of the wall-adjacent cells per wall patch, from the wall distance of the cell centres
(checkMesh -writeFields '(wallDistance)', written by Allrun.pre) and the same flat-plate u_tau with the
local chord as the GUI's prediction (core/sizing.py), so both can be compared directly.
'''
import os
import re

import numpy as np

from .planform import section_params
from .sizing import u_tau


def _strip(text):
    text = re.sub(r"FoamFile\s*\{.*?\}", "", text, count=1, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _numbers(path, key=None):
    t = _strip(open(path).read())
    if key:
        t = t[t.index(key):]
    m = re.search(r"(\d+)\s*\(", t)
    n = int(m.group(1))
    body = t[m.end():t.index(")", m.end())]
    return np.fromstring(body, sep=" ", count=n)


def _points(path):
    t = _strip(open(path).read())
    m = re.search(r"(\d+)\s*\(", t)
    n = int(m.group(1))
    return np.fromstring(t[m.end():].translate(str.maketrans("()", "  ")), sep=" ", count=3 * n).reshape(n, 3)


def patches(folder):
    '''[(name, type, nFaces, startFace)] of all patches in constant/polyMesh/boundary ([] without mesh).'''
    path = os.path.join(folder, "constant", "polyMesh", "boundary")
    if not os.path.isfile(path):
        return []
    b = _strip(open(path).read())
    out = []
    for name, body in re.findall(r"(\w+)\s*\{([^{}]*)\}", b):
        t = re.search(r"\btype\s+(\w+)\s*;", body)
        n = re.search(r"nFaces\s+(\d+)", body)
        s = re.search(r"startFace\s+(\d+)", body)
        if t and n and s:
            out.append((name, t.group(1), int(n.group(1)), int(s.group(1))))
    return out


def wall_patches(folder):
    '''[(name, nFaces, startFace)] of the wall patches.'''
    return [(name, n, s) for name, t, n, s in patches(folder) if t == "wall"]


def wall_surfaces(folder):
    '''Wall patches of the built mesh: {name: (points (m x 3), quads/polygons as point-index lists)}.'''
    pm = os.path.join(folder, "constant", "polyMesh")
    walls = wall_patches(folder)
    if not walls:
        return {}
    pts = _points(os.path.join(pm, "points"))
    first = min(s for _, _, s in walls)
    last = max(s + n for _, n, s in walls)
    faces = _faces(os.path.join(pm, "faces"), first, last - first)
    out = {}
    for name, n, s in walls:
        fl = faces[s - first:s - first + n]
        used = np.unique(np.concatenate([np.asarray(f) for f in fl]))
        index = {int(p): i for i, p in enumerate(used)}
        out[name] = (pts[used], [[index[p] for p in f] for f in fl])
    return out


def fluent_zones(msh_file):
    '''Zones of a Fluent .msh (foamMeshToFluent): rows {zone, type, name, faces}; [] without the file.'''
    import subprocess
    if not os.path.isfile(msh_file):
        return []
    txt = subprocess.run(["grep", "-a", "-E", r"^\((13|39|12) \(", msh_file], capture_output=True, text=True).stdout
    counts = {}
    for zid, a, b in re.findall(r"^\(13 \(([0-9a-f]+) ([0-9a-f]+) ([0-9a-f]+) ", txt, flags=re.M):
        if int(zid, 16) != 0:
            counts[int(zid, 16)] = int(b, 16) - int(a, 16) + 1
    for zid, a, b in re.findall(r"^\(12 \(([0-9a-f]+) ([0-9a-f]+) ([0-9a-f]+) ", txt, flags=re.M):
        if int(zid, 16) != 0:
            counts[int(zid, 16)] = int(b, 16) - int(a, 16) + 1
    rows = []
    for zid, typ, name in re.findall(r"^\(39 \((\d+) ([\w-]+) ([\w.-]+)\)", txt, flags=re.M):
        rows.append({"zone": int(zid), "type": typ, "name": name, "faces or cells": counts.get(int(zid))})
    return rows


def _faces(path, start, count):
    '''
    Point labels of faces start ... start+count-1 (ASCII polyMesh). A face is one line "n(a b c ...)", or, for long
    faces (e.g. the polygons of a merged non-conformal interface), several lines: "n", "(", one label per line, ")".
    '''
    with open(path) as f:
        for line in f:                                  # skip the header up to the face count and "("
            if re.match(r"^\d+\s*$", line):
                break
        next(f)                                         # "("
        out, k = [], 0
        for line in f:
            if k >= start + count:
                break
            if not line.strip():
                continue
            if "(" in line:                             # one-line face
                if k >= start:
                    inner = line[line.index("(") + 1:line.rindex(")")]
                    out.append([int(v) for v in inner.split()])
                k += 1
                continue
            n = int(line)                               # multi-line face: "n", "(", n labels, ")"
            next(f)
            labels = [int(next(f)) for _ in range(n)]
            next(f)
            if k >= start:
                out.append(labels)
            k += 1
    return out


def measured_yplus(folder, case):
    '''Per wall patch: faces, y+ min/median/max (first cell centre), or None without wallDistance.'''
    pm = os.path.join(folder, "constant", "polyMesh")
    wd_file = os.path.join(folder, "constant", "wallDistance")
    if not (os.path.isfile(wd_file) and os.path.isfile(os.path.join(pm, "points"))):
        return None
    patches = wall_patches(folder)
    if not patches:
        return {}
    wd = _numbers(wd_file, "internalField")
    owner = _numbers(os.path.join(pm, "owner")).astype(np.int64)
    pts = _points(os.path.join(pm, "points"))
    first = min(s for _, _, s in patches)
    last = max(s + n for _, n, s in patches)
    faces = _faces(os.path.join(pm, "faces"), first, last - first)
    st = case.wing_stations or [[0, 0, 1.0, 0, 0, 0]]
    out = {}
    for name, n, s in patches:
        ys = np.array([pts[f].mean(axis=0)[1] for f in faces[s - first:s - first + n]])
        ut = np.array([u_tau(case, section_params(st, y)[0]) for y in ys]) if case.dim == "3D" else u_tau(case, 1.0)
        yp = ut * wd[owner[s:s + n]] / case.flow.nu
        out[name] = {"faces": n, "y+ min": float(yp.min()), "y+ median": float(np.median(yp)), "y+ max": float(yp.max())}
    return out
