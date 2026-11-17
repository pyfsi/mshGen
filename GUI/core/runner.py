'''
runner.py: build a case (run its Allrun.pre) and read the results (checkMesh summary)
'''
import os
import re
import subprocess


def run_allrun(folder, images=False, on_line=None):
    '''
    Run folder/Allrun.pre (optionally with --images). Every output line is passed to on_line
    (default: print). Returns the exit code.
    '''
    on_line = on_line or (lambda line: print(line, flush=True))
    cmd = ["bash", "./Allrun.pre"] + (["--images"] if images else [])
    proc = subprocess.Popen(cmd, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        on_line(line.rstrip("\n"))
    return proc.wait()


_NUM = r"(-?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"


def _number(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def checkmesh_summary(folder):
    '''Main numbers of folder/log.checkMesh as a dict (None if there is no log).'''
    log = os.path.join(folder, "log.checkMesh")
    if not os.path.isfile(log):
        return None
    t = open(log).read()
    failed = _number(r"Failed (\d+) mesh checks", t, int)
    s = {
        "cells": _number(r"\n\s*cells:\s+(\d+)", t, int),
        "points": _number(r"\n\s*points:\s+(\d+)", t, int),
        "hexahedra": _number(r"hexahedra:\s+(\d+)", t, int),
        "polyhedra": _number(r"polyhedra:\s+(\d+)", t, int),
        # "Max aspect ratio = 12.3 OK." or "***High aspect ratio cells found, Max aspect ratio: 7534.7, ..."
        "max_aspect_ratio": _number(r"[Mm]ax aspect ratio\s*[:=]?\s*" + _NUM, t),
        "max_non_orthogonality": _number(r"non-orthogonality Max:\s*" + _NUM, t),
        "avg_non_orthogonality": _number(r"non-orthogonality Max:\s*\S+\s+average:\s*" + _NUM, t),
        "max_skewness": _number(r"Max skewness = " + _NUM, t),
        "min_volume": _number(r"Min volume = " + _NUM, t),
        "failed_checks": failed if failed is not None else 0,
        "mesh_ok": "Mesh OK." in t,
        "warnings": [line.strip().lstrip("*") for line in t.splitlines() if line.strip().startswith("***")],
        "patches": {},
    }
    # Patch table: "   Patch    Faces   Points   Surface topology" followed by one line per patch
    lines = t.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*Patch\s+Faces\s+Points", line):
            for row in lines[i + 1:]:
                m = re.match(r"\s*(\S+)\s+(\d+)\s+(\d+)\s", row)
                if not m:
                    break
                s["patches"][m.group(1)] = int(m.group(2))
            break
    return s
