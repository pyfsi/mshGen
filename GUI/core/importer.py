'''
importer.py: create a case from an mshGen example script (examples/generate_mesh_*.py)

The example is executed with a stand-in for mshGen's Mesh class that only records the inputs
(names, files, params, flags); no mesh is generated and nothing is written outside a temporary folder.
The settings come from the CSVs the example would read (DEFAULT=False) or from mshGen's defaults
computed in a temporary folder (DEFAULT=True, as the example would generate them).
'''
import ast
import contextlib
import os
import shutil
import sys
import tempfile
import types

from . import paths
from .case import Case, Airfoil, Wing, Domain, Blocks, Meshing, Options
from . import mshgen_io

MESH_MODULE = "mshGen.meshGeneration.mesh_class"


class _Recorder:
    '''Stand-in for mshGen's Mesh: records the constructor arguments, all methods do nothing.'''
    calls = []

    def __init__(self, names, files, params, flags):
        _Recorder.calls.append((names, files, params, flags))

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@contextlib.contextmanager
def _recording_mesh():
    paths.enable_mshgen_import()
    import mshGen.meshGeneration  # noqa: F401  (real parent packages)
    saved = sys.modules.get(MESH_MODULE)
    stub = types.ModuleType(MESH_MODULE)
    stub.Mesh = _Recorder
    sys.modules[MESH_MODULE] = stub
    _Recorder.calls = []
    try:
        yield _Recorder.calls
    finally:
        if saved is not None:
            sys.modules[MESH_MODULE] = saved
        else:
            del sys.modules[MESH_MODULE]


def read_example_inputs(script):
    '''Run the example with the recording Mesh -> (names, files, params, flags, docstring).'''
    script = os.path.abspath(script)
    src = open(script).read()
    doc = ast.get_docstring(ast.parse(src)) or ""
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="mshgen_import_")
    try:
        os.chdir(tmp)                                   # examples create ./output_files
        with _recording_mesh() as calls:
            exec(compile(src, script, "exec"), {"__file__": script, "__name__": "__main__"})
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    if len(calls) != 1:
        raise RuntimeError("Expected one Mesh(...) call in " + script + ", found " + str(len(calls)))
    names, files, params, flags = calls[0]
    return names, files, params, flags, doc


def case_from_example(script, name, cases_dir=paths.CASES_DIR, overwrite=False):
    '''Create cases_dir/<name> (case.json + airfoil.txt) from an example script and return the Case.'''
    names, files, params, flags, doc = read_example_inputs(script)
    casename, airfoil_name, wing_name = names
    airfoil_file, wing_file, default_folder, _ = files
    dim, cog, slices, layer_res, layer_grad, layer_aileron, mesh_scale, mesh_level, skip_layer = params
    write, prnt, plot, default, split, artificious = flags

    folder = os.path.join(cases_dir, name)
    if os.path.exists(os.path.join(folder, "case.json")) and not overwrite:
        raise FileExistsError("Case already exists: " + folder)

    if layer_aileron is None:
        aileron = None
    elif isinstance(layer_aileron, (list, tuple)):
        aileron = sorted(int(a) for a in layer_aileron)
    else:
        aileron = [int(layer_aileron)]

    # Settings: as the example would use them
    if default:
        interior, exterior, edges = mshgen_io.generate_defaults(
            airfoil_name, airfoil_file, wing_name, wing_file, dim, list(cog), list(slices), list(layer_res),
            list(layer_grad), layer_aileron, mesh_scale, mesh_level, skip_layer)
    else:
        interior, exterior, edges = mshgen_io.read_settings(default_folder, casename)

    rel = lambda f: os.path.relpath(f, paths.MSHGEN_DIR)  # noqa: E731
    case = Case(
        name=name,
        description=doc,
        dim=dim,
        cog=[float(v) for v in cog],
        airfoil=Airfoil(name=airfoil_name, file="airfoil.txt", source=rel(airfoil_file)),
        wing=Wing(name=wing_name,
                  stations=mshgen_io.read_wing(wing_file) if wing_file else None,
                  source=rel(wing_file) if wing_file else ""),
        domain=Domain(mesh_scale=float(mesh_scale), mesh_level=mesh_level),
        blocks=Blocks(slice_locations=[float(s) for s in slices], interior_points=interior, exterior_points=exterior),
        meshing=Meshing(layer_resolution=[int(n) for n in layer_res], layer_grading=[float(g) for g in layer_grad],
                        edges=edges),
        options=Options(split=bool(split), aileron_layers=aileron, artificious=bool(artificious),
                        skip_layer=int(skip_layer)),
        imported_from=rel(os.path.abspath(script)),
    )

    # Are the vertices mshGen's defaults for this geometry? Then they follow later geometry changes.
    from .geometry import mark_vertex_source
    mark_vertex_source(case, airfoil_file)
    if case.wing.stations:
        from .planform import Planform, fit
        p = fit(case.wing.stations)
        if not p.problems() and _same_rows(p.to_stations(), case.wing.stations):
            case.wing.planform = p.to_dict()

    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(airfoil_file, os.path.join(folder, case.airfoil.file))
    case.save(folder)
    return case


def _same_rows(a, b, tol=1e-9):
    return len(a) == len(b) and all(abs(x - y) <= tol * (1 + abs(y)) for r, s in zip(a, b) for x, y in zip(r, s))
