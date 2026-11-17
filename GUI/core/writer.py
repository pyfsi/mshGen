'''
writer.py: write a case folder from its case.json

Written (always, from case.json):  wing.txt, settings/*.csv, generate_mesh.py, Allrun.pre,
                                    constant/transportProperties, images.json (ParaView views)
Written only if missing (may be edited by the user):  system/controlDict, fvSchemes, fvSolution,
                                    constant/turbulenceProperties, <name>.foam
The airfoil file is part of the case and is never rewritten.
system/blockMeshDict is written by generate_mesh.py, from the lists mshGen actually produced.
'''
import datetime
import os
import shutil
import stat

from . import paths
from .case import Case
from . import mshgen_io

TRANSPORT_PROPERTIES = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              @NU@;

// ************************************************************************* //
'''


def _render(text, values):
    for key, value in values.items():
        text = text.replace("@" + key + "@", value)
    return text


def _template(name):
    with open(os.path.join(paths.TEMPLATES_DIR, name)) as f:
        return f.read()


def _docstring_text(text, indent="    "):
    text = (text or "(none)").replace("\\", "\\\\").replace("'''", '"""')
    return "\n".join(indent + line if line.strip() else "" for line in text.splitlines())


def _write(filename, text, executable=False):
    with open(filename, "w") as f:
        f.write(text)
    if executable:
        os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR | stat.S_IXGRP)


def write_case(folder, case=None):
    '''
    Write all files of the case in folder (case.json is read from folder if case is None,
    otherwise case is saved to folder/case.json first). Returns the list of written files.
    '''
    if case is None:
        case = Case.load(folder)
    else:
        case.save(folder)
    problems = case.validate()
    if problems:
        raise ValueError("Case " + case.name + " is not valid:\n  - " + "\n  - ".join(problems))
    if not os.path.isfile(os.path.join(folder, case.airfoil.file)):
        raise FileNotFoundError("Airfoil file missing: " + os.path.join(folder, case.airfoil.file))

    written = []

    # Wing file
    wing_file = None
    if case.wing_stations:
        wing_file = os.path.join(folder, "wing.txt")
        mshgen_io.write_wing(case.wing_stations, wing_file)
        written.append(wing_file)
    elif os.path.isfile(os.path.join(folder, "wing.txt")):
        os.remove(os.path.join(folder, "wing.txt"))         # 2D: no wing file

    # Settings CSVs
    settings = os.path.join(folder, "settings")
    mshgen_io.write_settings(case, settings)
    written += [mshgen_io.settings_file(settings, k, case.casename) for k in mshgen_io.SETTINGS_KINDS]

    # generate_mesh.py
    relpath = os.path.relpath(paths.MSHGEN_DIR, os.path.abspath(folder))
    values = {
        "NAME": case.name,
        "NAME_LITERAL": repr(case.name),
        "DATE": datetime.date.today().strftime("%d.%m.%Y"),
        "DESCRIPTION": _docstring_text(case.description),
        "MSHGEN_RELPATH": repr(relpath),
        "AIRFOIL_NAME": repr(case.airfoil.name),
        "AIRFOIL_FILE": repr(case.airfoil.file),
        "WING_NAME": repr(case.wing.name),
        "WING_FILE": 'os.path.join(case_dir, "wing.txt")' if wing_file else "None",
        "DIM": repr(case.dim),
        "COG": repr([float(v) for v in case.cog]),
        "MESH_SCALE": repr(float(case.domain.mesh_scale)),
        "MESH_LEVEL": repr(case.domain.mesh_level),
        "SLICE_LOCATIONS": repr([float(v) for v in case.blocks.slice_locations]),
        "LAYER_RESOLUTION": repr([int(v) for v in case.meshing.layer_resolution]),
        "LAYER_GRADING": repr([float(v) for v in case.meshing.layer_grading]),
        "LAYER_AILERON": repr(case.options.aileron_layers if case.options.aileron_layers else None),
        "SKIP_LAYER": repr(int(case.options.skip_layer)),
        "SPLIT": repr(bool(case.options.split)),
        "ARTIFICIOUS": repr(bool(case.options.artificious)),
    }
    script = _render(_template("generate_mesh.py.in"), values)
    _write(os.path.join(folder, "generate_mesh.py"), script)
    written.append(os.path.join(folder, "generate_mesh.py"))

    # Allrun.pre
    env = case.environment
    allrun = _render(_template("Allrun.pre.in"), {
        "NAME": case.name, "PYTHON_MODULE": env.python_module, "OPENFOAM_MODULE": env.openfoam_module,
        "PARAVIEW_MODULE": env.paraview_module, "FLUENT": "yes" if env.fluent else "no",
        "MSHGEN_RELPATH": relpath})
    _write(os.path.join(folder, "Allrun.pre"), allrun, executable=True)
    written.append(os.path.join(folder, "Allrun.pre"))

    # Camera positions and zoom windows of the ParaView images (GUI/tools/visualize_mesh.py)
    from .views import write_spec
    written.append(write_spec(case, folder))

    # OpenFOAM files
    for sub in ("system", "constant"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
        for f in os.listdir(os.path.join(paths.TEMPLATES_DIR, sub)):
            target = os.path.join(folder, sub, f)
            if not os.path.exists(target):
                shutil.copyfile(os.path.join(paths.TEMPLATES_DIR, sub, f), target)
                written.append(target)
    tp = os.path.join(folder, "constant", "transportProperties")
    _write(tp, _render(TRANSPORT_PROPERTIES, {"NU": repr(float(case.flow.nu))}))
    written.append(tp)
    foam = os.path.join(folder, case.name + ".foam")
    if not os.path.exists(foam):
        open(foam, "w").close()
        written.append(foam)

    return written
