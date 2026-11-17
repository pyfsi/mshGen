'''
geometry.py: changes to the geometry of a case and what follows from them

- Block vertices: mshGen's default vertex positions depend on the airfoil, the root section (chord,
  twist, dx, dz), COG, mesh_scale and mesh_level (Mesh.generate_default only uses the root section).
  defaults_key() describes that input; when it changes, recompute_vertices() asks mshGen for new
  defaults. The edge resolution/grading is kept unless the airfoil, mesh_scale or mesh_level changed
  (only those select mshGen's default resolution).
- Span: when the half-span changes, the slice locations can be scaled with it.
- 2D/3D: a 2D case has slices [0, 1] and one cell; the 3D slices are kept to switch back.
- New wing cases from scratch.
'''
import os
import shutil
import tempfile

from . import airfoils, mshgen_io, paths
from .case import Case, Airfoil, Wing, Domain, Blocks, Meshing, Options
from .planform import Planform


def defaults_key(case, airfoil_file):
    c, beta, dx, dz = case.root_section()
    return {"airfoil": case.airfoil.name, "airfoil_file": airfoils.file_hash(airfoil_file), "dim": case.dim,
            "root": [c, beta, dx, dz], "cog": [float(v) for v in case.cog],
            "mesh_scale": float(case.domain.mesh_scale), "mesh_level": case.domain.mesh_level}


def vertex_state(case, airfoil_file):
    '''"ok" (defaults for this geometry), "outdated" (defaults for another geometry) or "custom".'''
    if case.blocks.vertex_source != "mshgen-defaults":
        return "custom"
    return "ok" if case.blocks.defaults_key == defaults_key(case, airfoil_file) else "outdated"


def edge_defaults_changed(case):
    '''True if airfoil, mesh_scale or mesh_level differ from the last defaults (or there were none).'''
    k = case.blocks.defaults_key
    if not k:
        return True
    return (k.get("airfoil"), k.get("mesh_scale"), k.get("mesh_level")) != (
        case.airfoil.name, float(case.domain.mesh_scale), case.domain.mesh_level)


def check_level(case):
    if not airfoils.has_defaults(case.airfoil.name):
        raise ValueError("mshGen's default block vertices do not work for " + case.airfoil.name
                         + " (crossing blocks); use its example cases, which have their own vertices")
    used, levels, note = airfoils.default_levels(case.airfoil.name, case.domain.mesh_scale)
    if case.domain.mesh_level not in levels:
        raise ValueError("mshGen has no default settings for {} at R = {:g}c with level {} (available: {}; {})".format(
            case.airfoil.name, used if used else case.domain.mesh_scale, case.domain.mesh_level,
            ", ".join(levels), note))


def mshgen_defaults(case, airfoil_file):
    '''mshGen's default (interior_points, exterior_points, edges) for the geometry of case.'''
    check_level(case)
    tmp = tempfile.mkdtemp(prefix="mshgen_wing_")
    try:
        wing_file = None
        if case.wing_stations:
            wing_file = os.path.join(tmp, "wing.txt")
            mshgen_io.write_wing(case.wing_stations, wing_file)
        # Slices and layers do not influence the defaults (generate_default uses the root section only)
        return mshgen_io.generate_defaults(case.airfoil.name, airfoil_file, case.wing.name, wing_file, case.dim,
                                           [float(v) for v in case.cog], [0.0, 1.0], [1], [1.0], None,
                                           float(case.domain.mesh_scale), case.domain.mesh_level)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def recompute_vertices(case, airfoil_file, reset_edges=None):
    '''
    Set the block vertices to mshGen's defaults for the current geometry. reset_edges: also take
    mshGen's default resolution/grading (None: only if airfoil, mesh_scale or mesh_level changed).
    Returns True if the edges were reset.
    '''
    if reset_edges is None:
        reset_edges = edge_defaults_changed(case)
    interior, exterior, edges = mshgen_defaults(case, airfoil_file)
    case.blocks.interior_points, case.blocks.exterior_points = interior, exterior
    if reset_edges:
        case.meshing.edges = edges
    case.blocks.vertex_source = "mshgen-defaults"
    case.blocks.defaults_key = defaults_key(case, airfoil_file)
    return bool(reset_edges)


def mark_vertex_source(case, airfoil_file):
    '''Set vertex_source/defaults_key by comparing the vertices with mshGen's defaults (imported cases).'''
    try:
        interior, exterior, _ = mshgen_defaults(case, airfoil_file)
    except Exception:  # noqa: BLE001  (e.g. no defaults for this airfoil/scale/level)
        case.blocks.vertex_source, case.blocks.defaults_key = "custom", None
        return
    same = interior == case.blocks.interior_points and all(
        abs(a - b) < 1e-9 for p, q in zip(exterior, case.blocks.exterior_points) for a, b in zip(p, q))
    if same and len(exterior) == len(case.blocks.exterior_points):
        case.blocks.vertex_source, case.blocks.defaults_key = "mshgen-defaults", defaults_key(case, airfoil_file)
    else:
        case.blocks.vertex_source, case.blocks.defaults_key = "custom", None


# ---------------- span, dimension ---------------- #
def default_slices(half_span):
    '''Slices, cells and grading for a new 3D wing: refined towards the tip, far field at 2x the half-span.'''
    b = half_span
    return [0.0, 0.8 * b, b, 1.2 * b, 2.0 * b], [20, 15, 15, 20], [1.0, 0.1, 10.0, 2.5]


def set_stations(case, stations, scale_slices=True):
    '''New planform rows; if the half-span changes and scale_slices, scale the slices with it.'''
    old = case.half_span
    case.wing.stations = [[float(v) for v in r] for r in stations]
    new = case.half_span
    if scale_slices and old and new and abs(new - old) > 1e-12:
        f = new / old
        case.blocks.slice_locations = [s * f for s in case.blocks.slice_locations]
        return f
    return None


def set_dim(case, dim, memory):
    '''Switch between 2D and 3D. memory (dict) keeps the 3D slices for switching back.'''
    if dim == case.dim:
        return
    if dim == "2D":
        memory["3D"] = (list(case.blocks.slice_locations), list(case.meshing.layer_resolution),
                        list(case.meshing.layer_grading), case.options.aileron_layers)
        case.blocks.slice_locations, case.meshing.layer_resolution, case.meshing.layer_grading = [0.0, 1.0], [1], [1.0]
        case.options.aileron_layers = None
        case.dim = "2D"
    else:
        case.dim = "3D"
        if not case.wing.stations:
            p = Planform()
            case.wing.stations, case.wing.planform = p.to_stations(), p.to_dict()
        if "3D" in memory:
            s, r, g, a = memory["3D"]
        else:
            (s, r, g), a = default_slices(case.half_span), None
        case.blocks.slice_locations, case.meshing.layer_resolution, case.meshing.layer_grading = s, r, g
        case.options.aileron_layers = a


# ---------------- new case ---------------- #
def new_wing_case(name, airfoil_name, cases_dir=paths.CASES_DIR, planform=None, overwrite=False):
    '''New 3D wing case with an airfoil of the library, a straight-tapered planform and mshGen defaults.'''
    folder = os.path.join(cases_dir, name)
    if os.path.exists(os.path.join(folder, "case.json")) and not overwrite:
        raise FileExistsError("Case already exists: " + folder)
    p = planform or Planform()
    slices, res, grad = default_slices(p.half_span)
    src = airfoils.library_file(airfoil_name)
    case = Case(
        name=name,
        description="New wing case created with the mshGen GUI.",
        dim="3D",
        cog=[0.25 * p.root_chord, 0.0, 0.0],
        airfoil=Airfoil(name=airfoil_name, file="airfoil.txt", source=os.path.relpath(src, paths.MSHGEN_DIR)),
        wing=Wing(name="WING", stations=p.to_stations(), planform=p.to_dict()),
        domain=Domain(mesh_scale=airfoils.default_mesh_scale(airfoil_name), mesh_level="COARSE"),
        blocks=Blocks(slice_locations=slices),
        meshing=Meshing(layer_resolution=res, layer_grading=grad),
        options=Options(),
    )
    recompute_vertices(case, src, reset_edges=True)
    problems = case.validate()
    if problems:
        raise ValueError("New case is not valid: " + "; ".join(problems))
    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(src, os.path.join(folder, case.airfoil.file))
    case.save(folder)
    return case
