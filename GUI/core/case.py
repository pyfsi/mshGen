'''
case.py: definition of a mesh case (all settings of one mesh) and its storage in case.json

A case folder (cases/<name>/) holds case.json plus the airfoil file. Everything else in the folder
(wing.txt, settings/*.csv, generate_mesh.py, Allrun.pre, system/, ...) is written from case.json by
writer.py, so case.json is the single source of truth.

The fields map one-to-one onto the inputs of mshGen's Mesh class (see the example scripts):
    names  = [casename, airfoil.name, wing.name]
    params = [dim, cog, blocks.slice_locations, meshing.layer_resolution, meshing.layer_grading,
              options.aileron_layers, domain.mesh_scale, domain.mesh_level, options.skip_layer]
    flags  = [WRITE, PRINT, PLOT, DEFAULT, options.split, options.artificious]
and the three settings CSVs:
    default_interior_points_<casename>.csv     <- blocks.interior_points
    default_exterior_points_<casename>.csv     <- blocks.exterior_points
    default_resolution_grading_<casename>.csv  <- meshing.edges
'''
import hashlib
import json
import os
import re
from dataclasses import MISSING, dataclass, field, asdict, fields, is_dataclass
from typing import List, Optional

SCHEMA_VERSION = 1
CASE_FILE = "case.json"

# Vertex numbering of mshGen (per slice): 16 airfoil vertices (0-15), 6 helper vertices (16-21),
# 17 exterior vertices (22-38). Edge parameters e00-e10.
N_INTERIOR_POINTS = 16
EXTERIOR_POINT_IDS = list(range(22, 39))
EDGE_NAMES = ["e{:02d}".format(i) for i in range(11)]
EDGE_DESCRIPTION = {
    "e00": "wall-normal (airfoil -> outer boundary)",
    "e01": "far wake, streamwise (near-wake end -> outlet)",
    "e02": "near wake, streamwise (TE -> TE + 0.5c)",
    "e03": "chordwise TE -> vertices 1/3",
    "e04": "chordwise vertices 1/3 -> 4/6",
    "e05": "chordwise vertices 4/6 -> 7/9",
    "e06": "chordwise vertices 7/9 -> 10/11",
    "e07": "chordwise vertices 10/11 -> LE",
    "e08": "across the airfoil thickness (flow layers, aileron cut-out)",
    "e09": "grading only: vertical grading of the wake blocks at the outlet",
    "e10": "grading only: vertical grading of the wake blocks at the near-wake end",
}


@dataclass
class Airfoil:
    name: str = "NACA0012"          # mshGen airfoil name: selects built-in behaviour in mshGen (see LIMITATIONS.md)
    file: str = "airfoil.txt"       # coordinates, relative to the case folder
    source: str = ""                # where the file came from (information only)


@dataclass
class Wing:
    name: str = "WING"              # mshGen wing name: part of the settings file names
    # Planform rows [station, y, chord, beta (deg), dx, dz]; not used in a 2D case (no wing file)
    stations: Optional[List[List[float]]] = None
    source: str = ""
    # Parameters (planform.Planform) if the rows were generated from them, None if edited as a table
    planform: Optional[dict] = None


@dataclass
class Domain:
    mesh_scale: float = 5.0         # radius of the C-grid / root chord
    mesh_level: str = "COARSE"      # refinement level of mshGen's default settings


@dataclass
class Flow:
    U: float = 60.0                 # free-stream velocity (m/s), for y+ estimates
    nu: float = 1e-5                # kinematic viscosity (m2/s), also written to constant/transportProperties


@dataclass
class Blocks:
    slice_locations: List[float] = field(default_factory=lambda: [0.0, 1.0])
    # Airfoil vertices 0-15: [position along the spline (0 = LE, 1 = TE), spline 's'|'c'|'p']
    interior_points: List[list] = field(default_factory=list)
    # Exterior vertices 22-38: [x, z] in m (absolute, origin at COG)
    exterior_points: List[List[float]] = field(default_factory=list)
    # "mshgen-defaults": interior/exterior points are mshGen's defaults for the geometry in defaults_key
    # (recomputed when that geometry changes); "custom": set by hand, never recomputed automatically
    vertex_source: str = "custom"
    defaults_key: Optional[dict] = None


@dataclass
class Meshing:
    layer_resolution: List[int] = field(default_factory=lambda: [1])
    layer_grading: List[float] = field(default_factory=lambda: [1.0])
    # Edge parameters e00-e10: {name: [cells, grad1 (airfoil side), grad2 (outer side)]}
    edges: dict = field(default_factory=dict)
    # Targets of the mesh optimiser (core/meshopt.py), None: the preset of domain.mesh_level
    targets: Optional[dict] = None


@dataclass
class Options:
    split: bool = False                         # wing surface as wingUp / wingDown
    aileron_layers: Optional[List[int]] = None  # layers with aileron cut-out, None = clean wing
    artificious: bool = False                   # first skip_layer wing layers to patch 'extra'
    skip_layer: int = 1


@dataclass
class Simulation:
    '''Test run of step 6 (core/simulation.py); the free-stream velocity is flow.U.'''
    cores: int = 8
    alpha: float = 4.0                      # angle of attack (deg), in the x-z plane
    turbulence: str = "kOmegaSST"           # kOmegaSST | SpalartAllmaras | realizableKE | laminar
    iterations: int = 500
    sections: Optional[List[float]] = None  # spanwise section planes (m); None: 25/50/75/95 % of the half-span


@dataclass
class Environment:
    python_module: str = "Anaconda3-python/2023.09-0"
    openfoam_module: str = "OpenFOAM/v2112-foss-2021b"
    paraview_module: str = "ParaView/5.9.1-foss-2021b-mpi"
    fluent: bool = False            # export fluentInterface/<case>.msh with foamMeshToFluent


@dataclass
class Case:
    name: str = "new_case"
    description: str = ""
    dim: str = "3D"                 # "2D" or "3D"
    cog: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    airfoil: Airfoil = field(default_factory=Airfoil)
    wing: Wing = field(default_factory=Wing)
    domain: Domain = field(default_factory=Domain)
    flow: Flow = field(default_factory=Flow)
    blocks: Blocks = field(default_factory=Blocks)
    meshing: Meshing = field(default_factory=Meshing)
    options: Options = field(default_factory=Options)
    simulation: Simulation = field(default_factory=Simulation)
    environment: Environment = field(default_factory=Environment)
    imported_from: str = ""
    schema_version: int = SCHEMA_VERSION

    # ---------------- derived values ---------------- #
    @property
    def casename(self):
        '''mshGen case name (same formula as the example scripts): names the settings CSVs.'''
        return (self.airfoil.name + "_" + self.dim + "_" + self.wing.name + "_R"
                + "{:05.0f}".format(1e2 * self.domain.mesh_scale) + "C_" + self.domain.mesh_level)

    @property
    def n_layers(self):
        return len(self.blocks.slice_locations) - 1

    @property
    def half_span(self):
        '''Half-span of the wing (None in 2D: mshGen then uses no wing file).'''
        if self.dim != "3D" or not self.wing.stations:
            return None
        return max(row[1] for row in self.wing.stations)

    @property
    def wing_stations(self):
        '''Planform rows mshGen uses: the table in 3D, none in 2D (chord 1, no twist).'''
        return self.wing.stations if self.dim == "3D" else None

    def root_section(self):
        '''(chord, beta, dx, dz) of the root section as mshGen uses it.'''
        st = self.wing_stations
        if not st:
            return 1.0, 0.0, 0.0, 0.0
        r = st[0]
        return float(r[2]), float(r[3]), float(r[4]), float(r[5])

    # ---------------- checks ---------------- #
    def validate(self):
        '''List of problems (empty if the case can be written and meshed).'''
        p = []
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name or ""):
            p.append("the case name may only contain letters, digits, '_', '-' and '.'")
        if self.dim not in ("2D", "3D"):
            p.append("dim must be '2D' or '3D'")
        if len(self.cog) != 3:
            p.append("cog needs 3 values [x, y, z]")
        s = self.blocks.slice_locations
        if len(s) < 2:
            p.append("at least two slice locations are needed")
        elif s[0] != 0.0:
            p.append("the first slice must be at y = 0 (root)")
        if any(b <= a for a, b in zip(s, s[1:])):
            p.append("slice locations must increase")
        for key in ("layer_resolution", "layer_grading"):
            if len(getattr(self.meshing, key)) != self.n_layers:
                p.append(key + " needs one value per layer (" + str(self.n_layers) + ")")
        if any(int(n) != n or n < 1 for n in self.meshing.layer_resolution):
            p.append("layer_resolution must be positive integers")
        if any(g <= 0 for g in self.meshing.layer_grading):
            p.append("layer_grading must be positive")
        if len(self.blocks.interior_points) != N_INTERIOR_POINTS:
            p.append("interior_points needs " + str(N_INTERIOR_POINTS) + " vertices")
        elif any(sp not in ("s", "c", "p") for _, sp in self.blocks.interior_points):
            p.append("interior_points: spline must be 's', 'c' or 'p'")
        if len(self.blocks.exterior_points) != len(EXTERIOR_POINT_IDS):
            p.append("exterior_points needs " + str(len(EXTERIOR_POINT_IDS)) + " vertices")
        missing = [e for e in EDGE_NAMES if e not in self.meshing.edges]
        if missing:
            p.append("meshing.edges misses " + ", ".join(missing))
        else:
            for e in EDGE_NAMES:
                n, g1, g2 = self.meshing.edges[e]
                if int(n) != n or n < 1:
                    p.append(e + ": cell count must be a positive integer")
                if g1 <= 0 or g2 <= 0:
                    p.append(e + ": gradings must be positive")
        if self.dim == "2D":
            if self.n_layers != 1 or self.meshing.layer_resolution[:1] != [1]:
                p.append("a 2D case has one layer with one cell (slices [0, 1], layer_resolution [1])")
        if self.dim == "3D":
            if not self.wing.stations:
                p.append("a 3D case needs a wing planform (wing.stations)")
            else:
                from .planform import stations_problems
                p += ["planform: " + q for q in stations_problems(self.wing.stations)]
                b2 = self.half_span
                if b2 and len(s) >= 2 and s[-1] > b2 * (1 + 1e-3):
                    if not any(abs(y - b2) <= 1e-3 * b2 for y in s):
                        p.append("there are slices beyond the tip, so one slice must be at the tip (y = "
                                 + repr(b2) + ")")
        a = self.options.aileron_layers
        if a:
            if a != list(range(a[0], a[-1] + 1)):
                p.append("aileron_layers must be consecutive")
        return p

    # ---------------- storage ---------------- #
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, CASE_FILE), "w") as f:
            f.write(_dumps(self.to_dict()) + "\n")

    @classmethod
    def load(cls, folder):
        with open(os.path.join(folder, CASE_FILE)) as f:
            d = json.load(f)
        if d.get("schema_version", SCHEMA_VERSION) > SCHEMA_VERSION:
            raise ValueError("case.json was written by a newer GUI (schema " + str(d["schema_version"]) + ")")
        return cls.from_dict(d)


# ---------------------------------------------------------------------------------------------- #
# The settings a mesh was built with
# ---------------------------------------------------------------------------------------------- #

BUILD_FILE = "build.json"                          # written by the GUI after a successful build
MESH_FREE = ("name", "description", "simulation", "environment")     # settings that do not change the mesh


def mesh_key(case):
    '''Fingerprint of the settings that make the mesh (all but MESH_FREE).'''
    d = {k: v for k, v in case.to_dict().items() if k not in MESH_FREE}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


def save_build_key(folder, key):
    with open(os.path.join(folder, BUILD_FILE), "w") as f:
        json.dump({"mesh settings": key}, f)


def build_key(folder):
    '''The mesh_key of the last build of the mesh in folder, or None if unknown (no build by the GUI, or the mesh
    was built again afterwards, e.g. with Allrun.pre).'''
    p, owner = os.path.join(folder, BUILD_FILE), os.path.join(folder, "constant", "polyMesh", "owner")
    if not (os.path.isfile(p) and os.path.isfile(owner)) or os.path.getmtime(p) < os.path.getmtime(owner):
        return None
    try:
        with open(p) as f:
            return json.load(f).get("mesh settings")
    except (OSError, ValueError):
        return None


def _dumps(obj, indent=0):
    '''JSON with lists of numbers/strings on one line (readable tables), everything else indented.'''
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [inner + json.dumps(k) + ": " + _dumps(v, indent + 1) for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            return json.dumps(obj)
        return "[\n" + ",\n".join(inner + _dumps(v, indent + 1) for v in obj) + "\n" + pad + "]"
    return json.dumps(obj)


def _from_dict(cls, d):
    '''Build a (nested) dataclass from a dict; unknown keys are ignored, missing keys get defaults.'''
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value)
        kwargs[f.name] = value
    return cls(**kwargs)
