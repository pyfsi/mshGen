'''
planform.py: wing planform (rows of the wing file) from parameters, derived values, 3D sections

Wing file rows: [station, y, chord, beta, dx, dz]  (see README.md and aero_functions.read_wing_file)
  y      spanwise position from the root (m)
  chord  chord (m)
  beta   twist (deg), positive nose up, rotation about the leading edge
  dx     shift of the section upstream (m): negative dx moves it downstream (sweep back)
  dz     vertical shift (m): dihedral
mshGen interpolates linearly between rows; beyond the tip the tip row is used.
'''
from dataclasses import dataclass, asdict
import math

import numpy as np

COLUMNS = ["station", "y", "chord", "beta", "dx", "dz"]


@dataclass
class Planform:
    '''
    Straight-tapered wing with an optional inboard part of constant chord:
      0 <= y <= kink_y      : chord = root_chord, twist = twist_root, no sweep or dihedral
      kink_y <= y <= tip     : chord, twist linear to the tip values, LE sweep and dihedral angle
    '''
    half_span: float = 5.0
    root_chord: float = 1.0
    tip_chord: float = 0.5
    kink_y: float = 0.0
    le_sweep_deg: float = 0.0       # leading-edge sweep outboard of the kink, positive backwards
    twist_root_deg: float = 0.0
    twist_tip_deg: float = 0.0
    dihedral_deg: float = 0.0       # outboard of the kink, positive tip up

    def problems(self):
        p = []
        if self.half_span <= 0:
            p.append("half-span must be positive")
        if self.root_chord <= 0 or self.tip_chord <= 0:
            p.append("chords must be positive")
        if not 0 <= self.kink_y < self.half_span:
            p.append("kink must be between the root and the tip")
        if abs(self.le_sweep_deg) >= 80 or abs(self.dihedral_deg) >= 80:
            p.append("sweep and dihedral must be below 80 deg")
        return p

    def to_stations(self):
        '''Rows [station, y, chord, beta, dx, dz]: root, kink (if any) and tip (linear in between).'''
        if self.problems():
            raise ValueError("; ".join(self.problems()))
        k, t = self.kink_y, self.half_span
        rows = [[0.0, 0.0, self.root_chord, self.twist_root_deg, 0.0, 0.0]]
        if k > 0:
            rows.append([k, k, self.root_chord, self.twist_root_deg, 0.0, 0.0])
        dy = t - k
        rows.append([t, t, self.tip_chord, self.twist_tip_deg,
                     -dy * math.tan(math.radians(self.le_sweep_deg)),
                     dy * math.tan(math.radians(self.dihedral_deg))])
        return [[float(v) for v in r] for r in rows]

    def to_dict(self):
        return asdict(self)


def fit(stations):
    '''Planform parameters that approximate a table (exact for tables made by Planform.to_stations).'''
    st = np.asarray(stations, dtype=float)
    y, c, beta, dx, dz = st[:, 1], st[:, 2], st[:, 3], st[:, 4], st[:, 5]
    tip = y[-1]
    same = np.isclose(c, c[0]) & np.isclose(beta, beta[0]) & np.isclose(dx, 0) & np.isclose(dz, 0)
    k = 0
    while k + 1 < len(y) and same[k + 1] and y[k + 1] < tip:
        k += 1
    kink = float(y[k])
    dy = tip - kink if tip > kink else 1.0
    return Planform(half_span=float(tip), root_chord=float(c[0]), tip_chord=float(c[-1]), kink_y=kink,
                    le_sweep_deg=math.degrees(math.atan2(-(dx[-1] - dx[k]), dy)),
                    twist_root_deg=float(beta[0]), twist_tip_deg=float(beta[-1]),
                    dihedral_deg=math.degrees(math.atan2(dz[-1] - dz[k], dy)))


def stations_problems(stations):
    p = []
    if not stations or len(stations) < 2:
        return ["the planform needs at least two rows (root and tip)"]
    st = np.asarray(stations, dtype=float)
    if st.shape[1] != 6:
        return ["rows need 6 values " + str(COLUMNS)]
    if st[0, 1] != 0.0:
        p.append("the first row must be at y = 0")
    if np.any(np.diff(st[:, 1]) <= 0):
        p.append("y must increase from row to row")
    if np.any(st[:, 2] <= 0):
        p.append("chords must be positive")
    if not np.all(np.isfinite(st)):
        p.append("all values must be numbers")
    return p


def _integral(st, f):
    '''Integral over the half-span of f(rows), the rows linear in y between the table rows (as mshGen interpolates
    them): Simpson's rule per interval, exact for products of up to three linear quantities (chord, y, dx, dz).'''
    dy = np.diff(st[:, 1])
    a, m = f(st), f(0.5 * (st[:-1] + st[1:]))
    return float(np.sum(dy / 6.0 * (a[:-1] + 4.0 * m + a[1:])))


def reference_values(stations, cog=(0.0, 0.0, 0.0)):
    '''
    Reference values of the half wing, exact for the planform rows: the projected area S, the mean aerodynamic chord
    MAC = (1/S) int c^2 dy, its spanwise position (1/S) int c y dy and leading-edge x (1/S) int c x_LE dy (from the
    root LE), and its quarter-chord point in mesh coordinates (x, z): (1/S) int c (quarter chord of the section) dy,
    with the sections placed as mshGen places them (twist about the LE, dx, dz, the COG).
    '''
    st = np.asarray(stations, dtype=float)
    S = _integral(st, lambda r: r[:, 2])

    def quarter(r, i):
        a = np.radians(-r[:, 3])
        if i == 0:
            return r[:, 2] * 0.25 * np.cos(a) - (cog[0] + r[:, 4])
        return r[:, 2] * 0.25 * np.sin(a) - cog[2] + r[:, 5]
    return {"area": S, "mac": _integral(st, lambda r: r[:, 2] ** 2) / S,
            "y_mac": _integral(st, lambda r: r[:, 2] * r[:, 1]) / S,
            "x_le_mac": _integral(st, lambda r: -r[:, 2] * r[:, 4]) / S,
            "quarter_chord": tuple(_integral(st, lambda r, i=i: r[:, 2] * quarter(r, i)) / S for i in (0, 1))}


def describe(stations):
    '''Derived values of the half wing (planform as mshGen builds it, twist and dihedral not projected). Sweep and
    dihedral outboard of the constant-chord part, as the planform parameters (fit).'''
    st = np.asarray(stations, dtype=float)
    y, c = st[:, 1], st[:, 2]
    b2 = y[-1]
    ref = reference_values(st)
    s_half, p = ref["area"], fit(stations)
    return {"half-span (m)": b2, "span (m)": 2 * b2, "root chord (m)": float(c[0]), "tip chord (m)": float(c[-1]),
            "taper ratio": float(c[-1] / c[0]), "area, full wing (m2)": 2 * s_half,
            "aspect ratio": (2 * b2) ** 2 / (2 * s_half) if s_half > 0 else float("nan"),
            "mean aerodynamic chord (m)": ref["mac"], "MAC at y (m)": ref["y_mac"],
            "LE sweep outboard (deg)": p.le_sweep_deg, "dihedral outboard (deg)": p.dihedral_deg,
            "tip twist - root twist (deg)": float(st[-1, 3] - st[0, 3]), "tip height dz (m)": float(st[-1, 5])}


def section_params(stations, y):
    '''(chord, beta, dx, dz) at y, as aero_functions.interpolate_wing_shape (tip values beyond the tip).'''
    st = np.asarray(stations, dtype=float)
    yy = min(abs(y), st[:, 1].max())
    return tuple(float(np.interp(yy, st[:, 1], st[:, i])) for i in (2, 3, 4, 5))


def place(outline, chord, beta, dx, dz, cog):
    '''Normalised outline (N x 2) placed as mshGen does it: rotate by -beta about the LE, scale, shift.'''
    a = math.radians(-beta)
    x, z = outline[:, 0], outline[:, 1]
    xr, zr = math.cos(a) * x - math.sin(a) * z, math.sin(a) * x + math.cos(a) * z
    return chord * xr - (cog[0] + dx), chord * zr - cog[2] + dz


def surface(stations, outline, cog, n_span=None):
    '''3D surface of the half wing: arrays X, Y, Z (stations x outline points).'''
    st = np.asarray(stations, dtype=float)
    ys = st[:, 1] if n_span is None else np.linspace(0, st[-1, 1], n_span)
    X, Y, Z = [], [], []
    for y in ys:
        x, z = place(outline, *section_params(stations, y), cog)
        X.append(x)
        Z.append(z)
        Y.append(np.full_like(x, y))
    return np.array(X), np.array(Y), np.array(Z)
