'''
step6_simulation.py: step 6 of the GUI, a test run with OpenFOAM on the built mesh (core/simulation.py)

Inputs:   cores, free-stream velocity V (the case's flow.U), angle of attack, turbulence model, iterations, spanwise
          section planes; the other settings of the set-up are listed in a table
Run:      saves the case, writes <case>/run/ and runs Allrun (decomposePar, mpirun simpleFoam, reconstructPar) in the
          background; Stop asks simpleFoam to write the current iteration and end
Results (updated while the solver runs): convergence (residuals, force coefficients), y+ of the solution next to the
          prediction, pressure distribution at a section and on the wing, contours of pressure and velocity at the
          section, the tip vortex in crossflow planes, and an AI review of the solution (Claude through Claude Code)
'''
import datetime
import io
import json
import math
import os
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, Range1d, Span
from bokeh.palettes import Category10
from bokeh.plotting import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.tri import LinearTriInterpolator, Triangulation

from core import aisummary, meshquality
from core import simulation as sm
from core.planform import place, section_params
from core.results import measured_yplus
from ui.common import bokeh, note

_RUNS = {}                  # run folder -> {"proc", "thread", "lines", "code"} (shared by all browser sessions)
_RUNS_LOCK = threading.Lock()
RES_FIELDS = ["p", "Ux", "Uy", "Uz", "k", "omega", "epsilon", "nuTilda"]
RES_COLORS = dict(zip(RES_FIELDS, Category10[8]))
DIVIDER = {"margin": (18, 0, 6, 0)}
PARSE_EVERY = 2.0           # s between reading the log and the force coefficients
RUN_LABEL = "Save and run (Allrun)"
RUN_ANYWAY = "Run anyway (machine busy)"
CONFIRM_FOR = 60.0          # s: a second click on Run within this time starts the run on a busy machine


def section(title):
    return [pn.layout.Divider(**DIVIDER), pn.pane.Markdown("## " + title)]


def _procs(busy):
    n = int(round(busy))
    return "{} busy process{}".format(n, "" if n == 1 else "es")


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return buf.getvalue()


def _tri(u, v, box, max_edge, outline=None):
    '''Triangulation of scattered face centres in a box, without long triangles and triangles inside the outline.'''
    sel = (u > box[0]) & (u < box[1]) & (v > box[2]) & (v < box[3])
    idx = np.where(sel)[0]
    if len(idx) < 10:
        return None, idx
    tri = Triangulation(u[idx], v[idx])
    t = tri.triangles
    pu, pv = u[idx][t], v[idx][t]
    edge = np.max(np.hypot(pu - np.roll(pu, 1, axis=1), pv - np.roll(pv, 1, axis=1)), axis=1)
    mask = edge > max_edge
    if outline is not None:
        mask |= Path(outline).contains_points(np.column_stack((pu.mean(axis=1), pv.mean(axis=1))))
    tri.set_mask(mask)
    return tri, idx


def render_section(plane, outline, V, y, chord):
    '''PNG: Cp and |U|/V in the section plane around the airfoil (outline: (x, z) of the section).'''
    if not plane or "p" not in plane:
        return None
    x, z = plane["xyz"][:, 0], plane["xyz"][:, 2]
    xo, zo = outline[:, 0], outline[:, 1]
    box = (xo.min() - 0.5 * chord, xo.max() + 1.0 * chord, zo.mean() - 0.55 * chord, zo.mean() + 0.55 * chord)
    tri, idx = _tri(x, z, box, 0.25 * chord, outline)
    if tri is None:
        return None
    cp = plane["p"][idx] / (0.5 * V ** 2)
    um = np.linalg.norm(plane["U"][idx], axis=1) / V if "U" in plane else None
    fig = Figure(figsize=(13, 4.2))
    FigureCanvasAgg(fig)
    for k, (val, title, cmap) in enumerate(((cp, "pressure coefficient Cp", "coolwarm"),
                                            (um, "velocity magnitude |U| / V", "viridis"))):
        ax = fig.add_subplot(1, 2, k + 1)
        if val is None:
            continue
        lo, hi = np.percentile(val, [1, 99])
        cs = ax.tricontourf(tri, val, levels=np.linspace(lo, hi, 25), cmap=cmap, extend="both")
        ax.fill(xo, zo, color="#bbbbbb", zorder=3)
        ax.plot(xo, zo, color="black", lw=0.6, zorder=4)
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_title("{} at y = {:.4g} m".format(title, y), fontsize=10)
        fig.colorbar(cs, ax=ax, shrink=0.85, format="%.2f")
    fig.tight_layout()
    return _png(fig)


def render_top(case, wall, V, flow_dir, sections):
    '''PNG: Cp on the upper and the lower side of the wing seen from above (flow downwards); black dots: reversed flow
    at the wall (the wall shear against the flow); dashed: the section planes. wall: sm.wall_surface.'''
    if not wall or "p" not in wall or not case.wing_stations:
        return None
    xyz = wall["xyz"]
    b2 = case.half_span or float(xyz[:, 1].max())
    keep, s, upper = ~wall["side"], wall["s"], wall["upper"]            # without the tip cap
    cp = wall["p"] / (0.5 * V ** 2)
    rev = sm.reversed_flow(s, -(np.asarray(wall["wallShearStress"]) @ np.asarray(flow_dir))) \
        if "wallShearStress" in wall else np.zeros(len(s), bool)
    lo, hi = float(np.percentile(cp[keep], 2)), min(1.0, float(np.percentile(cp[keep], 99.5)))
    levels = np.linspace(lo, hi, 25)
    ys = np.linspace(0.0, b2, 201)                          # leading and trailing edge: triangles outside are masked
    sec = [[float(v) for v in sm.mesh_sections(case, y)] for y in ys]     # the wing as the mesh has it
    le = np.array([place(np.array([[0.0, 0.0]]), *p, case.cog)[0][0] for p in sec])
    te = np.array([place(np.array([[1.0, 0.0]]), *p, case.cog)[0][0] for p in sec])
    fig = Figure(figsize=(13, 6.6))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    cs = None
    for ax, (side, mask) in zip(axes, (("upper side", upper), ("lower side, seen from above", ~upper))):
        m = keep & mask
        if m.sum() < 10:
            continue
        y, x = xyz[m, 1], xyz[m, 0]
        tri = Triangulation(y, x)
        cy, cx = y[tri.triangles].mean(axis=1), x[tri.triangles].mean(axis=1)
        tol = 0.02 * np.interp(cy, ys, np.abs(te - le))
        tri.set_mask((cx < np.interp(cy, ys, np.minimum(le, te)) - tol) | (cx > np.interp(cy, ys, np.maximum(le, te)) + tol))
        cs = ax.tricontourf(tri, cp[m], levels=levels, cmap="coolwarm", extend="both")
        if rev[m].any():
            ax.scatter(y[rev[m]], x[rev[m]], s=2, color="black", lw=0)
        for ysec in sections:
            ax.axvline(ysec, color="black", ls="--", lw=0.7)
        ax.set_xlim(0.0, b2)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_ylabel("x (m)")
        ax.set_title("Cp, {} (black dots: reversed flow at the wall, x/c > {:g}; dashed: the section planes)".format(
            side, sm.NOSE), fontsize=10)
    axes[-1].set_xlabel("y (m)")
    if cs is not None:
        fig.colorbar(cs, ax=list(axes), shrink=0.9, format="%.2f", label="Cp")
    return _png(fig)


def render_wake(plane, V, c_tip, tip_y, tip_z, x, label):
    '''PNG: streamwise vorticity (omega_x c_tip / V) with the in-plane velocity in a crossflow plane near the tip.'''
    if not plane or "vorticity" not in plane:
        return None
    y, z = plane["xyz"][:, 1], plane["xyz"][:, 2]
    w = np.asarray(plane["vorticity"])[:, 0] * c_tip / V
    near = np.abs(y - tip_y) < 2 * c_tip
    zc = float(z[near][np.argmax(np.abs(w[near]))]) if near.any() else tip_z
    box = (tip_y - 1.6 * c_tip, tip_y + 1.0 * c_tip, zc - 1.0 * c_tip, zc + 1.0 * c_tip)
    tri, idx = _tri(y, z, box, 1.0 * c_tip)         # no body in the plane: only very long triangles are masked
    if tri is None:
        return None
    val = w[idx]
    lim = max(float(np.percentile(np.abs(val), 99.5)), 1e-6)
    fig = Figure(figsize=(6.4, 4.6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    cs = ax.tricontourf(tri, val, levels=np.linspace(-lim, lim, 25), cmap="RdBu_r", extend="both")
    if "U" in plane:
        U = np.asarray(plane["U"])[idx]
        gy, gz = np.meshgrid(np.linspace(box[0], box[1], 22), np.linspace(box[2], box[3], 17))
        vy = LinearTriInterpolator(tri, U[:, 1])(gy, gz)
        vz = LinearTriInterpolator(tri, U[:, 2])(gy, gz)
        ax.quiver(gy, gz, vy, vz, color="black", alpha=0.6, width=0.003)
    ax.axvline(tip_y, color="black", ls="--", lw=0.8)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("tip vortex at x = {:.4g} m, {}\nomega_x c_tip / V and the in-plane velocity (dashed: the tip)"
                 .format(x, label), fontsize=10)
    fig.colorbar(cs, ax=ax, shrink=0.85, format="%.2f")
    fig.tight_layout()
    return _png(fig)


class SimulationStep:
    title = "6. Test run"
    intro = 'A short steady RANS run (OpenFOAM simpleFoam) on the mesh; the results update while it runs.'

    def __init__(self, app):
        self.app = app
        self._updating = False
        n = self.ncpu = os.cpu_count() or 1
        self.cores = pn.widgets.IntSlider(name="Cores (this machine has {})".format(n), start=1, end=n, value=min(8, n),
                                          width=320)
        self.load_box = pn.Column(sizing_mode="stretch_width")       # busy processes now, warning if oversubscribed
        self._load_samples = deque(maxlen=8)                         # procs_running, one per poll (500 ms)
        self._load_text = None
        self._overload_ok = 0.0                                      # time of the first click on a busy machine
        self.V = pn.widgets.FloatInput(name="Free-stream velocity V (m/s)", start=0.1, step=1.0, width=240)
        self.alpha = pn.widgets.FloatInput(name="Angle of attack alpha (deg)", start=-30, end=30, step=0.5, width=240)
        self.model = pn.widgets.Select(name="Turbulence model", options={v: k for k, v in sm.TURBULENCE.items()},
                                       width=240)
        self.iters = pn.widgets.IntInput(name="Iterations", start=10, step=50, width=240)
        self.sections_in = pn.widgets.TextInput(name="Section planes y (m), comma separated (empty: 25/50/75/95 % "
                                                     "of the half-span)", width=500)
        for w in (self.cores, self.V, self.alpha, self.model, self.iters, self.sections_in):
            w.param.watch(self.on_input, "value")
        self.defaults = pn.Column(sizing_mode="stretch_width")
        self.run_btn = pn.widgets.Button(name=RUN_LABEL, button_type="primary", width=220)
        self.stop_btn = pn.widgets.Button(name="Stop (write and end)", button_type="light", width=220, disabled=True)
        self.run_btn.on_click(lambda e: self.on_run())
        self.stop_btn.on_click(lambda e: self.on_stop())
        self.progress = pn.pane.Markdown()
        self.log = pn.pane.Str("", styles={"font-size": "12px"})
        self.log_box = pn.Column(self.log, height=260, scroll=True, styles={"border": "1px solid #ccc", "padding": "4px"})
        # convergence (live)
        self.res_src = {f: ColumnDataSource(dict(it=[], r=[])) for f in RES_FIELDS}
        self.force_src = ColumnDataSource(dict(it=[], Cl=[], Cd=[], Cm=[]))
        self.conv = pn.Row(*self._conv_figs(), sizing_mode="stretch_width")
        # results at write times
        self.section_sel = pn.widgets.Select(name="Section", options=[], width=260)
        self.section_sel.param.watch(lambda e: self.update_results(force=True), "value")
        self.yplus_box = pn.Column(sizing_mode="stretch_width")
        self.cp_box = pn.Column(sizing_mode="stretch_width")
        self.contour_box = pn.Column(sizing_mode="stretch_width")
        self.vortex_box = pn.Column(sizing_mode="stretch_width")
        # AI review
        self.ai_btn = pn.widgets.Button(name="Write AI review", button_type="primary", width=200)
        self.ai_btn.on_click(lambda e: self.on_ai_review())
        self.ai_status = pn.pane.Markdown()
        self.ai_facts = pn.Column(sizing_mode="stretch_width")
        self.ai_text = pn.Column(sizing_mode="stretch_width")
        self.reader = None
        self._run_stamp = None      # modification time of run/settings.json when the reader was made
        self._last_parse = 0.0
        self._shown_time = None
        self._yplus_pred = None
        self._busy = False

    # ---------------- case ---------------- #
    @property
    def run_dir(self):
        return os.path.join(self.app.folder, sm.RUN_DIR)

    def load(self):
        '''A case was loaded: forget the results shown for the previous one.'''
        self.reader, self._run_stamp = None, None
        self._shown_time = None
        self._yplus_pred = None
        self.ai_status.object = ""
        self.ai_facts.objects = []
        self.clear_results()

    def clear_results(self):
        for s in self.res_src.values():
            s.data = dict(it=[], r=[])
        self.force_src.data = dict(it=[], Cl=[], Cd=[], Cm=[])
        self._conv_set = {}                 # another run or case: the ranges follow the data again
        for box in (self.yplus_box, self.cp_box, self.contour_box, self.vortex_box):
            box.objects = []
        self.log.object = ""

    def _fill_inputs(self):
        c = self.app.case
        sim = c.simulation
        self._updating = True
        try:
            self.cores.value = max(1, min(int(sim.cores), self.cores.end))
            self.V.value = float(c.flow.U)
            self.alpha.value = float(sim.alpha)
            self.model.value = sim.turbulence if sim.turbulence in sm.TURBULENCE else "kOmegaSST"
            self.iters.value = int(sim.iterations)
            self.sections_in.value = ", ".join("{:g}".format(y) for y in sim.sections) if sim.sections else ""
        finally:
            self._updating = False
        self.defaults.objects = [pn.widgets.Tabulator(pd.DataFrame(sm.settings_table(c)), show_index=False,
                                                      disabled=True, layout="fit_columns", widths={"setting": 230},
                                                      sizing_mode="stretch_width",
                                                      formatters={"value": {"type": "textarea"}},
                                                      configuration={"columnDefaults": {"headerSort": False}})]

    def on_input(self, event):
        if self._updating:
            return
        c = self.app.case
        sim = c.simulation
        try:
            text = self.sections_in.value.strip()
            secs = [float(v) for v in text.replace(";", ",").split(",") if v.strip()] if text else None
        except ValueError:
            self.app.notify("warning", "Section planes: numbers separated by commas")
            return
        if self.V.value and self.V.value > 0:
            c.flow.U = float(self.V.value)
        sim.cores, sim.alpha = int(self.cores.value), float(self.alpha.value or 0.0)
        sim.turbulence, sim.iterations, sim.sections = self.model.value, int(self.iters.value or 10), secs
        self.app.changed()
        self._fill_inputs()
        self.update_load_note()

    # ---------------- machine load ---------------- #
    def busy_now(self):
        '''Busy processes on the machine (all users): the median of the last samples, or a short measurement.'''
        if len(self._load_samples) >= 3:
            return float(np.median(self._load_samples))
        return sm.busy_processes()

    def update_load_note(self, alive=None):
        '''The line under the cores slider: busy processes now; a warning if the run would share cores.'''
        alive = self.running() if alive is None else alive
        busy, n = self.busy_now(), self.ncpu
        if alive:
            mine = int((sm.load_settings(self.run_dir) or {}).get("cores", 0))
            if busy > n + 2:
                kind, text = "warning", ("{:.0f} busy processes share the {} cores of this machine, {} of them this "
                                         "run: it is slowed down a lot (its processes wait for each other). Close "
                                         "other jobs, or stop it and run again with fewer cores.".format(busy, n, mine))
            else:
                kind, text = None, "Machine now: {} for {} cores, {} of them this run.".format(_procs(busy), n, mine)
        else:
            msg = sm.oversubscribed(int(self.cores.value), busy, n)
            kind, text = ("warning", msg) if msg else (None, "Machine now: {} for {} cores (all users).".format(
                _procs(busy), n))
            if not msg and self._overload_ok:                 # no longer busy: no confirmation needed
                self._overload_ok = 0.0
                self.run_btn.name = RUN_LABEL
        if (kind, text) != self._load_text:
            self._load_text = (kind, text)
            self.load_box.objects = [note(text, kind)] if kind else [
                pn.pane.Markdown(text, styles={"color": "#555", "font-size": "12px"}, margin=(0, 10))]

    # ---------------- view ---------------- #
    def view(self):
        self._fill_inputs()
        c = self.app.case
        mesh = os.path.isfile(os.path.join(self.app.folder, "constant", "polyMesh", "owner"))
        inputs = pn.Column(self.cores, self.load_box, pn.Row(self.V, self.alpha), pn.Row(self.model, self.iters),
                           self.sections_in,
                           pn.Row(self.run_btn, self.stop_btn), self.progress, width=540, sizing_mode="fixed",
                           margin=(5, 40, 5, 10))
        out = []
        if not mesh:
            out.append(note("No mesh yet: build it in step 5 first.", "warning"))
        out += [pn.Row(inputs, pn.Card(self.defaults, title="Settings of the test run", collapsed=True,
                                       sizing_mode="stretch_width"), sizing_mode="stretch_width")]
        out += section("Log") + [self.log_box]
        if os.path.isdir(self.run_dir) and sm.stale(self.app.folder, self.run_dir):
            out.append(note("The mesh was built again after this run: the results below belong to the old mesh.",
                            "warning"))
        out += section("Convergence") + [self.conv]
        out += section("y+ of the solution and the prediction") + [self.yplus_box]
        out += section("Pressure distribution") + [self.section_sel, self.cp_box]
        out += section("Contours of pressure and velocity") + [self.contour_box]
        if c.dim == "3D":
            out += section("Tip vortex") + [self.vortex_box]
        out += section("AI review of the solution") + [self.ai_view()]
        self.poll(force=True)
        return out

    def _conv_figs(self):
        # explicit ranges, set from the data (_conv_ranges): automatic ranges did not follow the residuals down and
        # included the first iterations of the force coefficients (about 10 times their converged values)
        self.conv_ranges = {"res x": Range1d(0, 1), "res y": Range1d(1e-6, 1.0), "force x": Range1d(0, 1),
                            "force y": Range1d(-1.0, 1.0)}
        self._conv_set = {}                 # the values set last, per range: a range moved by the user stays
        f = figure(title="Initial residuals", height=300, sizing_mode="stretch_width", y_axis_type="log",
                   tools="pan,wheel_zoom,box_zoom,reset,save", x_range=self.conv_ranges["res x"],
                   y_range=self.conv_ranges["res y"])
        f.toolbar.logo = None
        f.xaxis.axis_label, f.yaxis.axis_label = "iteration", "initial residual"
        for name, src in self.res_src.items():
            f.line("it", "r", source=src, color=RES_COLORS[name], legend_label=name, line_width=1.5)
        f.legend.click_policy, f.legend.label_text_font_size = "hide", "10px"
        g = figure(title="Force coefficients (range: the last 80 % of the iterations)", height=300,
                   sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save",
                   x_range=self.conv_ranges["force x"], y_range=self.conv_ranges["force y"])
        g.toolbar.logo = None
        g.xaxis.axis_label, g.yaxis.axis_label = "iteration", "coefficient"
        g.line("it", "Cl", source=self.force_src, color="#1f77b4", legend_label="Cl", line_width=1.5)
        g.line("it", "Cd", source=self.force_src, color="#d62728", legend_label="Cd", line_width=1.5)
        g.line("it", "Cm", source=self.force_src, color="#2ca02c", legend_label="Cm (pitch)", line_width=1.2)
        g.add_tools(HoverTool(tooltips=[("iteration", "@it"), ("Cl", "@Cl{0.0000}"), ("Cd", "@Cd{0.00000}"),
                                        ("Cm", "@Cm{0.0000}")], mode="vline"))
        g.legend.click_policy, g.legend.label_text_font_size = "hide", "10px"
        return [pn.pane.Bokeh(f, sizing_mode="stretch_width"), pn.pane.Bokeh(g, sizing_mode="stretch_width")]

    def _set_range(self, key, lo, hi):
        '''Set a range of the convergence plots, unless the user zoomed or panned it (reset: it follows again).'''
        rng, last = self.conv_ranges[key], self._conv_set.get(key)
        if last is not None and (rng.start, rng.end) != last:
            return
        rng.update(start=lo, end=hi, reset_start=lo, reset_end=hi)
        self._conv_set[key] = (rng.start, rng.end)

    def _conv_ranges(self):
        '''Ranges of the convergence plots from the data: all residuals (log, padded), the force coefficients of
        the last 80 % of the iterations (the first ones are far off), the iterations so far.'''
        res = [np.asarray(src.data["r"], float) for src in self.res_src.values() if len(src.data["r"])]
        its = [v for src in self.res_src.values() for v in src.data["it"][-1:]]
        if res:
            r = np.concatenate(res)
            r = r[r > 0]
            if len(r):
                lo, hi = math.log10(r.min()), math.log10(r.max())
                self._set_range("res y", 10 ** (lo - 0.25), 10 ** (hi + 0.15))
            self._set_range("res x", 0, max(its + [1]))
        d = self.force_src.data
        if len(d["it"]):
            n = len(d["it"])
            k = n // 5 if n >= 10 else 0
            v = np.concatenate([np.asarray(d[c][k:], float) for c in ("Cl", "Cd", "Cm")])
            v = v[np.isfinite(v)]
            if len(v):
                lo, hi = float(v.min()), float(v.max())
                pad = max(0.1 * (hi - lo), 0.02 * max(abs(lo), abs(hi)), 1e-4)
                self._set_range("force y", lo - pad, hi + pad)
            self._set_range("force x", 0, max(float(d["it"][-1]), 1))

    # ---------------- run ---------------- #
    def on_run(self):
        c, folder = self.app.case, self.app.folder
        if not os.path.isfile(os.path.join(folder, "constant", "polyMesh", "owner")):
            self.app.notify("warning", "No mesh yet: build it in step 5 first")
            return
        from ui.build import _BUILDING
        if folder in _BUILDING:
            self.app.notify("warning", "The mesh of this case is being built")
            return
        if self.running():
            self.app.notify("warning", "A test run of this case is already running")
            return
        cores, busy = int(self.cores.value), self.busy_now()
        if sm.oversubscribed(cores, busy, self.ncpu) and time.time() - self._overload_ok > CONFIRM_FOR:
            self._overload_ok = time.time()                          # the first click warns, the second starts
            self.run_btn.name = RUN_ANYWAY
            self.update_load_note()
            self.app.notify("warning", "Machine busy: {} processes would share {} cores (see under Cores). Choose "
                                       "fewer cores, or click \"{}\".".format(cores + int(round(busy)), self.ncpu,
                                                                            RUN_ANYWAY))
            return
        self._overload_ok = 0.0
        self.run_btn.name = RUN_LABEL
        self.run_btn.disabled = True

        def start(ok):
            if not ok:
                self.run_btn.disabled = False
                self.app.notify("error", "Not started: the case could not be saved")
                return
            try:
                run = sm.write_run(self.app.case, folder)
            except Exception as e:  # noqa: BLE001
                self.run_btn.disabled = False
                self.app.notify("error", "Cannot write the run: " + repr(e))
                return
            state = {"proc": None, "lines": ["$ ./Allrun ({} cores)".format(c.simulation.cores)], "code": None,
                     "started": time.time()}

            def target():
                try:
                    proc = sm.start(run)
                    state["proc"] = proc
                    for line in proc.stdout:
                        state["lines"].append(line.rstrip("\n"))
                    state["code"] = proc.wait()
                except Exception as e:  # noqa: BLE001
                    state["lines"].append("== error: " + repr(e))
                    state["code"] = -1
            state["thread"] = threading.Thread(target=target, daemon=True)
            with _RUNS_LOCK:
                _RUNS[run] = state
            state["thread"].start()
            self.reader = None
            self._shown_time = None
            self._yplus_pred = None
            self.clear_results()
            self.app.notify("info", "Test run started")
            self.app.refresh_view(self)          # settings table, stale-mesh note and AI review of the new run
        self.app.save(then=start)

    def running(self):
        st = _RUNS.get(self.run_dir)
        return bool(st and st["thread"].is_alive())

    def on_stop(self):
        st = _RUNS.get(self.run_dir)
        if not st or not st["thread"].is_alive():
            return
        if st.get("stop_requested") and time.time() - st["stop_requested"] > 60 and st.get("proc"):
            sm.kill(st["proc"])
            self.app.notify("warning", "Test run killed")
            return
        sm.request_stop(self.run_dir)
        st["stop_requested"] = time.time()
        self.stop_btn.name = "Stopping ... (click again after a minute to kill)"
        self.app.notify("info", "simpleFoam writes the current iteration and ends")

    # ---------------- live update ---------------- #
    def poll(self, force=False):
        '''Called periodically by the app (and when the step is shown).'''
        if self.app.case is None or self.app.step.value != self.title and not force:
            return
        v = sm.procs_running()
        if v is not None:
            self._load_samples.append(v)
        st = _RUNS.get(self.run_dir)
        alive = bool(st and st["thread"].is_alive())
        self.run_btn.disabled = alive
        self.stop_btn.disabled = not alive
        if not alive:
            self.stop_btn.name = "Stop (write and end)"
        now = time.time()
        if self._overload_ok and now - self._overload_ok > CONFIRM_FOR:
            self._overload_ok = 0.0
            self.run_btn.name = RUN_LABEL
        if not force and now - self._last_parse < PARSE_EVERY:
            return
        self._last_parse = now
        self.update_load_note(alive)
        if not os.path.isdir(self.run_dir):
            self.progress.object = "_No test run yet._"
            return
        sf = os.path.join(self.run_dir, sm.SETTINGS_FILE)
        stamp = os.path.getmtime(sf) if os.path.isfile(sf) else None
        if self.reader is None or stamp != self._run_stamp:    # a new run (also one started in another session)
            if self._run_stamp is not None and stamp != self._run_stamp:
                self.clear_results()
            self.reader, self._run_stamp, self._shown_time = sm.LogReader(self.run_dir), stamp, None
            force = True
        changed = self.reader.update()
        lines = st["lines"] if st else []
        tail = []
        lp = os.path.join(self.run_dir, "log.simpleFoam")
        if os.path.isfile(lp):
            with open(lp, errors="replace") as fh:
                fh.seek(max(0, os.path.getsize(lp) - 4000))
                tail = fh.read().splitlines()[-25:]
        self.log.object = "\n".join(lines[-8:] + (["--- log.simpleFoam (last lines) ---"] + tail if tail else []))
        if changed or force:
            for f, src in self.res_src.items():
                it, r = self.reader.res.get(f, ([], []))
                src.data = dict(it=list(it), r=list(r))
            fh = sm.force_history(self.run_dir)
            if fh.get("Cl") is not None and len(fh["Cl"]):
                self.force_src.data = dict(it=list(fh["Time"]), Cl=list(fh["Cl"]), Cd=list(fh["Cd"]),
                                           Cm=list(fh.get("CmPitch", np.zeros(len(fh["Cl"])))))
            self._conv_ranges()
        self.progress.object = self._progress_text(st, alive)
        self.update_results()

    def _progress_text(self, st, alive):
        s = sm.load_settings(self.run_dir) or {}
        n = s.get("iterations", 0)
        it = self.reader.iteration if self.reader else 0
        speed = self.reader.clock_time / it if self.reader and it else None
        parts = ["**Iteration {} of {}**".format(it, n)]
        if speed:
            parts.append("{:.2f} s per iteration".format(speed))
            if alive and n > it:
                parts.append("about {:.0f} min left".format(speed * (n - it) / 60))
        if self.reader and self.reader.converged:
            parts.append("converged (residual criteria) at iteration {}".format(self.reader.converged))
        if self.reader and self.reader.error:
            parts.append("**error:** " + self.reader.error)
        if alive:
            parts.append("running")
        elif st and st.get("code") is not None:
            parts.append("finished (exit code {})".format(st["code"]))
        elif self.reader and self.reader.ended:
            parts.append("finished")
        return " · ".join(parts)

    # ---------------- results at write times ---------------- #
    def update_results(self, force=False):
        run = self.run_dir
        times = sm.surface_times(run)
        if times and self.running():
            # the newest write may still be in progress: wait until its files are a few seconds old
            newest = os.path.join(run, "postProcessing", "surfaces", times[-1])
            if time.time() - max([os.path.getmtime(newest)] + [os.path.getmtime(os.path.join(newest, f))
                                                                 for f in os.listdir(newest)]) < 5.0:
                times = times[:-1]
        if not times:
            return
        s = sm.load_settings(run) or {}
        secs = s.get("sections", [])
        opts = {"y = {:g} m".format(y): y for y in secs}
        if list(self.section_sel.options) != list(opts):
            self._updating = True
            try:
                self.section_sel.options = opts
                if secs:
                    self.section_sel.value = secs[len(secs) // 2]
            finally:
                self._updating = False
        key = (times[-1], self.section_sel.value)
        if (key == self._shown_time and not force) or self._busy:
            return
        self._busy = True
        case, folder, af = self.app.case, self.app.folder, self.app.airfoil_file
        y_sel = self.section_sel.value

        def job():
            from ui.step1_geometry import airfoil_data
            airfoil = airfoil_data(case.airfoil.name, af)
            t = times[-1]
            V = float(s.get("V", case.flow.U))
            a = math.radians(float(s.get("alpha", 0.0)))
            fd = (math.cos(a), 0.0, math.sin(a))
            wall = sm.wall_surface(case, airfoil, run, t)
            out = {"t": t, "V": V, "fd": fd, "cp": sm.cp_section(case, airfoil, wall, y_sel, V, fd) if y_sel is not None
                   else None, "yplus": sm.yplus_history(run), "y_sel": y_sel, "share": sm.separated_share(wall, fd),
                   "top_png": render_top(case, wall, V, fd, secs) if case.dim == "3D" else None}
            if self._yplus_pred is None:
                try:
                    self._yplus_pred = measured_yplus(folder, case) or {}
                except Exception:  # noqa: BLE001
                    self._yplus_pred = {}
            if y_sel is not None and y_sel in secs:
                i = secs.index(y_sel)
                plane = sm.surface(run, t, "section{}".format(i), ["p", "U"])
                sec = [float(v) for v in sm.mesh_sections(case, y_sel)]        # the section as the mesh has it
                ox, oz = place(airfoil.outline(), *sec, case.cog)
                out["section_png"] = render_section(plane, np.column_stack((ox, oz)), V, y_sel, sec[0])
            pngs = []
            if case.dim == "3D" and case.wing_stations:
                b2 = case.half_span or 0.0
                tip_z = float(np.mean(place(airfoil.outline(), *section_params(case.wing_stations, b2), case.cog)[1]))
                for i, (x, c_tip) in enumerate(s.get("wake_planes", [])):
                    pl = sm.surface(run, t, "wake{}".format(i), ["U", "vorticity"])
                    pngs.append(render_wake(pl, V, c_tip, b2, tip_z, x, "{:g} tip chords behind the most downstream "
                                                                         "trailing edge".format(sm.WAKE_PLANES[i])))
            out["wake_pngs"] = pngs
            return out

        def done(res, error):
            self._busy = False
            if error:
                if not self.running():          # while running: a write in progress, read again at the next poll
                    self._shown_time = key
                    self.cp_box.objects = [note("Cannot read the results: " + repr(error), "danger")]
                return
            self._shown_time = key
            self.show_results(res)
        self.app.run_background(job, done)

    def show_results(self, res):
        t = res["t"]
        # y+
        rows = []
        pred = self._yplus_pred or {}
        target = None
        try:
            from core.meshopt import targets
            target = targets(self.app.case)["yplus"]
        except Exception:  # noqa: BLE001
            pass
        for patch, arr in sorted(res["yplus"].items()):
            p = pred.get(patch, {}).get("y+ median")
            rows.append({"patch": patch, "predicted (flat plate, median)": "{:.3g}".format(p) if p else "",
                         "solution min": "{:.3g}".format(arr[-1, 1]), "solution average": "{:.3g}".format(arr[-1, 3]),
                         "solution max": "{:.3g}".format(arr[-1, 2]),
                         "average / predicted": "{:.2f}".format(arr[-1, 3] / p) if p else ""})
        ytab = pn.widgets.Tabulator(pd.DataFrame(rows), show_index=False, disabled=True, layout="fit_data_table",
                                    sizing_mode="stretch_width")
        items = [pn.pane.Markdown("_Iteration {}. Target of the mesh: y+ {}. The prediction uses the flat-plate skin "
                                  "friction with the local chord (step 4); the solution's y+ follows from its wall shear "
                                  "stress._".format(t, "{:g}".format(target) if target else "-")), ytab]
        cp = res["cp"]
        if cp:
            f = figure(title="y+ along the airfoil at y = {:.4g} m".format(cp["y"]), height=280,
                       sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
            f.toolbar.logo = None
            f.xaxis.axis_label, f.yaxis.axis_label = "x / c", "y+"
            for side, col in (("upper", "#1f77b4"), ("lower", "#2ca02c")):
                o = cp[side]
                if "yPlus" in o and len(o["s"]):
                    f.line(o["s"], o["yPlus"], color=col, legend_label=side + " side")
                    f.scatter(o["s"], o["yPlus"], color=col, size=3)
            if target:
                f.add_layout(Span(location=target, dimension="width", line_dash="dashed", line_color="black"))
            items.append(bokeh(f))
        self.yplus_box.objects = items
        # pressure distribution
        items = []
        if cp:
            f = figure(title="Pressure coefficient at y = {:.4g} m, the wall faces nearest to the section plane "
                             "(red x: reversed flow at the wall)".format(cp["y"]),
                       height=340, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
            f.toolbar.logo = None
            f.xaxis.axis_label, f.yaxis.axis_label = "x / c", "Cp"
            f.y_range.flipped = True
            for side, col in (("upper", "#1f77b4"), ("lower", "#2ca02c")):
                o = cp[side]
                if not len(o["s"]):
                    continue
                f.line(o["s"], o["Cp"], color=col, legend_label=side + " side", line_width=1.6)
                f.scatter(o["s"], o["Cp"], color=col, size=3)
                if "tau_x" in o:
                    rev = sm.reversed_flow(o["s"], o["tau_x"])
                    if rev.any():
                        f.scatter(o["s"][rev], o["Cp"][rev], color="#d62728", size=6, marker="x")
            f.add_tools(HoverTool(tooltips=[("x/c", "$x{0.000}"), ("Cp", "$y{0.000}")]))
            items.append(bokeh(f))
        share = res.get("share")
        if share:
            items.append(pn.pane.Markdown(
                "Reversed flow at the wall (the wall shear against the flow, behind the nose x/c > {:g}; chord- and "
                "span-weighted): **{:.1%}** of the upper side, **{:.1%}** of the lower side".format(
                    sm.NOSE, share["upper"], share["lower"])))
        if res.get("top_png"):
            items += [pn.pane.Markdown("**The wing seen from above** (flow downwards)"),
                      pn.pane.PNG(res["top_png"], sizing_mode="scale_width", max_width=1300)]
        self.cp_box.objects = items
        png = res.get("section_png")
        self.contour_box.objects = [pn.pane.PNG(png, sizing_mode="scale_width", max_width=1300)] if png else \
            [pn.pane.Markdown("_No section plane at this position._")]
        self.vortex_box.objects = [pn.Row(*[pn.pane.PNG(p, sizing_mode="scale_width", max_width=640)
                                            for p in res["wake_pngs"] if p], sizing_mode="stretch_width")] \
            if any(res["wake_pngs"]) else []

    # ---------------- AI review ---------------- #
    def ai_view(self):
        found, status = aisummary.cli_status()
        items = [pn.pane.Markdown(
            "Claude ({}) reviews the test run: convergence, plausibility of the solution, effects of the mesh on the "
            "solution and the convergence, recommendations. **Write AI review** sends technical facts of the run and "
            "its mesh to Claude through Claude Code with your login (no API key, no extra bill; no case name, user "
            "name or file paths). It takes about a minute and is saved in `run/`.".format(aisummary.MODEL)),
            pn.pane.Markdown(status, styles={"color": "#555", "font-size": "12px"}) if found else note(status, "warning"),
            pn.Row(self.ai_btn, self.ai_status), self.ai_facts]
        saved = aisummary.load(self.run_dir, aisummary.REVIEW_FILE) if os.path.isdir(self.run_dir) else None
        if saved:
            text, info, mtime = saved
            log = os.path.join(self.run_dir, "log.simpleFoam")
            later = os.path.isfile(log) and os.path.getmtime(log) > mtime
            meta = "_Written {} by {} via {}{}._".format(
                datetime.datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M"), info.get("model", "Claude"),
                info.get("via", "Claude Code"), "; the run continued afterwards" if later else "")
            self.ai_text.objects = [pn.pane.Markdown(meta), pn.pane.Markdown(text, sizing_mode="stretch_width")]
        else:
            self.ai_text.objects = []
        items.append(self.ai_text)
        return pn.Column(*items, sizing_mode="stretch_width")

    def on_ai_review(self):
        c, folder, af, run = self.app.case, self.app.folder, self.app.airfoil_file, self.run_dir
        if not sm.surface_times(run):
            self.app.notify("warning", "No results yet: run the test first")
            return
        self.ai_btn.disabled, self.ai_btn.name = True, "Writing ..."
        self.ai_status.object = "_collecting the facts, then waiting for Claude (about a minute) ..._"
        pred = self._yplus_pred

        def job():
            from ui.step1_geometry import airfoil_data
            groups = meshquality.analyse(folder)
            facts = sm.solution_facts(c, folder, airfoil_data(c.airfoil.name, af), run, groups,
                                      pred if pred is not None else measured_yplus(folder, c))
            try:
                text, info = aisummary.write_review(facts)
            except RuntimeError as e:
                return facts, None, str(e)
            aisummary.save(run, text, info, aisummary.REVIEW_FILE)
            return facts, (text, info), None

        def done(result, error):
            self.ai_btn.disabled, self.ai_btn.name = False, "Write AI review"
            if error:
                self.ai_status.object = "**Failed:** " + repr(error)
                return
            facts, review, message = result
            self.ai_facts.objects = [pn.Card(pn.pane.JSON(json.dumps(facts), depth=2, sizing_mode="stretch_width"),
                                             title="Facts sent to Claude", collapsed=True,
                                             sizing_mode="stretch_width")]
            if message:
                self.ai_status.object = "**No review:** " + message
                return
            self.ai_status.object = "_{}: {:,} input and {:,} output tokens_".format(
                review[1].get("via", ""), review[1]["input tokens"], review[1]["output tokens"])
            self.app.refresh_view(self)
        self.app.run_background(job, done)
