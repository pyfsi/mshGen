'''
step1_geometry.py: step 1 of the GUI, geometry definition

Inputs:  2D/3D, airfoil (library of mshGen's airfoils), COG, planform (parameters or table)
Plots:   airfoil as mshGen reads it (points, removed point, splines, camber line), root section in mesh
         coordinates, planform top/front view with the slices, 3D view of the half wing, derived values
Follow-up: block vertices are mshGen defaults for the root section, COG, airfoil, mesh_scale/level;
         after a change they are recomputed (button, or automatically when the case is saved).
'''
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from core import airfoils, blockcheck, geometry  # noqa: F401
from core.planform import COLUMNS, Planform, describe, fit, place, section_params, stations_problems, surface
from ui.common import COLORS, bfig, bokeh, note, table
from ui.view3d import View3D

_AIRFOIL_CACHE = {}

PARAMS = [  # key, label, step
    ("half_span", "Half-span (m)", 0.1),
    ("root_chord", "Root chord (m)", 0.05),
    ("tip_chord", "Tip chord (m)", 0.05),
    ("kink_y", "Constant-chord part up to y (m)", 0.1),
    ("le_sweep_deg", "LE sweep outboard (deg, + backwards)", 1.0),
    ("twist_root_deg", "Twist at the root (deg, + nose up)", 0.5),
    ("twist_tip_deg", "Twist at the tip (deg)", 0.5),
    ("dihedral_deg", "Dihedral outboard (deg, + tip up)", 0.5),
]


def wing_figure(case, outline):
    '''Plotly 3D view of the half wing (step 1); also used for the tutorial video (static export).'''
    st = case.wing_stations or [[0, 0, 1.0, 0, 0, 0]]
    idx = np.linspace(0, len(outline) - 1, min(len(outline), 121)).astype(int)
    X, Y, Z = surface(st, outline[idx], case.cog, n_span=41)
    fig = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, "#9ecae1"], [1, "#3182bd"]], showscale=False,
                               surfacecolor=Z, lighting=dict(ambient=0.6, diffuse=0.6)))
    fig.update_layout(title="3D view of the half wing (drag to rotate)", height=420,
                      margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                                 camera=dict(eye=dict(x=-2.6, y=-1.8, z=1.8))))
    return fig


def airfoil_data(name, filename):
    key = (name, filename, airfoils.file_hash(filename))
    if key not in _AIRFOIL_CACHE:
        _AIRFOIL_CACHE[key] = airfoils.AirfoilData(name, filename)
    return _AIRFOIL_CACHE[key]


class GeometryStep:
    title = "1. Geometry"

    def __init__(self, app):
        self.app = app
        self._updating = False

        self.dim = pn.widgets.RadioButtonGroup(name="Dimension", options=["2D", "3D"], button_type="primary")
        self.airfoil = pn.widgets.Select(name="Airfoil (mshGen library)", options=airfoils.names_with_defaults())
        self.airfoil_info = pn.pane.Markdown()
        self.support = pn.Column()
        self.cog_x = pn.widgets.FloatInput(name="COG x (m)", step=0.01)
        self.cog_z = pn.widgets.FloatInput(name="COG z (m)", step=0.01)

        self.params = {k: pn.widgets.FloatInput(name=label, step=step) for k, label, step in PARAMS}
        self.param_msg = pn.pane.Markdown()
        self.table = pn.widgets.Tabulator(pd.DataFrame(columns=COLUMNS), show_index=False, height=240,
                                          layout="fit_data_stretch", selectable="checkbox",
                                          sizing_mode="stretch_width")
        self.add_row_btn = pn.widgets.Button(name="Insert row before the tip", button_type="light")
        self.del_row_btn = pn.widgets.Button(name="Delete selected rows", button_type="light")
        self.table_msg = pn.pane.Markdown()
        self.scale_slices = pn.widgets.Checkbox(name="Scale the slices when the span changes", value=True)
        self.planform_tabs = pn.Tabs(
            ("Parameters", pn.Column(*self.params.values(), self.param_msg)),
            ("Table", pn.Column(self.table, pn.Row(self.add_row_btn, self.del_row_btn), self.table_msg)))
        self.derived = pn.Column(sizing_mode="stretch_width")
        self.planform_box = pn.Column(pn.pane.Markdown("### Planform"), self.planform_tabs, self.scale_slices,
                                      pn.pane.Markdown("### Derived values"), self.derived)

        self.vertex_box = pn.Column(sizing_mode="stretch_width")
        self.plots = pn.Column(sizing_mode="stretch_width")

        self.dim.param.watch(self.on_dim, "value")
        self.airfoil.param.watch(self.on_airfoil, "value")
        self.cog_x.param.watch(self.on_cog, "value")
        self.cog_z.param.watch(self.on_cog, "value")
        for w in self.params.values():
            w.param.watch(self.on_param, "value")
        self.table.on_edit(self.on_table_edit)
        self.add_row_btn.on_click(self.on_add_row)
        self.del_row_btn.on_click(self.on_del_rows)

        self.inputs = pn.Column(
            pn.pane.Markdown("### Case type"), self.dim,
            pn.pane.Markdown("### Airfoil"), self.airfoil, self.airfoil_info,
            pn.Card(self.support, title="What mshGen does for this airfoil", collapsed=True),
            pn.pane.Markdown("### Mesh origin (COG)"), pn.Row(self.cog_x, self.cog_z),
            self.planform_box,
            pn.pane.Markdown("### Effect on the blocks"), self.vertex_box,
            width=430, sizing_mode="fixed")

    # ---------------- load case into widgets ---------------- #
    def load(self):
        c = self.app.case
        self._updating = True
        try:
            self.dim.value = c.dim
            options = airfoils.names_with_defaults()
            if c.airfoil.name not in options:       # e.g. Mrev-v2_aileron from its example (own vertices)
                options = options + [c.airfoil.name]
            self.airfoil.options = options
            self.airfoil.value = c.airfoil.name
            self.cog_x.value, self.cog_z.value = float(c.cog[0]), float(c.cog[2])
            self._fill_planform()
        finally:
            self._updating = False
        self.render()

    def _fill_planform(self):
        c = self.app.case
        if not c.wing.stations:
            return
        p = Planform(**c.wing.planform) if c.wing.planform else fit(c.wing.stations)
        for k, w in self.params.items():
            w.value = float(getattr(p, k))
        self.table.value = pd.DataFrame(c.wing.stations, columns=COLUMNS)
        if c.wing.planform:
            self.param_msg.object = "_The table is generated from these parameters._"
            self.planform_tabs.active = 0
        else:
            self.param_msg.object = ("_The table was edited by hand (or imported): the parameters show an "
                                     "approximation, and changing one replaces the table._")
            self.planform_tabs.active = 1

    def view(self):
        return [pn.Row(self.inputs, self.plots, sizing_mode="stretch_width")]

    # ---------------- callbacks ---------------- #
    def _changed(self, render=True):
        self.app.changed()
        if render:
            self.render()

    def on_dim(self, event):
        if self._updating:
            return
        geometry.set_dim(self.app.case, event.new, self.app.dim_memory)
        self._updating = True
        try:
            self._fill_planform()
        finally:
            self._updating = False
        self.app.notify("info", "Switched to " + event.new + ": slices set to " + str(self.app.case.blocks.slice_locations))
        self._changed()

    def on_airfoil(self, event):
        if self._updating:
            return
        c = self.app.case
        c.airfoil.name = event.new
        c.airfoil.source = "meshGeneration/airfoils/" + airfoils.LIBRARY[event.new]["file"]
        self.app.set_airfoil_file(airfoils.library_file(event.new))
        try:
            geometry.check_level(c)
        except ValueError:
            if not airfoils.has_defaults(event.new):
                self.app.notify("warning", "mshGen's default block vertices do not work for " + event.new)
                self._changed()
                return
            c.domain.mesh_scale, c.domain.mesh_level = airfoils.default_mesh_scale(event.new), "COARSE"
            self.app.notify("warning", "No mshGen defaults for this airfoil at the previous R/c and level: "
                                       "set R = {:g}c, COARSE".format(c.domain.mesh_scale))
        self._changed()

    def on_cog(self, event):
        if self._updating:
            return
        c = self.app.case
        c.cog = [float(self.cog_x.value or 0.0), float(c.cog[1]), float(self.cog_z.value or 0.0)]
        self._changed()

    def on_param(self, event):
        if self._updating:
            return
        p = Planform(**{k: float(w.value if w.value is not None else 0.0) for k, w in self.params.items()})
        problems = p.problems()
        if problems:
            self.param_msg.object = "**Not applied:** " + "; ".join(problems)
            return
        c = self.app.case
        f = geometry.set_stations(c, p.to_stations(), self.scale_slices.value)
        c.wing.planform = p.to_dict()
        self.param_msg.object = "_The table is generated from these parameters._"
        self._updating = True
        try:
            self.table.value = pd.DataFrame(c.wing.stations, columns=COLUMNS)
        finally:
            self._updating = False
        if f:
            self.app.notify("info", "Slices scaled with the span (x{:.4g})".format(f))
        self._changed()

    def _apply_table(self, df):
        c = self.app.case
        rows = df[COLUMNS].astype(float).values.tolist()
        problems = stations_problems(rows)
        if problems:
            self.table_msg.object = "**Not applied:** " + "; ".join(problems)
            return
        self.table_msg.object = ""
        f = geometry.set_stations(c, rows, self.scale_slices.value)
        c.wing.planform = None
        self._updating = True
        try:
            p = fit(rows)
            for k, w in self.params.items():
                w.value = float(getattr(p, k))
            self.param_msg.object = ("_The table was edited by hand: the parameters show an approximation, "
                                     "and changing one replaces the table._")
        finally:
            self._updating = False
        if f:
            self.app.notify("info", "Slices scaled with the span (x{:.4g})".format(f))
        self._changed()

    def on_table_edit(self, event):
        if not self._updating:
            self._apply_table(self.table.value)

    def on_add_row(self, event):
        df = self.table.value
        if len(df) < 2:
            return
        mid = (df.iloc[-2] + df.iloc[-1]) / 2
        df = pd.concat([df.iloc[:-1], mid.to_frame().T, df.iloc[-1:]], ignore_index=True)
        self._updating = True
        try:
            self.table.value = df
        finally:
            self._updating = False
        self._apply_table(df)

    def on_del_rows(self, event):
        sel = self.table.selection
        df = self.table.value
        if not sel:
            self.table_msg.object = "Select rows with the checkboxes first."
            return
        if len(df) - len(sel) < 2:
            self.table_msg.object = "**Not applied:** at least two rows (root and tip) must remain."
            return
        df = df.drop(index=df.index[sel]).reset_index(drop=True)
        self._updating = True
        try:
            self.table.selection = []
            self.table.value = df
        finally:
            self._updating = False
        self._apply_table(df)

    # ---------------- plots ---------------- #
    def refresh(self):
        self.render()

    def render(self):
        c = self.app.case
        info = airfoils.LIBRARY.get(c.airfoil.name, {})
        self.airfoil_info.object = "{}  \nFile `{}`; used in examples: {}".format(
            info.get("description", "(not in the library)"), c.airfoil.source,
            ", ".join(info.get("examples", [])) or "-")
        self.support.objects = [table([{"topic": t, "mshGen": v} for t, v in airfoils.support(c.airfoil.name)])]
        self.planform_box.visible = c.dim == "3D"
        self.vertex_box.objects = [self.app.vertex_panel.view(compact=True)]
        if c.dim == "3D":
            try:
                d = describe(c.wing_stations or [[0, 0, 1.0, 0, 0, 0]])
                self.derived.objects = [table([{"quantity": k, "value": v} for k, v in d.items()],
                                              sizing_mode="stretch_width")]
            except Exception as e:  # noqa: BLE001
                self.derived.objects = [note("Cannot compute the derived values: " + repr(e), "danger")]
        try:
            self.plots.objects = self.make_plots()
        except Exception as e:  # noqa: BLE001
            self.plots.objects = [note("Cannot draw the geometry: " + repr(e), "danger")]

    def make_plots(self):
        c = self.app.case
        a = airfoil_data(c.airfoil.name, self.app.airfoil_file)
        out = []

        # Airfoil as mshGen reads it
        f = bfig("Airfoil {} as mshGen reads it: t/c = {:.1f} %, camber = {:.1f} %".format(
            c.airfoil.name, 100 * a.max_thickness, 100 * a.max_camber), "x/c", "z/c", height=280,
            hover=[("x/c", "$x{0.0000}"), ("z/c", "$y{0.0000}")])
        f.line(a.upper[:, 0], a.upper[:, 1], color=COLORS["upper"], line_width=2, legend_label="upper spline")
        f.line(a.lower[:, 0], a.lower[:, 1], color=COLORS["lower"], line_width=2, legend_label="lower spline")
        f.line(a.camber[:, 0], a.camber[:, 1], color=COLORS["camber"], line_dash="dashed", legend_label="camber line")
        f.scatter(a.points[:, 0], a.points[:, 1], size=4, color=COLORS["points"], legend_label="points used")
        if a.removed is not None:
            f.scatter([a.removed[0]], [a.removed[1]], marker="x", size=14, line_width=3, color=COLORS["removed"],
                      legend_label="point removed by mshGen")
        out.append(bokeh(f))

        # Root section in mesh coordinates
        st = c.wing_stations or [[0, 0, 1.0, 0, 0, 0]]
        outline = a.outline()
        x, z = place(outline, *section_params(st, 0.0), c.cog)
        f = bfig("Root section in mesh coordinates (origin = COG, chord {:.4g} m)".format(section_params(st, 0.0)[0]),
                 "x (m)", "z (m)", height=260, hover=[("x", "$x{0.000}"), ("z", "$y{0.000}")])
        f.patch(x, z, fill_color=COLORS["fill"], line_color=COLORS["upper"])
        f.scatter([0], [0], marker="cross", size=16, line_width=2, color="black", legend_label="COG (origin)")
        out.append(bokeh(f))

        if c.dim == "3D":
            out += self.planform_plots(c, st, outline)
        return out

    def planform_plots(self, c, st, outline):
        out = []
        b2 = c.half_span
        ys = np.unique(np.r_[np.linspace(0, b2, 101), np.asarray(st)[:, 1]])
        le = np.array([place(np.array([[0.0, 0.0]]), *section_params(st, y), c.cog) for y in ys])[:, :, 0]
        te = np.array([place(np.array([[1.0, 0.0]]), *section_params(st, y), c.cog) for y in ys])[:, :, 0]

        # Top view: span horizontal, flow downwards
        f = bfig("Top view of the half wing (flow downwards, dashed: slices)", "y (m)", "x (m)", height=360,
                 hover=[("y", "$x{0.000}"), ("x", "$y{0.000}")])
        f.patch(np.r_[ys, ys[::-1]], np.r_[le[:, 0], te[::-1, 0]], fill_color=COLORS["fill"], line_alpha=0)
        f.line(ys, le[:, 0], color=COLORS["le"], line_width=2, legend_label="leading edge")
        f.line(ys, te[:, 0], color=COLORS["te"], line_width=2, legend_label="trailing edge")
        xr = (float(min(le[:, 0].min(), te[:, 0].min())), float(max(le[:, 0].max(), te[:, 0].max())))
        pad = 0.15 * (xr[1] - xr[0] + 1e-9)
        for s in c.blocks.slice_locations:
            f.line([s, s], [xr[0] - pad, xr[1] + pad], color=COLORS["slice"], line_dash="dashed")
        xw = blockcheck.near_wake_x(c)
        if xw is not None:
            f.line([0, c.blocks.slice_locations[-1]], [xw, xw], color="purple", line_dash="dotdash",
                   legend_label="near-wake end: TE must stay upstream")
        f.scatter(np.asarray(st)[:, 1], np.interp(np.asarray(st)[:, 1], ys, le[:, 0]), size=6,
                  color=COLORS["le"], legend_label="table rows")
        f.y_range.flipped = True
        out.append(bokeh(f))

        # Front view: height of LE and TE along the span
        f = bfig("Front view: height of LE and TE along the span (dihedral, twist)", "y (m)", "z (m)", height=220,
                 equal=False, hover=[("y", "$x{0.000}"), ("z", "$y{0.0000}")])
        f.line(ys, le[:, 1], color=COLORS["le"], line_width=2, legend_label="leading edge")
        f.line(ys, te[:, 1], color=COLORS["te"], line_width=2, legend_label="trailing edge")
        out.append(bokeh(f))

        # 3D view
        out.append(View3D(self.app, wing_figure(c, outline), height=440, slot="geometry", width=760).panel)
        return out
