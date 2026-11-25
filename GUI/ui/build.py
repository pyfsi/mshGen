'''
build.py: step 5 of the GUI, build and check

Options:  Fluent export, images (ParaView views), environment modules
Build:    save the case, block check, Allrun.pre with a live log
Results:  checkMesh summary next to the cells and faces that fail checkMesh or have a lower quality (top and side
          view, core/meshquality.py), Fluent zones, AI summary of the mesh (Claude, core/aisummary.py), 3D view of the
          wall patches (Plotly), images
'''
import datetime
import glob
import json
import os
import threading

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.plotting import figure

from core import aisummary, meshquality
from core.results import fluent_zones, measured_yplus, wall_surfaces
from core.runner import checkmesh_summary, run_allrun, run_quality
from ui.common import bokeh, note, table
from ui.view3d import View3D

_BUILDING = set()            # case folders being built (shared by all browser sessions)
_BUILDING_LOCK = threading.Lock()
WALL_COLORS = {"wingUp": "#4682b4", "wing": "#4682b4", "wingDown": "#e6873a", "wingTip": "#999999",
               "wingCut": "#c0504d", "extra": "#9b59b6"}
STATUS_COLORS = {"failed": ["#d62728", "#ff7f0e", "#8c564b", "#e377c2"],
                 "warning": ["#9467bd", "#bcbd22", "#7f7f7f"],
                 "lower quality": ["#1f77b4", "#17becf", "#2ca02c"]}
AI_TEXT = ("Claude ({}) reviews the mesh: what is good, a suggested modelling approach, what can be improved, "
           "things to consider, a crazy idea. **Write AI summary** sends the technical facts below to Claude through "
           "Claude Code with your login of the VS Code extension: no API key, no extra bill (it counts towards your "
           "Claude plan); no case name, user name or file paths are sent. It takes about a minute and is saved in the "
           "case folder.")
DIVIDER = {"margin": (18, 0, 6, 0)}


class BuildStep:
    title = "5. Build & check"
    intro = 'Builds the mesh (mshGen, blockMesh) and checks it (checkMesh); optionally images and a Fluent mesh.'

    def __init__(self, app):
        self.app = app
        self._updating = False
        self.log_lines, self.log_shown = [], 0
        self.log_lock = threading.Lock()
        self.thread = None
        self.build_btn = pn.widgets.Button(name="Save and build mesh (Allrun.pre)", button_type="primary")
        self.images_chk = pn.widgets.Checkbox(name="also make images (ParaView views)", value=True)
        self.fluent_chk = pn.widgets.Checkbox(name="export to Fluent (foamMeshToFluent)", value=False)
        self.mod_python = pn.widgets.TextInput(name="Python module")
        self.mod_openfoam = pn.widgets.TextInput(name="OpenFOAM module")
        self.mod_paraview = pn.widgets.TextInput(name="ParaView module")
        self.log = pn.pane.Str("", styles={"font-size": "12px"})
        self.log_box = pn.Column(self.log, height=320, scroll=True,
                                 styles={"border": "1px solid #ccc", "padding": "4px"})
        self.build_btn.on_click(self.on_build)
        self.fluent_chk.param.watch(self.on_options, "value")
        for w in (self.mod_python, self.mod_openfoam, self.mod_paraview):
            w.param.watch(self.on_options, "value")
        # quality of the built mesh
        self.quality_box = pn.Column(sizing_mode="stretch_width")
        self.quality_plots = pn.Column(sizing_mode="stretch_width")      # top and side view, side by side
        self.analyse_btn = pn.widgets.Button(name="Analyse the mesh quality (checkMesh, about 30 s)",
                                             button_type="light", width=380)
        self.analyse_btn.on_click(lambda e: self.on_analyse())
        self._quality = (None, None)            # (stamp, groups)
        self.q_zoom = pn.widgets.RadioButtonGroup(options=["Whole domain", "Wing"], value="Whole domain",
                                                  button_type="primary", width=260)
        self.q_zoom.param.watch(lambda e: self.set_quality_zoom(), "value")
        self._q_ranges = []                     # per plot: (x range, y range, extents {zoom: (x0, x1, y0, y1)})
        # AI summary
        self.ai_btn = pn.widgets.Button(name="Write AI summary", button_type="primary", width=200)
        self.ai_btn.on_click(lambda e: self.on_ai_summary())
        self.ai_status = pn.pane.Markdown()
        self.ai_facts = pn.Column(sizing_mode="stretch_width")
        self.ai_text = pn.Column(sizing_mode="stretch_width")
        # 3D view
        self.view3d_btn = pn.widgets.Button(name="Show 3D view of the walls", button_type="light")
        self.edges_chk = pn.widgets.Checkbox(name="cell edges", value=True)
        self.view3d_btn.on_click(lambda e: self.show_3d())
        self.view3d = pn.Column(sizing_mode="stretch_width")

    def load(self):
        c = self.app.case
        with self.log_lock:
            self.log_lines, self.log_shown = [], 0
        self.log.object = ""
        self.view3d.objects = []
        self.ai_status.object = ""
        self.ai_facts.objects = []
        self._quality = (None, None)
        self._updating = True
        try:
            self.fluent_chk.value = bool(c.environment.fluent)
            self.mod_python.value = c.environment.python_module
            self.mod_openfoam.value = c.environment.openfoam_module
            self.mod_paraview.value = c.environment.paraview_module
        finally:
            self._updating = False

    @property
    def running(self):
        return self.thread is not None

    def on_options(self, event):
        if self._updating:
            return
        env = self.app.case.environment
        env.fluent = bool(self.fluent_chk.value)
        env.python_module = self.mod_python.value.strip()
        env.openfoam_module = self.mod_openfoam.value.strip()
        env.paraview_module = self.mod_paraview.value.strip()
        self.app.changed()

    # ---------------- view ---------------- #
    def view(self):
        app = self.app
        self.log_box.visible = bool(self.log_lines)          # no empty box before the first build
        out = [pn.Row(self.build_btn, pn.Column(self.images_chk, self.fluent_chk)),
               pn.Card(self.mod_python, self.mod_openfoam, self.mod_paraview, title="Environment modules",
                       collapsed=True, sizing_mode="stretch_width"),
               self.log_box]
        out += self.section("Mesh (last build)") + [
            pn.Row(pn.Column(self.checkmesh_view(), width=470, sizing_mode="fixed", margin=(5, 40, 5, 10)),
                   self.quality_box, sizing_mode="stretch_width"), self.quality_plots]
        self.update_quality()
        out += self.fluent_view()
        out += self.section("AI summary of the mesh") + [self.ai_view()]
        out += self.section("3D view of the walls") + [pn.Row(self.view3d_btn, self.edges_chk), self.view3d]
        imgs = sorted(f for f in glob.glob(os.path.join(app.folder, "images", "*.png"))
                      if not os.path.basename(f).startswith("mshgen"))
        if imgs:
            out += self.section("Images (last build with images)")
            out.append(pn.FlexBox(*[pn.Column(pn.pane.Markdown("`" + os.path.basename(f) + "`"),
                                              pn.pane.PNG(f, width=440, sizing_mode="fixed"), sizing_mode="fixed",
                                              width=460) for f in imgs]))
        return out

    @staticmethod
    def section(title):
        '''A horizontal line and the section heading (as in step 4).'''
        return [pn.layout.Divider(**DIVIDER), pn.pane.Markdown("## " + title)]

    def checkmesh_view(self):
        s = checkmesh_summary(self.app.folder)
        if s is None:
            return pn.pane.Markdown("_No checkMesh log yet: build the mesh._")
        rows = [{"quantity": k, "value": s[k]} for k in ("cells", "hexahedra", "polyhedra", "max_aspect_ratio",
                                                         "max_non_orthogonality", "avg_non_orthogonality",
                                                         "max_skewness", "min_volume", "failed_checks")]
        alert = pn.pane.Alert("Mesh OK" if s["mesh_ok"] else
                              "Failed checks:\n\n" + "\n".join("- " + w for w in s["warnings"]),
                              alert_type="success" if s["mesh_ok"] else "warning")
        return pn.Column(alert, table(rows, width=450, sizing_mode="fixed"))

    def fluent_view(self):
        app = self.app
        msh = os.path.join(app.folder, "fluentInterface", app.case.name + ".msh")
        if not os.path.isfile(msh):
            return self.section("Fluent mesh") + [pn.pane.Markdown("_No Fluent mesh (enable the export and build)._")]
        zones = fluent_zones(msh)
        notes = ["`fluentInterface/{}.msh` ({:.0f} MB), units m.".format(app.case.name, os.path.getsize(msh) / 1e6)]
        if any(z["name"] == "inlet" and z["type"] != "velocity-inlet" for z in zones):
            notes.append("foamMeshToFluent writes `inlet` as **{}**: set it to velocity-inlet in Fluent.".format(
                next(z["type"] for z in zones if z["name"] == "inlet")))
        s = checkmesh_summary(app.folder)
        if s and s.get("polyhedra"):
            notes.append("{} polyhedral cells (merged interface beyond the tip, LIMITATIONS.md item 9); Fluent "
                         "reads them.".format(s["polyhedra"]))
        return self.section("Fluent mesh") + [pn.pane.Markdown("  \n".join(notes)),
                table(zones, width=500, sizing_mode="fixed")]

    # ---------------- quality of the built mesh ---------------- #
    def update_quality(self):
        '''Fill the quality view (cached per build; reading takes a few seconds, in the background).'''
        folder = self.app.folder
        self.quality_plots.objects = []
        if checkmesh_summary(folder) is None:
            self.quality_box.objects = []
            return
        if not meshquality.available(folder):
            self.quality_box.objects = [
                note("The cells that fail checkMesh or have a lower quality are shown here. This mesh was built "
                     "without the quality fields: build it again, or analyse it now.", "info"), self.analyse_btn]
            return
        st = meshquality.stamp(folder)
        if self._quality[0] == st:
            self.quality_box.objects = self.quality_view(self._quality[1])
            return
        self.quality_box.objects = [pn.pane.Markdown("_Reading the cells that fail checkMesh or have a lower "
                                                     "quality ..._")]

        def done(groups, error):
            if error:
                self.quality_box.objects = [note("Cannot read the mesh quality: " + repr(error), "danger")]
                return
            self._quality = (st, groups)
            self.quality_box.objects = self.quality_view(groups)
        self.app.run_background(lambda: meshquality.analyse(folder), done)

    def on_analyse(self):
        folder, module = self.app.folder, self.app.case.environment.openfoam_module
        self.analyse_btn.disabled, self.analyse_btn.name = True, "Running checkMesh ..."

        def done(code, error):
            self.analyse_btn.disabled = False
            self.analyse_btn.name = "Analyse the mesh quality (checkMesh, about 30 s)"
            if error or code != 0:
                self.app.notify("error", "Mesh quality analysis failed: " + (repr(error) if error else
                                                                              "see log.writeCellCentres"))
                return
            self.app.refresh_view(self)
        self.app.run_background(lambda: run_quality(folder, module, on_line=lambda line: None), done)

    def quality_view(self, groups):
        if not groups:
            return [note("No cells or faces flagged by checkMesh, and none with non-orthogonality above 50 deg or "
                         "skewness above 2.", "success")]
        c = self.app.case
        colors, used = {}, {k: 0 for k in STATUS_COLORS}
        for g in groups:
            pal = STATUS_COLORS[g["status"]]
            colors[g["name"]] = pal[used[g["status"]] % len(pal)]
            used[g["status"]] += 1
        figs, self._q_ranges = [], []
        for title, hx, hy, xl, yl in (("Top view (flow downwards)", "y", "x", "y (m)", "x (m)"),
                                      ("Side view (looking along the span)", "x", "z", "x (m)", "z (m)")):
            xr, yr = Range1d(0, 1), Range1d(0, 1)
            f = figure(title=title, height=360, sizing_mode="stretch_width", x_range=xr, y_range=yr,
                       tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom", output_backend="webgl")
            f.toolbar.logo = None
            f.xaxis.axis_label, f.yaxis.axis_label = xl, yl
            outlines = self._outlines(c, hx, hy)
            for xs, ys in outlines:
                f.patch(xs, ys, fill_color="#d9d9d9", fill_alpha=0.6, line_color="#636363")
            ph = np.concatenate([g[hx] for g in groups] + [np.asarray(o[0]) for o in outlines])
            pv = np.concatenate([g[hy] for g in groups] + [np.asarray(o[1]) for o in outlines])
            ext = {"Whole domain": (ph.min(), ph.max(), pv.min(), pv.max())}
            if outlines:
                oh = np.concatenate([np.asarray(o[0]) for o in outlines])
                ov = np.concatenate([np.asarray(o[1]) for o in outlines])
                m = 0.3 * max(np.ptp(oh), np.ptp(ov))
                ext["Wing"] = (oh.min() - m, oh.max() + m, ov.min() - m, ov.max() + m)
            else:
                ext["Wing"] = ext["Whole domain"]
            self._q_ranges.append((xr, yr, ext, hx == "y"))
            rends = []
            for g in groups:
                n = len(g["x"])
                val = g["value"] if g["value"] is not None else np.full(n, np.nan)
                src = ColumnDataSource(dict(h=g[hx], v=g[hy], x=g["x"], y=g["y"], z=g["z"], value=val,
                                            group=[g["criterion"]] * n))
                label = "{}: {} ({:,} {})".format(g["status"], g["criterion"], g["count"], g["kind"])
                rends.append(f.scatter("h", "v", source=src, size=4, color=colors[g["name"]], alpha=0.6,
                                       legend_label=label))
            f.add_tools(HoverTool(renderers=rends, tooltips=[("", "@group"), ("value", "@value{0.00}"),
                                                             ("x, y, z (m)", "@x{0.000}, @y{0.000}, @z{0.000}")]))
            pane = bokeh(f)                                  # legend below the plot, click to hide
            f.legend[0].orientation = "vertical"
            f.height += 22 * (len(groups) - 1)
            figs.append(pane)
        self.set_quality_zoom()
        self.quality_plots.objects = [self.q_zoom, pn.Row(*figs, sizing_mode="stretch_width")]
        rows = meshquality.summary_rows(groups)
        tab = pn.widgets.Tabulator(pd.DataFrame(rows), show_index=False, disabled=True, layout="fit_data_table",
                                   sizing_mode="stretch_width")
        return [pn.pane.Markdown("**Cells and faces that fail checkMesh (failed), that checkMesh warns about "
                                 "(warning), and lower-quality cells** (non-orthogonality 50-70 deg, skewness 2-4; "
                                 "at most {:,} points per group are drawn). Click a legend entry to hide it."
                                 .format(meshquality.MAX_POINTS)), tab]

    def set_quality_zoom(self):
        '''Both quality plots to the whole domain or to the wing (equal scales, top view with x downwards).'''
        for xr, yr, ext, flipped in self._q_ranges:
            x0, x1, y0, y1 = ext[self.q_zoom.value]
            w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
            ratio = 2.2                                     # plot width / height in data units
            if w / h < ratio:
                cx = 0.5 * (x0 + x1)
                x0, x1 = cx - 0.5 * h * ratio, cx + 0.5 * h * ratio
            else:
                cy = 0.5 * (y0 + y1)
                y0, y1 = cy - 0.5 * w / ratio, cy + 0.5 * w / ratio
            pad = 0.04 * (x1 - x0)
            xr.start, xr.end = x0 - pad, x1 + pad
            yr.start, yr.end = (y1 + pad, y0 - pad) if flipped else (y0 - pad, y1 + pad)

    def _outlines(self, case, hx, hy):
        '''Wing outlines for the quality plots: planform (top view) or root and tip sections (side view).'''
        from core.planform import place, section_params
        from ui.step1_geometry import airfoil_data
        st = case.wing_stations or [[0, 0, 1.0, 0, 0, 0]]
        try:
            a = airfoil_data(case.airfoil.name, self.app.airfoil_file)
        except Exception:  # noqa: BLE001
            a = None
        out = []
        if hx == "y":
            ys = np.unique(np.r_[np.linspace(0, case.half_span or 1.0, 61), np.asarray(st)[:, 1]])
            le = [place(np.array([[0.0, 0.0]]), *section_params(st, y), case.cog)[0][0] for y in ys]
            te = [place(np.array([[1.0, 0.0]]), *section_params(st, y), case.cog)[0][0] for y in ys]
            out.append((list(np.r_[ys, ys[::-1]]), list(np.r_[le, te[::-1]])))
        elif a is not None:
            for y in sorted({0.0, float(case.half_span or 0.0)}):
                x, z = place(a.outline(), *section_params(st, y), case.cog)
                out.append((list(x), list(z)))
        return out

    # ---------------- AI summary ---------------- #
    def ai_view(self):
        found, status = aisummary.cli_status()
        items = [pn.pane.Markdown(AI_TEXT.format(aisummary.MODEL)),
                 pn.pane.Markdown(status, styles={"color": "#555", "font-size": "12px"}) if found
                 else note(status, "warning"),
                 pn.Row(self.ai_btn, self.ai_status), self.ai_facts]
        saved = aisummary.load(self.app.folder)
        if saved:
            text, info, mtime = saved
            log = os.path.join(self.app.folder, "log.checkMesh")
            stale = os.path.isfile(log) and os.path.getmtime(log) > mtime
            meta = "_Written {} by {}{}{}._".format(
                datetime.datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M"), info.get("model", "Claude"),
                " via " + info["via"] if info.get("via") else "",
                "; the mesh was built again afterwards" if stale else "")
            self.ai_text.objects = [pn.pane.Markdown(meta), pn.pane.Markdown(text, sizing_mode="stretch_width")]
        else:
            self.ai_text.objects = []
        items.append(self.ai_text)
        return pn.Column(*items, sizing_mode="stretch_width")

    def on_ai_summary(self):
        c, folder, af = self.app.case, self.app.folder, self.app.airfoil_file
        if checkmesh_summary(folder) is None:
            self.app.notify("warning", "Build the mesh first")
            return
        self.ai_btn.disabled, self.ai_btn.name = True, "Writing ..."
        self.ai_status.object = "_collecting the facts, then waiting for Claude (about a minute) ..._"
        groups = self._quality[1] if self._quality[0] == meshquality.stamp(folder) else None

        def job():
            g = groups if groups is not None else meshquality.analyse(folder)
            facts = aisummary.mesh_facts(c, folder, af, g, measured_yplus(folder, c))
            try:
                text, info = aisummary.write_summary(facts)
            except RuntimeError as e:
                return facts, None, str(e)
            aisummary.save(folder, text, info)
            return facts, (text, info), None

        def done(result, error):
            self.ai_btn.disabled, self.ai_btn.name = False, "Write AI summary"
            if error:
                self.ai_status.object = "**Failed:** " + repr(error)
                return
            facts, summary, message = result
            self.ai_facts.objects = [pn.Card(pn.pane.JSON(json.dumps(facts), depth=2, sizing_mode="stretch_width"),
                                             title="Facts sent to Claude", collapsed=True,
                                             sizing_mode="stretch_width")]
            if message:
                self.ai_status.object = "**No summary:** " + message
                return
            self.ai_status.object = "_{}: {:,} input and {:,} output tokens_".format(
                summary[1].get("via", ""), summary[1]["input tokens"], summary[1]["output tokens"])
            self.app.refresh_view(self)
        self.app.run_background(job, done)

    # ---------------- 3D view ---------------- #
    def show_3d(self):
        folder = self.app.folder
        self.view3d_btn.disabled, self.view3d_btn.name = True, "Reading the mesh ..."
        edges = self.edges_chk.value

        def done(result, error):
            self.view3d_btn.disabled, self.view3d_btn.name = False, "Show 3D view of the walls"
            if error:
                self.view3d.objects = [note("Cannot read the mesh: " + repr(error), "danger")]
            elif not result:
                self.view3d.objects = [pn.pane.Markdown("_No mesh yet: build the mesh._")]
            else:
                patches = [(n, p, f, WALL_COLORS.get(n, "#888888")) for n, (p, f) in result.items()]
                self.view3d.objects = [View3D(self.app, self.figure_3d(result, edges), patches, height=620,
                                              slot="build", title="Wall patches of the built mesh",
                                              edges=edges).panel]
        self.app.run_background(lambda: wall_surfaces(folder), done)

    @staticmethod
    def figure_3d(surfaces, edges=True, max_edges=60000):
        fig = go.Figure()
        for name, (pts, faces) in surfaces.items():
            tri = []
            for f in faces:
                for k in range(1, len(f) - 1):
                    tri.append((f[0], f[k], f[k + 1]))
            tri = np.array(tri)
            fig.add_trace(go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                                    color=WALL_COLORS.get(name, "#888888"), name=name, showlegend=True,
                                    flatshading=True, lighting=dict(ambient=0.55, diffuse=0.7)))
            if edges and sum(len(f) for f in faces) <= max_edges:
                xs, ys, zs = [], [], []
                for f in faces:
                    loop = list(f) + [f[0]]
                    xs += list(pts[loop, 0]) + [None]
                    ys += list(pts[loop, 1]) + [None]
                    zs += list(pts[loop, 2]) + [None]
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="black", width=1),
                                           opacity=0.35, name=name + " cells", showlegend=False, hoverinfo="skip"))
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0), title="Wall patches of the built mesh",
                          scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                                     camera=dict(eye=dict(x=-2.6, y=-1.8, z=1.8))))
        return fig

    # ---------------- build ---------------- #
    def on_build(self, event):
        problems = self.app.case.validate()
        if problems:
            self.app.notify("error", "Cannot build: " + "; ".join(problems))
            return
        if self.app.simulation.running():
            self.app.notify("warning", "A test run (step 6) uses the mesh of this case: stop it first")
            return
        self.build_btn.disabled = True
        self.app.save(then=self.start_build)

    def start_build(self, ok):
        folder = self.app.folder
        if not ok:
            self.build_btn.disabled = False
            self.app.notify("error", "Build not started: the case has problems or invalid blocks (see step 1)")
            return
        with _BUILDING_LOCK:
            if folder in _BUILDING:
                self.app.notify("warning", "This case is already being built")
                self.build_btn.disabled = False
                return
            _BUILDING.add(folder)
        images = self.images_chk.value
        with self.log_lock:
            self.log_lines, self.log_shown = ["$ ./Allrun.pre" + (" --images" if images else "")], 0
        self.log_box.visible = True
        self.view3d.objects = []

        def run():
            try:
                code = run_allrun(folder, images=images, on_line=self._add_line)
                self._add_line("== exit code " + str(code))
            except Exception as e:  # noqa: BLE001
                self._add_line("== error: " + repr(e))
            finally:
                with _BUILDING_LOCK:
                    _BUILDING.discard(folder)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def _add_line(self, line):
        with self.log_lock:
            self.log_lines.append(line)

    def poll(self):
        '''Called periodically by the app: show new log lines, finish the build.'''
        with self.log_lock:
            if len(self.log_lines) != self.log_shown:
                self.log.object = "\n".join(self.log_lines[-400:])
                self.log_shown = len(self.log_lines)
        if self.thread is not None and not self.thread.is_alive():
            self.thread = None
            self.build_btn.disabled = False
            self.app.refresh_view(self)
