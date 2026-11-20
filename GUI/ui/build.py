'''
build.py: step 5 of the GUI, build and check

Options:  Fluent export, images (mshGen plots + ParaView views), environment modules
Build:    save the case, block check, Allrun.pre with a live log
Results:  checkMesh summary, y+ per wall patch (measured on the mesh), patches, Fluent zones,
          3D view of the wall patches (Plotly), images
'''
import glob
import os
import threading

import numpy as np
import panel as pn
import plotly.graph_objects as go

from core.results import fluent_zones, measured_yplus, patches, wall_surfaces
from core.runner import checkmesh_summary, run_allrun
from ui.common import note, table
from ui.view3d import View3D

_BUILDING = set()            # case folders being built (shared by all browser sessions)
_BUILDING_LOCK = threading.Lock()
WALL_COLORS = {"wingUp": "#4682b4", "wing": "#4682b4", "wingDown": "#e6873a", "wingTip": "#999999",
               "wingCut": "#c0504d", "extra": "#9b59b6"}


class BuildStep:
    title = "5. Build & check"

    def __init__(self, app):
        self.app = app
        self._updating = False
        self.log_lines, self.log_shown = [], 0
        self.log_lock = threading.Lock()
        self.thread = None
        self.build_btn = pn.widgets.Button(name="Save and build mesh (Allrun.pre)", button_type="primary")
        self.images_chk = pn.widgets.Checkbox(name="also make images (mshGen plots + ParaView views)", value=False)
        self.fluent_chk = pn.widgets.Checkbox(name="export to Fluent (foamMeshToFluent)", value=True)
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

        self.yplus_btn = pn.widgets.Button(name="Measure y+ on the mesh", button_type="light")
        self.yplus_btn.on_click(lambda e: self.measure_yplus())
        self.yplus_view = pn.Column(sizing_mode="stretch_width")
        self.view3d_btn = pn.widgets.Button(name="Show 3D view of the walls", button_type="light")
        self.edges_chk = pn.widgets.Checkbox(name="cell edges", value=True)
        self.view3d_btn.on_click(lambda e: self.show_3d())
        self.view3d = pn.Column(sizing_mode="stretch_width")

    def load(self):
        c = self.app.case
        with self.log_lock:
            self.log_lines, self.log_shown = [], 0
        self.log.object = ""
        self.yplus_view.objects = []
        self.view3d.objects = []
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
        out = [pn.Row(self.build_btn, pn.Column(self.images_chk, self.fluent_chk)),
               pn.Card(self.mod_python, self.mod_openfoam, self.mod_paraview, title="Environment modules",
                       collapsed=True, sizing_mode="stretch_width"),
               self.log_box,
               pn.pane.Markdown("### Mesh (last build)"), self.checkmesh_view(),
               pn.Row(self.yplus_btn), self.yplus_view]
        pt = patches(app.folder)
        if pt:
            out += [pn.pane.Markdown("**Patches**"),
                    table([{"patch": n, "type": t, "faces": f} for n, t, f, _ in pt], width=500, sizing_mode="fixed")]
        out += self.fluent_view()
        out += [pn.pane.Markdown("### 3D view of the walls"),
                pn.Row(self.view3d_btn, self.edges_chk), self.view3d]
        imgs = sorted(glob.glob(os.path.join(app.folder, "images", "*.png")))
        if imgs:
            out.append(pn.pane.Markdown("### Images (last build with images)"))
            out.append(pn.FlexBox(*[pn.Column(pn.pane.Markdown("`" + os.path.basename(f) + "`"),
                                              pn.pane.PNG(f, width=440, sizing_mode="fixed"), sizing_mode="fixed",
                                              width=460) for f in imgs]))
        return out

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
        return pn.Column(alert, table(rows, width=500, sizing_mode="fixed"))

    def fluent_view(self):
        app = self.app
        msh = os.path.join(app.folder, "fluentInterface", app.case.name + ".msh")
        if not os.path.isfile(msh):
            return [pn.pane.Markdown("_No Fluent mesh (enable the export and build)._")]
        zones = fluent_zones(msh)
        notes = ["`fluentInterface/{}.msh` ({:.0f} MB), units m.".format(app.case.name, os.path.getsize(msh) / 1e6)]
        if any(z["name"] == "inlet" and z["type"] != "velocity-inlet" for z in zones):
            notes.append("foamMeshToFluent writes `inlet` as **{}**: set it to velocity-inlet in Fluent.".format(
                next(z["type"] for z in zones if z["name"] == "inlet")))
        s = checkmesh_summary(app.folder)
        if s and s.get("polyhedra"):
            notes.append("{} polyhedral cells (merged interface beyond the tip, LIMITATIONS.md item 9); Fluent "
                         "reads them.".format(s["polyhedra"]))
        return [pn.pane.Markdown("### Fluent mesh"), pn.pane.Markdown("  \n".join(notes)),
                table(zones, width=500, sizing_mode="fixed")]

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
                self.view3d.objects = [View3D(self.app, self.figure_3d(result, edges), height=620,
                                              slot="build").panel]
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

    # ---------------- y+ ---------------- #
    def measure_yplus(self):
        '''y+ per wall patch from constant/wallDistance (background, a few seconds for millions of cells).'''
        folder, case = self.app.folder, self.app.case
        self.yplus_btn.disabled, self.yplus_btn.name = True, "Measuring ..."

        def done(result, error):
            self.yplus_btn.disabled, self.yplus_btn.name = False, "Measure y+ on the mesh"
            if error:
                self.yplus_view.objects = [pn.pane.Markdown("Cannot measure y+: " + repr(error))]
            elif result is None:
                self.yplus_view.objects = [pn.pane.Markdown("_No mesh with wallDistance yet: build the mesh._")]
            else:
                self.yplus_view.objects = [
                    pn.pane.Markdown("**y+ on the walls** (first cell centre; flat-plate u_tau with the local chord, "
                                     "U = {:g} m/s, nu = {:g} m2/s)".format(case.flow.U, case.flow.nu)),
                    table([{"patch": k, **v} for k, v in result.items()], width=600, sizing_mode="fixed")]
        self.app.run_background(lambda: measured_yplus(folder, case), done)

    # ---------------- build ---------------- #
    def on_build(self, event):
        problems = self.app.case.validate()
        if problems:
            self.app.notify("error", "Cannot build: " + "; ".join(problems))
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
        self.yplus_view.objects = []
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
            self.measure_yplus()                        # y+ of the new mesh
