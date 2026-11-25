'''
step2_domain.py: step 2 of the GUI, domain definition

Inputs:  C-grid radius R/c (mesh_scale), far field (symTip)
Plots:   root plane (C-domain and root section), 3D extent seen from above (planform, patches)
Fixed by mshGen (shown): wake length 2R, sizes based on the root chord.
The mesh level (mesh_level) and the flow conditions are set in step 4 (meshing); the exterior block
vertices in step 3b, the slices in step 3a.
'''
import copy
import json
import math

import numpy as np
import panel as pn

from core import blockopt, blocks, geometry, meshopt
from core.sizing import apply_proposal
from core.estimate import layer_types
from core.planform import place, section_params
from ui.common import COLORS, bfig, bokeh, note
from ui.step1_geometry import airfoil_data


MAKE = "Make mesh (slices, block vertices, cells)"


class DomainStep:
    title = "2. Domain"
    intro = 'The size of the C-grid and the far field; Make mesh does steps 3 and 4 in one go.'

    def __init__(self, app):
        self.app = app
        self._updating = False
        self.scale = pn.widgets.FloatInput(name="C-grid radius R / root chord (mesh_scale)", step=0.5, start=0.6)
        self.scale_info = pn.pane.Markdown()
        self.far = pn.widgets.FloatInput(name="Far field (symTip) at y (m)", step=0.5)
        self.far_info = pn.pane.Markdown()
        self.plots = pn.Column(sizing_mode="stretch_width")
        self.far_box = pn.Column(pn.pane.Markdown("### Far field"), self.far, self.far_info)

        self.scale.param.watch(self.on_scale, "value")
        self.far.param.watch(self.on_far, "value")
        self.make_btn = pn.widgets.Button(name=MAKE, button_type="primary", sizing_mode="stretch_width")
        self.make_btn.on_click(lambda e: self.on_make())
        self.make_info = pn.pane.Markdown(
            "_Does steps 3 and 4 for you: suggests the slices (3a), suggests the block vertices (3b) and optimises the "
            "cells and gradings for the targets of the mesh level (4). About 1 to 3 minutes; Undo (top right) reverts "
            "it. Check the result in steps 3 and 4, then build in step 5._", styles={"font-size": "13px"})
        self.make_progress = pn.pane.Markdown("", styles={"color": "#2c6e9b"})
        self._progress = None

        self.inputs = pn.Column(
            pn.pane.Markdown("### C-grid"), self.scale, self.scale_info,
            self.far_box,
            pn.pane.Markdown("### Blocking and meshing"), self.make_btn, self.make_progress, self.make_info,
            width=430, sizing_mode="fixed", margin=(5, 40, 5, 10))

    def load(self):
        c = self.app.case
        self._updating = True
        try:
            self.scale.value = float(c.domain.mesh_scale)
            self.far.value = float(c.blocks.slice_locations[-1])
        finally:
            self._updating = False

    def view(self):
        self.render()
        return [pn.Row(self.inputs, self.plots, sizing_mode="stretch_width")]

    def refresh(self):
        self.load()
        self.render()

    # ---------------- callbacks ---------------- #
    def on_scale(self, event):
        if self._updating or event.new is None:
            return
        c = self.app.case
        try:
            blocks.set_mesh_scale(c, event.new)
        except ValueError as e:
            self.app.notify("warning", str(e))
            return
        self.app.changed()
        self.render()

    def on_far(self, event):
        if self._updating or event.new is None:
            return
        try:
            blocks.set_far_field(self.app.case, float(event.new))
        except ValueError as e:
            self.app.notify("warning", str(e))
            return
        self.app.changed()
        self.render()

    # ---------------- make mesh (steps 3a, 3b, 4) ---------------- #
    def poll(self):
        if self._progress is not None:
            self.make_progress.object = "**" + self._progress + "**"

    def on_make(self):
        from ui.step1_geometry import airfoil_data
        app = self.app
        c, f = app.case, app.airfoil_file
        problems = c.validate() + meshopt.target_problems(meshopt.targets(c), c.dim)
        if problems:
            app.notify("warning", "Cannot make the mesh: " + "; ".join(problems))
            return
        key = json.dumps(c.to_dict(), sort_keys=True)
        work = copy.deepcopy(c)
        self.make_btn.disabled, self.make_btn.name = True, "Making the mesh ..."
        self._progress = "starting ..."
        notes = []

        def progress(text):
            self._progress = text

        def job():
            a = airfoil_data(work.airfoil.name, f)
            if work.dim == "3D":
                progress("1/4 suggesting the slices")
                blocks.suggest_slices(work)
            if geometry.vertex_state(work, f) == "outdated":
                progress("2/4 recomputing the block vertices (mshGen defaults)")
                geometry.recompute_vertices(work, f)
            progress("3/4 suggesting the block vertices (about 40 s)")
            try:
                r = blockopt.suggest(work, a, f)
                apply_proposal(work, {"interior_points": r["interior_points"],
                                      "exterior_points": r["exterior_points"]})
            except ValueError as e:
                notes.append("block vertices kept (" + str(e) + ")")
            progress("4/4 optimising the cells and gradings")
            r = meshopt.optimise(work, f, a, meshopt.targets(work),
                                 progress=lambda t: progress("4/4 optimising the cells: " + str(t)))
            work.meshing.edges = {k: list(v) for k, v in r["edges"].items()}
            if r["layers"] is not None:
                work.meshing.layer_resolution = [int(n) for n in r["layers"][0]]
                work.meshing.layer_grading = [float(g) for g in r["layers"][1]]
            return [row["metric"] for row in r["after"] if row["ok"] is False]

        def done(bad, error):
            self.make_btn.disabled, self.make_btn.name = False, MAKE
            self._progress = None
            self.make_progress.object = ""
            if error:
                app.notify("error", "Make mesh failed: " + str(error))
                return
            if json.dumps(app.case.to_dict(), sort_keys=True) != key:
                app.notify("warning", "The case changed meanwhile: the result was discarded")
                return
            app.apply_case(work)
            msg = "Mesh made (not saved yet): check steps 3 and 4, then build in step 5"
            extra = notes + (["not met: " + ", ".join(bad)] if bad else [])
            app.notify("warning" if bad else "success", msg + ("; " + "; ".join(extra) if extra else ""))
        app.run_background(job, done)

    # ---------------- view ---------------- #
    def render(self):
        c = self.app.case
        self.far_box.visible = c.dim == "3D"
        b2 = c.half_span
        if c.dim == "3D" and b2:
            flow = sum(t == "flow" for t in layer_types(c))
            self.far_info.object = ("= {:.3g} x half-span; {} layer(s) beyond the tip.".format(c.blocks.slice_locations[-1] / b2, flow)
                                    if flow else "_No slices beyond the tip: the domain ends at the last wing slice "
                                                 "(symTip cuts the wing)._")
        self.scale_info.object = ("_Changing R/c moves the outer boundary; the block vertices follow (see step 3b, "
                                  "Airfoil blocking). Optimise the cells again in step 4 afterwards, or use Make mesh._")
        try:
            self.plots.objects = self.make_plots()
        except Exception as e:  # noqa: BLE001
            self.plots.objects = [note("Cannot draw the domain: " + repr(e), "danger")]

    def make_plots(self):
        c = self.app.case
        R = blocks.radius(c)
        c0, beta0, dx0, dz0 = c.root_section()
        a = airfoil_data(c.airfoil.name, self.app.airfoil_file)
        out = []

        # Root plane
        f = bfig("Root plane: C-domain (R = {:.4g} m = {:g} root chords) and root section".format(
            R, c.domain.mesh_scale), "x (m)", "z (m)", height=430, hover=[("x", "$x{0.000}"), ("z", "$y{0.000}")])
        th = np.linspace(math.pi / 2, 3 * math.pi / 2, 120)
        bx = np.r_[2 * R, R * np.cos(th), 2 * R, 2 * R]
        bz = dz0 + np.r_[R, R * np.sin(th), -R, R]
        f.line(bx, bz, color="black", line_width=2, legend_label="outer boundary (inlet, outlet)")
        x, z = place(a.outline(), *section_params(c.wing_stations or [[0, 0, 1.0, 0, 0, 0]], 0.0), c.cog)
        f.patch(x, z, fill_color=COLORS["upper"], line_color=COLORS["upper"], legend_label="root section")
        f.line([2 * R, 2 * R], [dz0 - R, dz0 + R], color="red", line_width=3, legend_label="outlet")
        out.append(bokeh(f))

        if c.dim == "3D" and c.wing_stations:
            st = c.wing_stations
            far = c.blocks.slice_locations[-1]
            f = bfig("Domain seen from above (flow downwards): planform and patches", "y (m)", "x (m)", height=380)
            f.patch([0, far, far, 0], [-R, -R, 2 * R, 2 * R], fill_color="#f0f0f0", line_color="black",
                    legend_label="domain")
            ys = np.linspace(0, c.half_span, 60)
            le = [place(np.array([[0.0, 0.0]]), *section_params(st, y), c.cog)[0][0] for y in ys]
            te = [place(np.array([[1.0, 0.0]]), *section_params(st, y), c.cog)[0][0] for y in ys]
            f.patch(np.r_[ys, ys[::-1]], np.r_[le, te[::-1]], fill_color=COLORS["upper"], line_alpha=0,
                    legend_label="wing")
            f.text([0.01 * far, far * 0.99, 0.5 * far, 0.5 * far], [0.5 * R, 0.5 * R, -1.02 * R, 2.02 * R],
                   text=["symRoot", "symTip", "inlet (C-front, top, bottom)", "outlet"],
                   text_align="center", text_font_size="11px", angle=[math.pi / 2, math.pi / 2, 0, 0])
            f.y_range.flipped = True
            out.append(bokeh(f))
        return out
