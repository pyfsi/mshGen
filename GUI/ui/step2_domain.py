'''
step2_domain.py: step 2 of the GUI, domain definition

Inputs:  C-grid radius R/c (mesh_scale), far field (symTip)
Plots:   root plane (C-domain and root section), 3D extent seen from above (planform, patches)
Fixed by mshGen (shown): wake length 2R, sizes based on the root chord.
The mshGen level (mesh_level) and the flow conditions are set in step 4 (meshing); the exterior block
vertices and the slices in step 3 (blocks).
'''
import math

import numpy as np
import panel as pn

from core import airfoils, blocks
from core.estimate import layer_types
from core.planform import place, section_params
from ui.common import COLORS, bfig, bokeh, note
from ui.step1_geometry import airfoil_data


class DomainStep:
    title = "2. Domain"

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

        self.inputs = pn.Column(
            pn.pane.Markdown("### C-grid"), self.scale, self.scale_info,
            self.far_box,
            width=430, sizing_mode="fixed")

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
        levels = airfoils.default_levels(c.airfoil.name, c.domain.mesh_scale)[1]
        if c.domain.mesh_level not in levels:       # the level must exist for the new R/c (mshGen defaults)
            old = c.domain.mesh_level
            c.domain.mesh_level = "COARSE" if "COARSE" in levels else levels[0]
            self.app.notify("info", "mshGen has no {} defaults for R/c = {:g}: level set to {} (step 4)".format(
                old, c.domain.mesh_scale, c.domain.mesh_level))
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
        info = airfoils.default_levels(c.airfoil.name, c.domain.mesh_scale)[2]
        self.scale_info.object = ("_mshGen defaults for {}: {}._ Changing R/c moves the outer boundary; the block "
                                  "vertices follow (see Blocks in the sidebar).").format(c.airfoil.name, info)
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
