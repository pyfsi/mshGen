'''
step2_domain.py: step 2 of the GUI, domain definition

Inputs:  C-grid radius R/c (mesh_scale) and the mshGen level of the default settings, far field (symTip),
         flow conditions (U, nu) for Reynolds number and y+ estimates
Plots:   root plane (C-domain, airfoil, exterior vertices), 3D extent seen from above (planform, slices,
         patches); first estimate of the wall-normal first cell and its y+ (detailed tools in M4)
Fixed by mshGen (shown): wake length 2R, near-wake zone 0.5 x root chord, sizes based on the root chord.
'''
import math

import numpy as np
import panel as pn

from core import airfoils, blocks
from core.estimate import layer_types
from core.planform import place, section_params
from ui.common import COLORS, bfig, bokeh, note, table
from ui.step1_geometry import airfoil_data


class DomainStep:
    title = "2. Domain"

    def __init__(self, app):
        self.app = app
        self._updating = False
        self.scale = pn.widgets.FloatInput(name="C-grid radius R / root chord (mesh_scale)", step=0.5, start=0.6)
        self.level = pn.widgets.Select(name="mshGen default settings level (mesh_level)", options=airfoils.FMC)
        self.scale_info = pn.pane.Markdown()
        self.far = pn.widgets.FloatInput(name="Far field (symTip) at y (m)", step=0.5)
        self.far_info = pn.pane.Markdown()
        self.U = pn.widgets.FloatInput(name="Free-stream velocity U (m/s)", step=1.0, start=0.0)
        self.nu = pn.widgets.FloatInput(name="Kinematic viscosity nu (m2/s)", step=1e-6, start=0.0)
        self.flow_info = pn.Column()
        self.vertex_box = pn.Column(sizing_mode="stretch_width")
        self.plots = pn.Column(sizing_mode="stretch_width")
        self.far_box = pn.Column(pn.pane.Markdown("### Far field"), self.far, self.far_info)

        self.scale.param.watch(self.on_scale, "value")
        self.level.param.watch(self.on_level, "value")
        self.far.param.watch(self.on_far, "value")
        self.U.param.watch(self.on_flow, "value")
        self.nu.param.watch(self.on_flow, "value")

        self.inputs = pn.Column(
            pn.pane.Markdown("### C-grid"), self.scale, self.level, self.scale_info,
            self.far_box,
            pn.pane.Markdown("### Flow conditions"), self.U, self.nu, self.flow_info,
            pn.pane.Markdown("### Effect on the blocks"), self.vertex_box,
            width=430, sizing_mode="fixed")

    def load(self):
        c = self.app.case
        self._updating = True
        try:
            self.scale.value = float(c.domain.mesh_scale)
            self._levels()
            self.level.value = c.domain.mesh_level
            self.far.value = float(c.blocks.slice_locations[-1])
            self.U.value, self.nu.value = float(c.flow.U), float(c.flow.nu)
        finally:
            self._updating = False

    def view(self):
        self.render()
        return [pn.Row(self.inputs, self.plots, sizing_mode="stretch_width")]

    def refresh(self):
        self.load()
        self.render()

    def _levels(self):
        c = self.app.case
        used, levels, info = airfoils.default_levels(c.airfoil.name, c.domain.mesh_scale)
        opts = list(levels)
        if c.domain.mesh_level not in opts:
            opts.append(c.domain.mesh_level)        # keep the case's level visible (marked as unavailable)
        self.level.options = opts
        self.scale_info.object = ("_mshGen defaults for {}: {}._ Changing R/c moves the outer boundary: the exterior "
                                  "vertices follow (mshGen defaults) or must be moved (custom).").format(
                                      c.airfoil.name, info)

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
        self._updating = True
        try:
            self._levels()
            if c.domain.mesh_level not in airfoils.default_levels(c.airfoil.name, c.domain.mesh_scale)[1]:
                c.domain.mesh_level = "COARSE" if "COARSE" in self.level.options else self.level.options[0]
                self.level.value = c.domain.mesh_level
        finally:
            self._updating = False
        self.app.changed()
        self.render()

    def on_level(self, event):
        if self._updating:
            return
        self.app.case.domain.mesh_level = event.new
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

    def on_flow(self, event):
        if self._updating:
            return
        c = self.app.case
        if self.U.value and self.U.value > 0:
            c.flow.U = float(self.U.value)
        if self.nu.value and self.nu.value > 0:
            c.flow.nu = float(self.nu.value)
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
        est = blocks.first_cell_estimate(c)
        self.flow_info.objects = [table([{"quantity": k, "value": v} for k, v in est.items()]),
                                  pn.pane.Markdown("_First estimate from e00 ({} cells, expansion {:g}); y+ tools "
                                                   "follow in step 4 (M4)._".format(*c.meshing.edges["e00"][:2]))]
        self.vertex_box.objects = [self.app.vertex_panel.view(compact=True)]
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
        f = bfig("Root plane: C-domain (R = {:.4g} m = {:g} root chords), airfoil and exterior vertices".format(
            R, c.domain.mesh_scale), "x (m)", "z (m)", height=430, hover=[("x", "$x{0.000}"), ("z", "$y{0.000}")])
        th = np.linspace(math.pi / 2, 3 * math.pi / 2, 120)
        bx = np.r_[2 * R, R * np.cos(th), 2 * R, 2 * R]
        bz = dz0 + np.r_[R, R * np.sin(th), -R, R]
        f.line(bx, bz, color="black", line_width=2, legend_label="outer boundary (inlet, outlet)")
        x, z = place(a.outline(), *section_params(c.wing_stations or [[0, 0, 1.0, 0, 0, 0]], 0.0), c.cog)
        f.patch(x, z, fill_color=COLORS["upper"], line_color=COLORS["upper"], legend_label="root section")
        if c.blocks.exterior_points:
            ex = np.array(c.blocks.exterior_points)
            f.scatter(ex[:, 0], ex[:, 1] + dz0, size=7, color="purple", legend_label="exterior vertices 22-38")
            f.text(ex[:, 0], ex[:, 1] + dz0, text=[str(i) for i in range(22, 39)], text_font_size="9px",
                   x_offset=4, y_offset=-4)
            f.line([x.max(), ex[0, 0]], [z[np.argmax(x)], ex[0, 1] + dz0], color="purple", line_dash="dotdash",
                   legend_label="near wake (0.5 x root chord)")
        f.line([2 * R, 2 * R], [dz0 - R, dz0 + R], color="red", line_width=3, legend_label="outlet")
        out.append(bokeh(f))

        if c.dim == "3D" and c.wing_stations:
            st = c.wing_stations
            far = c.blocks.slice_locations[-1]
            f = bfig("Domain seen from above (flow downwards): planform, slices, patches", "y (m)", "x (m)", height=380)
            f.patch([0, far, far, 0], [-R, -R, 2 * R, 2 * R], fill_color="#f0f0f0", line_color="black",
                    legend_label="domain")
            ys = np.linspace(0, c.half_span, 60)
            le = [place(np.array([[0.0, 0.0]]), *section_params(st, y), c.cog)[0][0] for y in ys]
            te = [place(np.array([[1.0, 0.0]]), *section_params(st, y), c.cog)[0][0] for y in ys]
            f.patch(np.r_[ys, ys[::-1]], np.r_[le, te[::-1]], fill_color=COLORS["upper"], line_alpha=0,
                    legend_label="wing")
            for s in c.blocks.slice_locations:
                f.line([s, s], [-R, 2 * R], color=COLORS["slice"], line_dash="dashed")
            f.text([0.01 * far, far * 0.99, 0.5 * far, 0.5 * far], [0.5 * R, 0.5 * R, -1.02 * R, 2.02 * R],
                   text=["symRoot", "symTip", "inlet (C-front, top, bottom)", "outlet"],
                   text_align="center", text_font_size="11px", angle=[math.pi / 2, math.pi / 2, 0, 0])
            f.y_range.flipped = True
            out.append(bokeh(f))

        rows = [{"quantity": "x range (m): -R ... 2R (wake length 2R, fixed by mshGen)",
                 "value": "{:.4g} ... {:.4g}".format(-R, 2 * R)},
                {"quantity": "z range at the root (m)", "value": "{:.4g} ... {:.4g}".format(dz0 - R, dz0 + R)},
                {"quantity": "near-wake zone (fixed by mshGen)", "value": "0.5 x root chord = {:.4g} m".format(0.5 * c0)}]
        if c.dim == "3D":
            rows.append({"quantity": "spanwise extent (m)", "value": "0 ... {:.4g}".format(c.blocks.slice_locations[-1])})
        out.append(table(rows))
        return out
