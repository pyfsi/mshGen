'''
step4_meshing.py: step 4 of the GUI, meshing (manual mode, M3)

Inputs:  cells and gradings of the edge parameters e00-e10, spanwise cells and grading per layer
Plots:   2D mesh preview of a slice (as blockMesh builds it) coloured by cell size, aspect ratio (slice/3D),
         neighbour size ratio or corner angle; cells along the airfoil surface (length, first cell height);
         wall-normal and wake distributions; spanwise cell size with jumps between layers
Numbers: estimated cell count, slice summary, measured cell sizes per edge parameter
Helpers (M4, ui/helpers.py): wall y+, chordwise sizes, wake, spanwise sizes, cell budget; each shows the
change and its predicted result before it is applied.
'''
import math

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper, LogColorMapper, Range1d
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from core import estimate, preview
from core.case import EDGE_DESCRIPTION, EDGE_NAMES
from core.estimate import graded_sizes, layer_types
from ui.common import bfig, bokeh, note, table

COLOR_BY = {"cell size (m)": ("size", "log"), "aspect ratio, slice": ("ar", "log"),
            "aspect ratio, 3D (with spanwise size)": ("ar3d", "log"),
            "size ratio to neighbours": ("ratio", "lin"), "smallest corner angle (deg)": ("angle", "lin")}
ZOOMS = ["Whole domain", "Airfoil", "Leading edge", "Trailing edge"]
PREVIEW_W, PREVIEW_H = 690, 520


class MeshPreview:
    '''2D mesh of a slice with cell colours, block outlines and hover.'''

    def __init__(self):
        self.x_range, self.y_range = Range1d(0, 1), Range1d(0, 1)
        f = figure(width=PREVIEW_W, height=PREVIEW_H, x_range=self.x_range, y_range=self.y_range,
                   tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom", output_backend="webgl")
        f.toolbar.logo = None
        f.xaxis.axis_label, f.yaxis.axis_label = "x (m)", "z (m)"
        self.src = ColumnDataSource(dict(xs=[], zs=[], value=[], block=[], size=[], ar=[], ar3d=[], ratio=[], angle=[]))
        self.mapper_log = LogColorMapper(palette=Viridis256, low=1e-3, high=1)
        self.mapper_lin = LinearColorMapper(palette=Viridis256, low=0, high=1)
        self.r_log = f.patches("xs", "zs", source=self.src, line_color="#555555", line_width=0.25, line_alpha=0.6,
                               fill_color={"field": "value", "transform": self.mapper_log})
        self.r_lin = f.patches("xs", "zs", source=self.src, line_color="#555555", line_width=0.25, line_alpha=0.6,
                               fill_color={"field": "value", "transform": self.mapper_lin}, visible=False)
        self.src_outline = ColumnDataSource(dict(xs=[], zs=[]))
        f.multi_line("xs", "zs", source=self.src_outline, color="black", line_width=1.4)
        f.add_tools(HoverTool(renderers=[self.r_log, self.r_lin],
                              tooltips=[("block", "@block"), ("cell size", "@size{0.0000e}"), ("aspect ratio", "@ar{0.0}"),
                                        ("aspect ratio 3D", "@ar3d{0.0}"), ("neighbour ratio", "@ratio{0.00}"),
                                        ("min angle", "@angle{0.0} deg")]))
        self.bar_log = ColorBar(color_mapper=self.mapper_log, width=12, location=(0, 0))
        self.bar_lin = ColorBar(color_mapper=self.mapper_lin, width=12, location=(0, 0), visible=False)
        f.add_layout(self.bar_log, "right")
        f.add_layout(self.bar_lin, "right")
        self.figure = f
        self.pane = pn.pane.Bokeh(f, sizing_mode="fixed")
        self.metrics = None

    def show(self, blocks, metrics, color_by, title):
        self.metrics = metrics
        key, scale = COLOR_BY[color_by]
        val = metrics[key]
        n = len(val)
        if n == 0:
            self.src.data = dict(xs=[], zs=[], value=[], block=[], size=[], ar=[], ar3d=[], ratio=[], angle=[])
            return
        self.src.data = dict(xs=list(metrics["xs"]), zs=list(metrics["zs"]), value=val, block=metrics["block"],
                             size=metrics["size"], ar=metrics["ar"], ar3d=metrics["ar3d"], ratio=metrics["ratio"],
                             angle=metrics["angle"])
        lo, hi = float(np.nanmin(val)), float(np.nanmax(val))
        if scale == "log":
            self.mapper_log.low, self.mapper_log.high = max(lo, 1e-12), max(hi, lo * 1.0001, 1e-12)
        else:
            self.mapper_lin.low, self.mapper_lin.high = lo, max(hi, lo + 1e-9)
        self.r_log.visible = self.bar_log.visible = scale == "log"
        self.r_lin.visible = self.bar_lin.visible = scale != "log"
        oxs, ozs = [], []
        for b in blocks:
            P = b["P"]
            ring = np.vstack((P[:, 0], P[-1, :], P[::-1, -1], P[0, ::-1]))
            oxs.append(list(ring[:, 0]))
            ozs.append(list(ring[:, 2]))
        self.src_outline.data = dict(xs=oxs, zs=ozs)
        self.figure.title.text = title

    def zoom(self, which, blocks, R, dz):
        xs = [b["P"][..., 0] for b in blocks]
        surf = preview.surface_cells(blocks)
        if which == "Whole domain" or surf is None:
            cx, cz, hx, hz = 0.5 * R, dz, 1.6 * R, 1.08 * R
        else:
            x = np.r_[surf["upper"]["x"], surf["lower"]["x"]]
            le, te = x.min(), x.max()
            chord = te - le
            zc = dz
            if which == "Airfoil":
                cx, hx = 0.5 * (le + te) + 0.1 * chord, 0.75 * chord
            elif which == "Leading edge":
                cx, hx = le + 0.05 * chord, 0.12 * chord
            else:
                cx, hx = te, 0.12 * chord
            # z centre from the block points near cx
            allp = np.concatenate([b["P"][:, :2, :].reshape(-1, 3) for b in blocks])
            near = allp[np.abs(allp[:, 0] - cx) < hx]
            cz = float(np.median(near[:, 2])) if len(near) else zc
            hz = 0.4 * hx
        ratio = (PREVIEW_W - 110) / (PREVIEW_H - 70)
        if hx / hz < ratio:
            hx = hz * ratio
        else:
            hz = hx / ratio
        self.x_range.start, self.x_range.end = cx - hx, cx + hx
        self.y_range.start, self.y_range.end = cz - hz, cz + hz
        del xs


class MeshingStep:
    title = "4. Meshing"

    def __init__(self, app):
        self.app = app
        self._updating = False
        self.edges = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, layout="fit_data_stretch",
                                          sizing_mode="stretch_width", height=380,
                                          editors={"edge": None, "controls": None,
                                                   "cells": {"type": "number", "min": 1, "step": 1},
                                                   "grad1": {"type": "number", "step": 0.1},
                                                   "grad2": {"type": "number", "step": 0.1}})
        self.layers = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, layout="fit_data_stretch",
                                           sizing_mode="stretch_width", height=240,
                                           editors={c: None for c in ("layer", "type", "y0", "y1", "first cell",
                                                                      "last cell", "jump to next")}
                                           | {"cells": {"type": "number", "min": 1, "step": 1},
                                              "grading": {"type": "number", "step": 0.1}})
        self.count = pn.pane.Markdown()
        self.jump_msg = pn.Column(sizing_mode="stretch_width")
        self.vertex_box = pn.Column(sizing_mode="stretch_width")

        self.slice_sel = pn.widgets.IntSlider(name="Slice", start=0, end=1, value=0, width=300)
        self.color_by = pn.widgets.Select(name="Colour by", options=list(COLOR_BY), value="cell size (m)", width=300)
        self.zoom = pn.widgets.RadioButtonGroup(options=ZOOMS, value="Airfoil", button_type="primary", width=520)
        self.auto = pn.widgets.Checkbox(name="update the preview after every change", value=True)
        self.update_btn = pn.widgets.Button(name="Update preview", button_type="light", width=150)
        self.preview = MeshPreview()
        self.preview_msg = pn.pane.Markdown()
        self.slice_info = pn.Column(sizing_mode="stretch_width")
        self.plots = pn.Column(sizing_mode="stretch_width")

        self.edges.on_edit(self.on_edge_edit)
        self.layers.on_edit(self.on_layer_edit)
        self.slice_sel.param.watch(lambda e: self.update_preview(set_range=True), "value")
        self.color_by.param.watch(lambda e: self.update_preview(), "value")
        self.zoom.param.watch(lambda e: self.update_preview(set_range=True), "value")
        self.update_btn.on_click(lambda e: self.update_preview(force=True))

        from ui.helpers import HelpersPanel
        self.helpers = HelpersPanel(self)
        self.inputs = pn.Column(
            self.helpers.layout,
            pn.pane.Markdown("### Edges (chordwise, wall-normal, wake)"),
            pn.pane.Markdown("_cells: number of cells; grad1: expansion ratio (last/first cell) on the airfoil side, "
                             "grad2: on the outer side. e09/e10: only the gradings are used (and only grad2, see "
                             "LIMITATIONS.md item 12); e08 only matters beyond the tip and in aileron layers._"),
            self.edges,
            pn.pane.Markdown("### Spanwise layers"), self.layers, self.jump_msg, self.count,
            pn.pane.Markdown("### Block vertices"), self.vertex_box,
            width=520, sizing_mode="fixed")
        self.right = pn.Column(pn.Row(self.slice_sel, self.color_by), self.zoom,
                               pn.Row(self.auto, self.update_btn), self.preview_msg, self.preview.pane,
                               self.slice_info, self.plots, sizing_mode="stretch_width")

    # ---------------- load / view ---------------- #
    def load(self):
        self._first = True
        self.helpers.load()

    def view(self):
        self.refresh(set_range=getattr(self, "_first", True))
        self._first = False
        return [pn.Row(self.inputs, self.right, sizing_mode="stretch_width")]

    def refresh(self, set_range=False):
        self._fill_tables()
        self.vertex_box.objects = [self.app.vertex_panel.view()]
        self.helpers.refresh()
        self.update_preview(set_range=set_range, force=True)

    def after_helper(self):
        '''A helper proposal was applied.'''
        self.app.changed()
        self._fill_tables()
        self.helpers.refresh()
        self.update_preview(force=True)

    def _fill_tables(self):
        c = self.app.case
        e = c.meshing.edges
        self._updating = True
        try:
            self.edges.value = pd.DataFrame([{"edge": k, "cells": int(e[k][0]), "grad1": float(e[k][1]),
                                              "grad2": float(e[k][2]), "controls": EDGE_DESCRIPTION[k]}
                                             for k in EDGE_NAMES])
            rows = []
            jumps = {r["slice"]: r for r in preview.layer_jumps(c)}
            s = c.blocks.slice_locations
            for k, t in enumerate(layer_types(c)):
                n, g = c.meshing.layer_resolution[k], c.meshing.layer_grading[k]
                first, last = graded_sizes(s[k + 1] - s[k], n, g)
                j = jumps.get(k + 1)
                rows.append({"layer": k, "type": t, "y0": s[k], "y1": s[k + 1], "cells": int(n), "grading": float(g),
                             "first cell": first, "last cell": last,
                             "jump to next": ("{:.2f}{}".format(j["ratio"], " !" if j["jump"] else "") if j else "")})
            self.layers.value = pd.DataFrame(rows)
            self.slice_sel.end = max(1, len(s) - 1)
            self.slice_sel.value = min(self.slice_sel.value, len(s) - 1)
        finally:
            self._updating = False
        jumps = [r for r in preview.layer_jumps(c) if r["jump"]]
        self.jump_msg.objects = ([note("Spanwise size jumps between layers (ratio > 1.3): " + ", ".join(
            "slice {} (y = {:.4g} m): {:.2f}".format(r["slice"], r["y"], r["ratio"]) for r in jumps), "warning")]
            if jumps else [])
        self.count.object = "**Estimated cells:** {:,}".format(estimate.cell_count(c))

    # ---------------- edits ---------------- #
    def _changed(self):
        self.app.changed()
        self._fill_tables()
        self.vertex_box.objects = [self.app.vertex_panel.view()]
        self.helpers.refresh()
        if self.auto.value:
            self.update_preview(force=True)

    def on_edge_edit(self, event):
        if self._updating:
            return
        c = self.app.case
        name = self.edges.value.iloc[int(event.row)]["edge"]
        n, g1, g2 = c.meshing.edges[name]
        try:
            val = float(event.value)
            if event.column == "cells":
                if val < 1 or val != int(val):
                    raise ValueError("cells must be a positive integer")
                n = int(val)
            elif val <= 0:
                raise ValueError("gradings must be positive")
            elif event.column == "grad1":
                g1 = val
            else:
                g2 = val
        except (TypeError, ValueError) as e:
            self.app.notify("warning", "Not applied: " + str(e))
            self._fill_tables()
            return
        c.meshing.edges = dict(c.meshing.edges)
        c.meshing.edges[name] = [n, g1, g2]
        self._changed()

    def on_layer_edit(self, event):
        if self._updating:
            return
        c = self.app.case
        k = int(event.row)
        try:
            val = float(event.value)
            if event.column == "cells":
                if val < 1 or val != int(val):
                    raise ValueError("cells must be a positive integer")
                res = list(c.meshing.layer_resolution)
                res[k] = int(val)
                c.meshing.layer_resolution = res
            else:
                if val <= 0:
                    raise ValueError("grading must be positive")
                grad = list(c.meshing.layer_grading)
                grad[k] = val
                c.meshing.layer_grading = grad
        except (TypeError, ValueError) as e:
            self.app.notify("warning", "Not applied: " + str(e))
            self._fill_tables()
            return
        self._changed()

    # ---------------- preview ---------------- #
    def update_preview(self, set_range=False, force=False):
        if self._updating and not force:
            return
        c = self.app.case
        sl = int(self.slice_sel.value)
        try:
            model = self.app.model()
            blocks = preview.slice_mesh(c, self.app.airfoil_file, sl, model)
            dy = preview.spanwise_size(c, sl)
            met = preview.cell_metrics(blocks, dy)
        except Exception as e:  # noqa: BLE001
            self.preview_msg.object = "**Cannot compute the preview:** " + repr(e)
            return
        self.preview_msg.object = ""
        types = layer_types(c)
        layer = min(sl, len(types) - 1)
        y = c.blocks.slice_locations[sl]
        self.preview.show(blocks, met, self.color_by.value,
                          "Slice {} at y = {:.4g} m (layer {}, {}): {:,} cells".format(sl, y, layer, types[layer],
                                                                                        len(met["size"])))
        if set_range:
            from core.blocks import radius
            from core.planform import section_params
            dz = section_params(c.wing_stations or [[0, 0, 1.0, 0, 0, 0]], y)[3]
            self.preview.zoom(self.zoom.value, blocks, radius(c), dz)
        summ = preview.summary(met)
        if dy is not None:
            summ["spanwise cell size at the slice (m)"] = dy
        self.slice_info.objects = [
            pn.pane.Markdown("**This slice**"),
            table([{"quantity": k, "value": v} for k, v in summ.items()], width=PREVIEW_W, sizing_mode="fixed"),
            pn.pane.Markdown("**Measured cell sizes per edge parameter (this slice)**"),
            table(preview.edge_stats(blocks), width=PREVIEW_W, sizing_mode="fixed")]
        self.plots.objects = self.make_plots(blocks)

    def make_plots(self, blocks):
        c = self.app.case
        out = []
        surf = preview.surface_cells(blocks)
        if surf:
            f = bfig("Cells along the airfoil surface (this slice): chordwise length", "x (m)", "cell length (m)",
                     height=260, equal=False, hover=[("x", "$x{0.000}"), ("length", "$y{0.0000}")])
            f.line(surf["upper"]["x"], surf["upper"]["length"], color="#1f77b4", legend_label="upper side")
            f.scatter(surf["upper"]["x"], surf["upper"]["length"], color="#1f77b4", size=4)
            f.line(surf["lower"]["x"], surf["lower"]["length"], color="#2ca02c", legend_label="lower side")
            f.scatter(surf["lower"]["x"], surf["lower"]["length"], color="#2ca02c", size=4)
            out.append(bokeh(f))
            jumps = [(s, surf[s]["max_jump"], surf[s]["jump_x"]) for s in ("upper", "lower")]
            out.append(pn.pane.Markdown("Largest chordwise size jump between neighbouring cells: " + "; ".join(
                "{} side {:.2f} at x = {:.3f} m".format(s, j, x) for s, j, x in jumps)
                + ("  **(> 1.3)**" if max(j for _, j, _ in jumps) > 1.3 else "")))
            f = bfig("First wall-normal cell height along the surface (e00)", "x (m)", "first cell height (m)",
                     height=220, equal=False, hover=[("x", "$x{0.000}"), ("height", "$y{0.000000}")])
            f.line(surf["upper"]["x"], surf["upper"]["first_height"], color="#1f77b4", legend_label="upper side")
            f.line(surf["lower"]["x"], surf["lower"]["first_height"], color="#2ca02c", legend_label="lower side")
            out.append(bokeh(f))
        # wall-normal and wake distributions (cells along radial / wake lines)
        by = {b["block"]: b["P"] for b in blocks}
        f = figure(title="Wall-normal and wake cell sizes (log scale)", height=260, sizing_mode="stretch_width",
                   y_axis_type="log", tools="pan,wheel_zoom,box_zoom,reset,save")
        f.toolbar.logo = None
        f.xaxis.axis_label, f.yaxis.axis_label = "distance from the start (m)", "cell size (m)"
        lines = []
        if 17 in by:
            lines.append(("wall-normal at the mid-chord vertex (upper, e00)", by[17][0, :], "#1f77b4"))
        if 20 in by:
            lines.append(("wall-normal at the leading edge (e00)", by[20][-1, :], "#9467bd"))
        if 14 in by and 13 in by:
            lines.append(("wake line, TE -> outlet (e02, e01)", np.vstack((by[14][:, 0], by[13][1:, 0])), "#d62728"))
        for name, pts, col in lines:
            d = np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1)
            s = np.cumsum(d) - 0.5 * d
            f.line(s, d, color=col, legend_label=name)
            f.scatter(s, d, color=col, size=3)
        if lines:
            out.append(bokeh(f))
        if c.dim == "3D":
            span = estimate.spanwise_table(c)
            f = figure(title="Spanwise cell size (dashed: slices)", height=240, sizing_mode="stretch_width",
                       y_axis_type="log", tools="pan,wheel_zoom,box_zoom,reset,save")
            f.toolbar.logo = None
            f.xaxis.axis_label, f.yaxis.axis_label = "y (m)", "spanwise cell size (m)"
            ys, sizes = [], []
            for r in span:
                n, g = r["cells"], r["grading"]
                ratio = g ** (1.0 / (n - 1)) if n > 1 else 1.0
                d = r["first cell (m)"] * ratio ** np.arange(n)
                ys += list(r["y0"] + np.cumsum(d) - d / 2)
                sizes += list(d)
            f.line(ys, sizes, color="#1f77b4")
            f.scatter(ys, sizes, color="#1f77b4", size=3)
            lo, hi = min(sizes), max(sizes)
            for y in c.blocks.slice_locations:
                f.line([y, y], [lo, hi], color="#999999", line_dash="dashed")
            out.append(bokeh(f))
        return out
