'''
step4_meshing.py: step 4 of the GUI, meshing

Inputs:    mesh level (COARSE / MEDIUM / FINE: presets of the targets), the targets (y+, chordwise and spanwise cell
           ranges, max growth ratio, max number of cells; editable), flow conditions (U, nu)
Optimiser: Optimise mesh (core/meshopt.py, docs/mesh_optimiser.md) computes the cells and gradings of all edges
           (e00-e10) and of the spanwise layers from the targets; the result table compares the mesh before and
           after with the targets. Undo restores the previous cells and gradings.
Tables:    edges e00-e10 and spanwise layers, editable (the plots follow)
Plots:     top view with the slices and the spanwise cell faces, spanwise cell size; 2D mesh preview of a slice
           (as blockMesh builds it) coloured by cell size, aspect ratio, neighbour size ratio or corner angle
Detailed analysis: every slice (quantities and warnings), measured cell sizes per edge parameter, cells along the
           airfoil, wall-normal and wake distributions of the selected slice
'''
import copy
import json

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper, LogColorMapper, Range1d
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from core import blocks as blocks_mod, estimate, meshopt, preview
from core.case import EDGE_DESCRIPTION, EDGE_NAMES
from core.estimate import graded_sizes, layer_types
from ui.common import bfig, bokeh, note, scaled
from ui.step3_blocks import SliceTopView

COLOR_BY = {"cell size (m)": ("size", "log"), "aspect ratio, slice": ("ar", "log"),
            "aspect ratio, 3D (with spanwise size)": ("ar3d", "log"),
            "size ratio to neighbours": ("ratio", "lin"), "smallest corner angle (deg)": ("angle", "lin")}
ZOOMS = ["Whole domain", "Airfoil", "Leading edge", "Trailing edge"]
PREVIEW_W, PREVIEW_H = 690, 520
OPTIMISE = "Optimise mesh"

OPTIMISER_TEXT = "\n".join([
    "**Fixed by the targets (constraints)**",
    "- y+: the median y+ of the first cells on the wing equals the target (e00); the first spanwise cell at the "
    "tip face gives the same y+.",
    "- Cells: chordwise per side and spanwise on the wing within their ranges, the total at most the maximum. The "
    "optimiser uses as much of the ranges as the cell budget allows.",
    "- Growth: neighbouring cells inside every edge and layer differ by at most the max growth ratio.",
    "",
    "**Sizing (rules)**",
    "- Leading and trailing edge: the chordwise cell size follows the local radius of curvature (fine where the "
    "wall turns fast); the TE cell is as small as the LE cell.",
    "- Wake: starts with the TE cell and stays as fine as the airfoil cells up to the near-wake end; the cells at "
    "the wake line are as thin as at the wall.",
    "- Spanwise: uniform on the wing, refined towards the tip.",
    "- Beyond the tip (inside the extension of the airfoil contour): e08 starts with the wall-normal first cell at "
    "the contour and grows to the e07 cells inside.",
    "",
    "**Quality (optimised on the exact block mesh of the root and tip slices)**",
    "- Non-orthogonality: mean (weighted towards the airfoil), the worst 1 % of the faces, faces above 65 deg.",
    "- Size jumps between neighbouring cells above the max growth (also across block boundaries).",
    "- Beyond the tip also inside the airfoil contour (weight 0.5), against the fit of the chordwise cells.",
    "",
    "Variables: the outer-side gradings (grad2 of e01-e07), the wake gradings e09/e10; beyond the tip also e08 and "
    "grad1 of e03-e07. Takes 0.5-2 minutes. Details: GUI/docs/mesh_optimiser.md."])


def _json(case):
    return json.dumps(case.to_dict(), sort_keys=True)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return "{:,}".format(int(v))
    if isinstance(v, float):
        return "{:.4g}".format(v)
    return str(v)


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
        self.pane = scaled(f)

    def show(self, blocks, metrics, color_by, title):
        key, scale = COLOR_BY[color_by]
        val = metrics[key]
        if len(val) == 0:
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
        surf = preview.surface_cells(blocks)
        if which == "Whole domain" or surf is None:
            cx, cz, hx, hz = 0.5 * R, dz, 1.6 * R, 1.08 * R
        else:
            x = np.r_[surf["upper"]["x"], surf["lower"]["x"]]
            le, te = x.min(), x.max()
            chord = te - le
            if which == "Airfoil":
                cx, hx = 0.5 * (le + te) + 0.1 * chord, 0.75 * chord
            elif which == "Leading edge":
                cx, hx = le + 0.05 * chord, 0.12 * chord
            else:
                cx, hx = te, 0.12 * chord
            allp = np.concatenate([b["P"][:, :2, :].reshape(-1, 3) for b in blocks])
            near = allp[np.abs(allp[:, 0] - cx) < hx]
            cz = float(np.median(near[:, 2])) if len(near) else dz
            hz = 0.4 * hx
        ratio = (PREVIEW_W - 110) / (PREVIEW_H - 70)
        if hx / hz < ratio:
            hx = hz * ratio
        else:
            hz = hx / ratio
        self.x_range.start, self.x_range.end = cx - hx, cx + hx
        self.y_range.start, self.y_range.end = cz - hz, cz + hz


# Edge parameter of each block edge in a slice: x-direction (inner and outer side), z-direction at p0 and p1
# (Mesh.generate_blocks; checked against its gradings in tests/test_m8.py)
X_PARAM = {0: "e03", 1: "e04", 2: "e04", 3: "e05", 4: "e05", 5: "e05", 6: "e05", 7: "e06", 8: "e06", 9: "e06",
           10: "e06", 11: "e07", 12: "e07", 13: "e01", 14: "e02", 15: "e03", 16: "e04", 17: "e05", 18: "e06",
           19: "e07", 20: "e07", 21: "e06", 22: "e05", 23: "e04", 24: "e03", 25: "e02", 26: "e01"}
Z_PARAM = {0: ("e03", "e03"), 1: ("e03", "e03"), 2: ("e03", "e03"), 3: ("e08", "e08"), 4: ("e07", "e07"),
           5: ("e07", "e07"), 6: ("e08", "e08"), 7: ("e08", "e08"), 8: ("e07", "e07"), 9: ("e07", "e07"),
           10: ("e08", "e08"), 11: ("e08", "e08"), 12: ("e08", "e08"), 13: ("e10", "e09"), 14: ("e00", "e10"),
           25: ("e10", "e00"), 26: ("e09", "e10")}
Z_PARAM.update({k: ("e00", "e00") for k in range(15, 25)})
EDGE_COLORS = dict(zip(EDGE_NAMES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                                    "#17becf", "#bcbd22", "#7f7f7f", "#000000"]))
# where each parameter is labelled: (block, side)
LABEL_AT = {"e00": (19, "left"), "e01": (13, "bottom"), "e02": (14, "bottom"), "e03": (15, "bottom"),
            "e04": (16, "bottom"), "e05": (17, "bottom"), "e06": (18, "bottom"), "e07": (19, "bottom"),
            "e08": (3, "left"), "e09": (13, "right"), "e10": (13, "left")}
EDGE_W, EDGE_H = 640, 520


class EdgePlot:
    """
    Block layout of a slice with every block edge coloured by its edge parameter and the cell nodes along it (the
    cells and grading as blockMesh places them). The blocks inside the airfoil contour exist only beyond the tip
    (dashed in the other slices). Hover an edge for its cells and grading; a row selected in the edges table is
    highlighted.
    """

    def __init__(self):
        self.x_range, self.y_range = Range1d(0, 1), Range1d(0, 1)
        f = figure(width=EDGE_W, height=EDGE_H, x_range=self.x_range, y_range=self.y_range,
                   tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom", output_backend="webgl")
        f.toolbar.logo = None
        f.xaxis.axis_label, f.yaxis.axis_label = "x (m)", "z (m)"
        self.src, self.src_dash, self.src_nodes, self.lines, self.dashes, self.dots = {}, {}, {}, {}, {}, {}
        hover = []
        for name in EDGE_NAMES:
            col = EDGE_COLORS[name]
            self.src[name] = ColumnDataSource(dict(xs=[], ys=[], info=[]))
            self.src_dash[name] = ColumnDataSource(dict(xs=[], ys=[], info=[]))
            self.src_nodes[name] = ColumnDataSource(dict(x=[], y=[]))
            self.lines[name] = f.multi_line("xs", "ys", source=self.src[name], color=col, line_width=1.6,
                                            legend_label=name)
            self.dashes[name] = f.multi_line("xs", "ys", source=self.src_dash[name], color=col, line_width=1.2,
                                             line_dash="dashed", line_alpha=0.6, legend_label=name)
            self.dots[name] = f.scatter("x", "y", source=self.src_nodes[name], color=col, size=2.5,
                                        legend_label=name)
            hover += [self.lines[name], self.dashes[name]]
        self.src_lab = ColumnDataSource(dict(x=[], y=[], text=[], color=[]))
        f.text("x", "y", text="text", source=self.src_lab, text_color="color", text_font_size="12px",
               text_font_style="bold", x_offset=4, y_offset=-4)
        f.add_tools(HoverTool(renderers=hover, tooltips=[("edge", "@info")], line_policy="nearest"))
        f.legend.orientation, f.legend.click_policy = "horizontal", "hide"
        f.legend.label_text_font_size, f.legend.spacing = "10px", 4
        f.add_layout(f.legend[0], "below")
        f.height = EDGE_H + 40
        self.figure = f
        self.pane = scaled(f)
        self.zoom = pn.widgets.RadioButtonGroup(options=["Whole domain", "Airfoil", "Trailing edge"],
                                                value="Whole domain", button_type="primary", width=380)
        self.blocks = None

    def show(self, case, blocks, active, slice_index, R, dz):
        """blocks: all 27 blocks of the slice (preview.slice_mesh with blocks=range(27)); active: block numbers."""
        e = case.meshing.edges
        data = {n: dict(xs=[], ys=[], info=[]) for n in EDGE_NAMES}
        dash = {n: dict(xs=[], ys=[], info=[]) for n in EDGE_NAMES}
        nodes = {n: dict(x=[], y=[]) for n in EDGE_NAMES}
        lab = dict(x=[], y=[], text=[], color=[])
        seen = set()
        for b in blocks:
            k, P = b["block"], b["P"][..., [0, 2]]
            sides = {"bottom": (P[:, 0], X_PARAM[k], "inner" if k >= 13 else ""),
                     "top": (P[:, -1], X_PARAM[k], "outer" if k >= 13 else ""),
                     "left": (P[0, :], Z_PARAM[k][0], ""), "right": (P[-1, :], Z_PARAM[k][1], "")}
            for side, (pts, name, where) in sides.items():
                key = tuple(np.round(np.r_[pts[0], pts[-1]], 6)), tuple(np.round(np.r_[pts[-1], pts[0]], 6))
                if (LABEL_AT[name] == (k, side)) and (k in active or name == "e08"):
                    m = pts[len(pts) // 2]
                    lab["x"].append(float(m[0]))
                    lab["y"].append(float(m[1]))
                    lab["text"].append(name)
                    lab["color"].append(EDGE_COLORS[name])
                if key[0] in seen or key[1] in seen:
                    continue
                seen.add(key[0])
                n, g1, g2 = e[name]
                g = g2 if (where == "outer" or name in ("e09", "e10")) else g1
                info = "{} ({}): {} cells, grading {:.4g}{}{}".format(
                    name, EDGE_DESCRIPTION[name], int(n), g, " (outer side, grad2)" if where == "outer" else "",
                    "" if k in active else "; block {} only beyond the tip".format(k))
                tgt = data if k in active else dash
                tgt[name]["xs"].append(list(pts[:, 0]))
                tgt[name]["ys"].append(list(pts[:, 1]))
                tgt[name]["info"].append(info)
                if k in active:
                    nodes[name]["x"] += list(pts[:, 0])
                    nodes[name]["y"] += list(pts[:, 1])
        for name in EDGE_NAMES:
            self.src[name].data, self.src_dash[name].data = data[name], dash[name]
            self.src_nodes[name].data = nodes[name]
        self.src_lab.data = lab
        self.blocks = blocks
        self.figure.title.text = "Slice {}: edges and cell nodes (dashed: only beyond the tip)".format(slice_index)
        self.R, self.dz = R, dz

    def set_zoom(self):
        if self.blocks is None:
            return
        allp = np.concatenate([b["P"][..., [0, 2]].reshape(-1, 2) for b in self.blocks])
        wall = np.concatenate([b["P"][:, 0][:, [0, 2]] for b in self.blocks if 15 <= b["block"] <= 24])
        le, te = wall[:, 0].min(), wall[:, 0].max()
        c = te - le
        if self.zoom.value == "Whole domain":
            cx, cz = 0.5 * (allp[:, 0].min() + allp[:, 0].max()), 0.5 * (allp[:, 1].min() + allp[:, 1].max())
            hx, hz = 0.52 * (allp[:, 0].max() - allp[:, 0].min()), 0.52 * (allp[:, 1].max() - allp[:, 1].min())
        elif self.zoom.value == "Airfoil":
            cx, cz, hx, hz = 0.5 * (le + te), float(np.median(wall[:, 1])), 0.7 * c, 0.3 * c
        else:
            cx, cz, hx, hz = te, float(wall[np.argmax(wall[:, 0]), 1]), 0.15 * c, 0.1 * c
        ratio = (EDGE_W - 60) / (EDGE_H - 60)
        if hx / hz < ratio:
            hx = hz * ratio
        else:
            hz = hx / ratio
        self.x_range.start, self.x_range.end = cx - hx, cx + hx
        self.y_range.start, self.y_range.end = cz - hz, cz + hz

    def highlight(self, names):
        for name in EDGE_NAMES:
            on = not names or name in names
            self.lines[name].glyph.line_width = 4.0 if names and name in names else 1.6
            self.lines[name].glyph.line_alpha = 1.0 if on else 0.15
            self.dashes[name].glyph.line_alpha = 0.6 if on else 0.1
            self.dots[name].glyph.fill_alpha = self.dots[name].glyph.line_alpha = 1.0 if on else 0.15


class MeshingStep:
    title = "4. Meshing"
    intro = 'Cells and gradings of every block edge and spanwise layer, optimised for the targets of the mesh level.'

    def __init__(self, app):
        self.app = app
        self._updating = False
        self._progress = None
        self.result = None              # last optimiser result (shown until the case is loaded again / undone)
        self.undo_state = None
        # level and targets
        self.level = pn.widgets.Select(name="Mesh level", options=meshopt.LEVELS, width=240)
        self.preset_btn = pn.widgets.Button(name="Reset to the level presets", button_type="light", width=240,
                                            align="end")       # level with the selected level, not its label
        F, I = pn.widgets.FloatInput, pn.widgets.IntInput
        self.tw = {
            "yplus": F(name="Target y+ (first cell centre)", start=0.01, step=1.0, width=240),
            "growth": F(name="Max growth ratio (neighbouring cells)", start=1.01, end=2.0, step=0.01, width=240),
            "chord_min": I(name="Chordwise cells per side: min", start=5, step=10, width=240),
            "chord_max": I(name="max", start=5, step=10, width=120),
            "span_min": I(name="Spanwise cells on the wing: min", start=1, step=10, width=240),
            "span_max": I(name="max", start=1, step=10, width=120),
            "max_cells": I(name="Max number of cells", start=1000, step=100000, width=240),
        }
        self.span_row = pn.Row(self.tw["span_min"], self.tw["span_max"])
        self.level.param.watch(self.on_level, "value")
        self.preset_btn.on_click(self.on_presets)
        for w in self.tw.values():
            w.param.watch(self.on_target, "value")
        # flow
        self.U = F(name="Free-stream velocity U (m/s)", step=1.0, start=0.0, width=240)
        self.nu = F(name="Kinematic viscosity nu (m2/s)", step=1e-6, start=0.0, width=240)
        self.U.param.watch(self.on_flow, "value")
        self.nu.param.watch(self.on_flow, "value")
        # optimiser
        self.opt_btn = pn.widgets.Button(name=OPTIMISE, button_type="primary", width=200)
        self.opt_btn.on_click(lambda e: self.on_optimise())
        self.opt_progress = pn.pane.Markdown()
        self.undo_btn = pn.widgets.Button(name="Undo (cells and gradings before the optimisation)",
                                          button_type="light", width=360)
        self.undo_btn.on_click(lambda e: self.on_undo())
        self.undo_box = pn.Column(sizing_mode="stretch_width")      # under the optimise button
        self.opt_result = pn.Column(sizing_mode="stretch_width")    # result table (detailed analysis)
        self.opt_card = pn.Card(pn.pane.Markdown(OPTIMISER_TEXT), title="What the optimiser does",
                                collapsed=True, sizing_mode="stretch_width")
        # tables
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
        self.layers_title = pn.pane.Markdown("### Spanwise layers")
        self.count = pn.pane.Markdown()
        self.jump_msg = pn.Column(sizing_mode="stretch_width")
        self.edges.on_edit(self.on_edge_edit)
        self.layers.on_edit(self.on_layer_edit)
        # preview
        self.slice_sel = pn.widgets.IntSlider(name="Slice", start=0, end=1, value=0, width=300)
        self.top = SliceTopView(height=230)
        self.span_plot = pn.Column(sizing_mode="stretch_width")
        self.color_by = pn.widgets.Select(name="Colour by", options=list(COLOR_BY), value="cell size (m)", width=300)
        self.zoom = pn.widgets.RadioButtonGroup(options=ZOOMS, value="Airfoil", button_type="primary", width=520)
        self.auto = pn.widgets.Checkbox(name="update the preview after every change", value=True, width=270,
                                        align="center")
        self.update_btn = pn.widgets.Button(name="Update preview", button_type="light", width=150, align="center")
        self.preview = MeshPreview()
        self.preview_msg = pn.pane.Markdown()
        self.slice_sel.param.watch(lambda e: self.update_preview(set_range=True, detail=False), "value")
        self.color_by.param.watch(lambda e: self.update_preview(detail=False), "value")
        self.zoom.param.watch(lambda e: self.update_preview(set_range=True, detail=False), "value")
        self.update_btn.on_click(lambda e: self.update_preview(force=True))
        # detailed analysis
        self.slices_box = pn.Column(sizing_mode="stretch_width")
        self.slice_detail = pn.Column(sizing_mode="stretch_width")
        self.plots = pn.Column(sizing_mode="stretch_width")

        self.inputs = pn.Column(
            pn.pane.Markdown("### Mesh level and targets"),
            pn.pane.Markdown("_The level sets the targets below (presets); they can be changed. **Optimise mesh** "
                             "computes all cells and gradings from them._"),
            pn.Row(self.level, self.preset_btn), self.tw["yplus"], self.tw["growth"],
            pn.Row(self.tw["chord_min"], self.tw["chord_max"]), self.span_row, self.tw["max_cells"],
            pn.pane.Markdown("### Flow conditions (for y+)"), pn.Row(self.U, self.nu),
            pn.Row(self.opt_btn, self.opt_progress), self.undo_box, self.opt_card,
            width=540, sizing_mode="fixed", margin=(5, 40, 5, 10))
        self.tables = pn.Column(
            pn.pane.Markdown("_cells: number of cells; grad1: expansion ratio (last/first cell) on the airfoil side, "
                             "grad2: on the outer side. e09/e10: only grad2 is used (LIMITATIONS.md item 12); e08: "
                             "across the outer part of the airfoil thickness, only beyond the tip. Editable: the "
                             "plots follow. Select a row to highlight the edge in the plot._"),
            self.edges, self.layers_title, self.layers, self.jump_msg, self.count,
            width=560, sizing_mode="fixed", margin=(5, 40, 5, 10))
        self.edge_plot = EdgePlot()
        self.edge_plot.zoom.param.watch(lambda e: self.edge_plot.set_zoom(), "value")
        self.edges.param.watch(self.on_edge_select, "selection")
        self.edge_section = pn.Column(
            pn.layout.Divider(), pn.pane.Markdown("## Edges and layers"),
            pn.Row(self.tables, pn.Column(self.edge_plot.zoom, self.edge_plot.pane, sizing_mode="stretch_width"),
                   sizing_mode="stretch_width"), sizing_mode="stretch_width")
        self.right = pn.Column(self.slice_sel, self.top.pane, pn.Row(self.color_by), self.zoom,
                               pn.Row(self.auto, self.update_btn), self.preview_msg, self.preview.pane,
                               sizing_mode="stretch_width")
        self.detail = pn.Column(
            pn.layout.Divider(), pn.pane.Markdown("## Detailed analysis"), self.opt_result,
            pn.pane.Markdown("**All slices** (exact block mesh; warnings for checkMesh limits, the targets and known "
                             "mshGen limitations)"), self.slices_box, self.span_plot,
            self.slice_detail, self.plots, sizing_mode="stretch_width")

    # ---------------- load / view ---------------- #
    def load(self):
        self._first = True
        self.result, self.undo_state = None, None
        self.opt_result.objects = []
        self.undo_box.objects = []

    def view(self):
        self.refresh(set_range=getattr(self, "_first", True))
        self._first = False
        return [pn.Row(self.inputs, self.right, sizing_mode="stretch_width"), self.edge_section, self.detail]

    def refresh(self, set_range=False):
        self._fill_setup()
        self._fill_tables()
        self.update_preview(set_range=set_range, force=True)

    def poll(self):
        '''Called periodically by the app: progress of the optimiser.'''
        if self._progress is not None:
            self.opt_progress.object = "_" + self._progress + "_"
            self.opt_progress.visible = True

    def _fill_setup(self):
        c = self.app.case
        t = meshopt.targets(c)
        self._updating = True
        try:
            self.level.value = c.domain.mesh_level if c.domain.mesh_level in meshopt.LEVELS else "COARSE"
            for k, w in self.tw.items():
                w.value = float(t[k]) if k in ("yplus", "growth") else int(t[k])
            self.U.value, self.nu.value = float(c.flow.U), float(c.flow.nu)
        finally:
            self._updating = False
        self.span_row.visible = c.dim == "3D"
        self.layers_title.visible = self.layers.visible = c.dim == "3D"

    # ---------------- targets, flow ---------------- #
    def on_level(self, event):
        if self._updating:
            return
        c = self.app.case
        c.domain.mesh_level = event.new
        c.meshing.targets = None
        self._fill_setup()
        self.app.changed()
        self.app.notify("info", "Targets set to the {} presets: Optimise mesh computes the cells".format(event.new))

    def on_presets(self, event):
        self.app.case.meshing.targets = None
        self._fill_setup()
        self.app.changed()

    def on_target(self, event):
        if self._updating:
            return
        c = self.app.case
        t = {k: (float(w.value) if k in ("yplus", "growth") else int(w.value)) for k, w in self.tw.items()
             if w.value is not None}
        c.meshing.targets = dict(meshopt.targets(c), **t)
        probs = meshopt.target_problems(c.meshing.targets, c.dim)
        if probs:
            self.app.notify("warning", "Targets: " + "; ".join(probs))
        self.app.changed()

    def on_flow(self, event):
        if self._updating:
            return
        c = self.app.case
        if self.U.value and self.U.value > 0:
            c.flow.U = float(self.U.value)
        if self.nu.value and self.nu.value > 0:
            c.flow.nu = float(self.nu.value)
        self.app.changed()
        if self.auto.value:
            self.update_detail()

    # ---------------- optimiser ---------------- #
    def on_optimise(self):
        from ui.step1_geometry import airfoil_data
        c, f = self.app.case, self.app.airfoil_file
        t = meshopt.targets(c)
        probs = meshopt.target_problems(t, c.dim) + c.validate()
        if probs:
            self.app.notify("warning", "Cannot optimise: " + "; ".join(probs))
            return
        key = _json(c)
        work = copy.deepcopy(c)
        airfoil = airfoil_data(c.airfoil.name, f)
        self.opt_btn.disabled, self.opt_btn.name = True, "Optimising ..."
        self._progress = "starting ..."

        def progress(text):
            self._progress = text

        def job():
            return meshopt.optimise(work, f, airfoil, t, progress=progress)

        def done(r, error):
            self.opt_btn.disabled, self.opt_btn.name = False, OPTIMISE
            self._progress = None
            self.opt_progress.object = "&nbsp;"
            self.opt_progress.visible = False
            if error:
                self.app.notify("error", "Optimisation failed: " + str(error))
                return
            if _json(self.app.case) != key:
                self.app.notify("warning", "The case changed during the optimisation: result discarded")
                return
            cc = self.app.case
            self.undo_state = (copy.deepcopy(cc.meshing.edges), list(cc.meshing.layer_resolution),
                               list(cc.meshing.layer_grading))
            cc.meshing.edges = {k: list(v) for k, v in r["edges"].items()}
            if r["layers"] is not None:
                cc.meshing.layer_resolution = [int(n) for n in r["layers"][0]]
                cc.meshing.layer_grading = [float(g) for g in r["layers"][1]]
            self.result = r
            bad = [row["metric"] for row in r["after"] if row["ok"] is False]
            self.app.notify("warning" if bad else "success",
                            "Mesh optimised (not saved yet)" + ("; not met: " + ", ".join(bad) if bad else ""))
            self.show_result()
            self.app.changed()
            self._fill_tables()
            self.update_preview(force=True)
        self.app.run_background(job, done)

    def show_result(self):
        r = self.result
        if r is None:
            self.opt_result.objects = []
            self.undo_box.objects = []
            return
        mark = {True: "ok", False: "NOT MET", None: ""}
        rows = [{"metric": a["metric"], "before": _fmt(b["value"]), "after": _fmt(a["value"]),
                 "target": a["target"], "": mark[a["ok"]]} for b, a in zip(r["before"], r["after"])]
        tab = pn.widgets.Tabulator(pd.DataFrame(rows), show_index=False, disabled=True, layout="fit_data_table",
                                   sizing_mode="stretch_width")
        self.opt_result.objects = [pn.pane.Markdown("**Optimisation result** (" + r["note"] + ")"), tab]
        bad = sum(a["ok"] is False for a in r["after"])
        self.undo_box.objects = [
            pn.pane.Markdown("_Optimised: {}. The result table (before / after / target) is in the detailed analysis "
                             "below._".format("all targets met" if not bad else "{} target(s) not met".format(bad))),
            self.undo_btn]

    def on_undo(self):
        if not self.undo_state:
            return
        c = self.app.case
        c.meshing.edges = self.undo_state[0]
        c.meshing.layer_resolution, c.meshing.layer_grading = self.undo_state[1], self.undo_state[2]
        self.undo_state, self.result = None, None
        self.opt_result.objects = []
        self.undo_box.objects = []
        self.app.notify("info", "Cells and gradings restored")
        self._changed()

    # ---------------- tables ---------------- #
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
                             "jump to next": ("{:.2f}".format(j["ratio"]) if j else "")})
            self.layers.value = pd.DataFrame(rows)
            self.slice_sel.end = max(1, len(s) - 1)
            self.slice_sel.value = min(self.slice_sel.value, len(s) - 1)
        finally:
            self._updating = False
        g = meshopt.targets(c)["growth"]
        jumps = [r for r in preview.layer_jumps(c) if r["ratio"] > g * 1.02]
        self.jump_msg.objects = ([note("Spanwise size jumps above the max growth {:g}: ".format(g) + ", ".join(
            "slice {} (y = {:.4g} m): {:.2f}".format(r["slice"], r["y"], r["ratio"]) for r in jumps), "warning")]
            if jumps else [])
        self.count.object = "**Cells:** {:,} (max {:,})".format(estimate.cell_count(c),
                                                                 int(meshopt.targets(c)["max_cells"]))

    def _changed(self):
        self.app.changed()
        self._fill_tables()
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

    def on_edge_select(self, event):
        df = self.edges.value
        names = [df.iloc[i]["edge"] for i in (event.new or []) if i < len(df)]
        self.edge_plot.highlight(names)

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
    def span_faces(self):
        c = self.app.case
        if c.dim != "3D":
            return None
        s = c.blocks.slice_locations
        faces = [s[0]]
        for k in range(len(s) - 1):
            faces += list(blocks_mod.cell_faces(s[k], s[k + 1], c.meshing.layer_resolution[k],
                                                c.meshing.layer_grading[k])[1:])
        return faces

    def update_preview(self, set_range=False, force=False, detail=True):
        if self._updating and not force:
            return
        c = self.app.case
        sl = int(self.slice_sel.value)
        try:
            self.top.update(c, sl, self.span_faces())
            self.span_plot.objects = self.spanwise_plot(sl)
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
        self.slice_detail.objects = [
            pn.pane.Markdown("**Slice {}: measured cell sizes per edge parameter**".format(sl)),
            pn.widgets.Tabulator(pd.DataFrame(preview.edge_stats(blocks)), show_index=False, disabled=True,
                                 layout="fit_data_table", sizing_mode="stretch_width")]
        self.plots.objects = self.make_plots(blocks, sl)
        try:
            from core.blockcheck import active_blocks
            from core.blocks import radius
            allb = preview.slice_mesh(c, self.app.airfoil_file, sl, model, blocks=list(range(27)))
            layer = min(sl, len(active_blocks(c)) - 1)
            first = self.edge_plot.blocks is None
            self.edge_plot.show(c, allb, set(active_blocks(c)[layer]), sl, radius(c), 0.0)
            if first or set_range:
                self.edge_plot.set_zoom()
        except Exception as e:  # noqa: BLE001
            self.preview_msg.object = "**Cannot draw the edges:** " + repr(e)
        if detail:
            self.update_detail()

    def update_detail(self):
        '''Table of all slices with warnings (exact block mesh of every slice, about 0.1 s per slice).'''
        c = self.app.case
        try:
            rows, warn = meshopt.slice_table(c, self.app.airfoil_file, self.app.model())
        except Exception as e:  # noqa: BLE001
            self.slices_box.objects = [note("Cannot analyse the slices: " + repr(e), "danger")]
            return
        df = pd.DataFrame([{k: (v if k == "quantity" else _fmt(v)) for k, v in r.items()} for r in rows])
        tab = pn.widgets.Tabulator(df, show_index=False, disabled=True, frozen_columns=["quantity"],
                                   layout="fit_data_table", sizing_mode="stretch_width",
                                   height=min(60 + 30 * len(rows), 600))
        items = [tab]
        head = {"danger": "Errors", "warning": "Warnings", "info": "Known mshGen limitations (information)"}
        for kind in ("danger", "warning", "info"):
            lines = [text for k, text in warn if k == kind]
            if lines:
                items.append(note("**{}**\n\n".format(head[kind]) + "\n".join("- " + x for x in lines), kind))
        if not warn:
            items.append(note("No warnings in any slice.", "success"))
        self.slices_box.objects = items

    def spanwise_plot(self, sl):
        c = self.app.case
        if c.dim != "3D":
            return []
        span = estimate.spanwise_table(c)
        f = figure(title="Spanwise cell size (dashed: slices, red: the selected slice)", height=240,
                   sizing_mode="stretch_width", y_axis_type="log", tools="pan,wheel_zoom,box_zoom,reset,save")
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
        for k, y in enumerate(c.blocks.slice_locations):
            f.line([y, y], [lo, hi], color="#de2d26" if k == sl else "#999999",
                   line_dash="solid" if k == sl else "dashed", line_width=2 if k == sl else 1)
        f.add_tools(HoverTool(tooltips=[("y", "$x{0.000}"), ("size", "$y{0.0000e}")]))
        return [pn.pane.Bokeh(f, sizing_mode="stretch_width")]

    def make_plots(self, blocks, sl):
        out = []
        surf = preview.surface_cells(blocks)
        if surf:
            f = bfig("Slice {}: chordwise cell length along the airfoil".format(sl), "x (m)", "cell length (m)",
                     height=260, equal=False, hover=[("x", "$x{0.000}"), ("length", "$y{0.0000}")])
            f.line(surf["upper"]["x"], surf["upper"]["length"], color="#1f77b4", legend_label="upper side")
            f.scatter(surf["upper"]["x"], surf["upper"]["length"], color="#1f77b4", size=4)
            f.line(surf["lower"]["x"], surf["lower"]["length"], color="#2ca02c", legend_label="lower side")
            f.scatter(surf["lower"]["x"], surf["lower"]["length"], color="#2ca02c", size=4)
            out.append(bokeh(f))
            jumps = [(s, surf[s]["max_jump"], surf[s]["jump_x"]) for s in ("upper", "lower")]
            g = meshopt.targets(self.app.case)["growth"]
            out.append(pn.pane.Markdown("Largest chordwise size jump between neighbouring cells: " + "; ".join(
                "{} side {:.2f} at x = {:.3f} m".format(s, j, x) for s, j, x in jumps)
                + ("  **(above the max growth {:g})**".format(g) if max(j for _, j, _ in jumps) > g * 1.02 else "")))
            f = bfig("Slice {}: first wall-normal cell height along the surface (e00)".format(sl), "x (m)",
                     "first cell height (m)", height=220, equal=False,
                     hover=[("x", "$x{0.000}"), ("height", "$y{0.000000}")])
            f.line(surf["upper"]["x"], surf["upper"]["first_height"], color="#1f77b4", legend_label="upper side")
            f.line(surf["lower"]["x"], surf["lower"]["first_height"], color="#2ca02c", legend_label="lower side")
            out.append(bokeh(f))
        by = {b["block"]: b["P"] for b in blocks}
        f = figure(title="Slice {}: wall-normal and wake cell sizes (log scale)".format(sl), height=260,
                   sizing_mode="stretch_width", y_axis_type="log", tools="pan,wheel_zoom,box_zoom,reset,save")
        f.toolbar.logo = None
        f.xaxis.axis_label, f.yaxis.axis_label = "distance from the start (m)", "cell size (m)"
        lines = []
        if 17 in by:
            lines.append(("wall-normal at the mid-chord vertex (upper, e00)", by[17][0, :], "#1f77b4"))
        if 20 in by:
            lines.append(("wall-normal at the leading edge (e00)", by[20][-1, :], "#9467bd"))
        if 14 in by and 13 in by:
            lines.append(("wake line, TE -> outlet (e02, e01)", np.vstack((by[14][:, 0], by[13][1:, 0])), "#d62728"))
            lines.append(("vertical at the near-wake end (e10)", by[14][-1, :], "#ff7f0e"))
        for name, pts, col in lines:
            d = np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1)
            s = np.cumsum(d) - 0.5 * d
            f.line(s, d, color=col, legend_label=name)
            f.scatter(s, d, color=col, size=3)
        if lines:
            out.append(bokeh(f))
        return out
