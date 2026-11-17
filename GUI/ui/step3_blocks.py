'''
step3_blocks.py: step 3 of the GUI, blocks of the domain

Inputs:  slices (add at a cell face / move / delete / suggest), chordwise divisions of the airfoil,
         all airfoil vertices (advanced), exterior vertices (table, or drag them in the plot),
         aileron cut-out layers, patches (SPLIT, ARTIFICIOUS)
Plot:    block layout of a selected slice as mshGen generates it: blocks coloured by status (valid / skewed /
         folded), block numbers and hover info, edge names e00-e10, airfoil, vertices; zoom presets
Edits of vertices make them "custom" (not recomputed automatically; the vertex panel can reset them).
'''
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, PointDrawTool, Range1d
from bokeh.plotting import figure

from core import blocks
from core.blockcheck import BLOCKS_2D
from core.estimate import layer_types
from core.planform import place, section_params
from ui.common import note
from ui.step1_geometry import airfoil_data

GROUP_COLOR = {"wake": "#c6dbef", "upper": "#9ecae1", "lower": "#a1d99b", "interior": "#fdd0a2"}
STATUS_COLOR = {"skewed": "#fd8d3c", "folded": "#de2d26", "inverted": "#a50f15"}
EDGE_LABELS = [("e03", 0, 1), ("e04", 1, 4), ("e05", 4, 7), ("e06", 7, 10), ("e07", 10, 13),
               ("e00", 7, 29), ("e01", 22, 23), ("e02", 0, 22), ("e09", 23, 24), ("e10", 22, 25)]
PLOT_W, PLOT_H = 690, 580
AIRFOIL_LABELS = ("e02", "e03", "e04", "e05", "e06", "e07", "e08")   # shown only when zoomed in


def block_group(k):
    if k < 13:
        return "interior"
    if k in (13, 14, 25, 26):
        return "wake"
    return "upper" if k <= 19 else "lower"


def editors_readonly(columns):
    return {c: None for c in columns}


class BlockPlot:
    '''Block layout of one slice; exterior vertices can be dragged (kept on their boundary).'''

    def __init__(self, step):
        self.step = step
        self._updating = False
        self._ext_prev = None
        self.slice = 0
        self.zoom = "Whole domain"
        self.x_range, self.y_range = Range1d(0, 1), Range1d(0, 1)
        f = figure(width=PLOT_W, height=PLOT_H, x_range=self.x_range, y_range=self.y_range,
                   tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom")
        f.toolbar.logo = None
        f.xaxis.axis_label, f.yaxis.axis_label = "x (m)", "z (m)"
        self.src_blocks = ColumnDataSource(dict(xs=[], ys=[], color=[], block=[], status=[], angles=[], group=[]))
        r_blocks = f.patches("xs", "ys", source=self.src_blocks, fill_color="color", fill_alpha=0.65,
                             line_color="#333333", line_width=0.7)
        f.add_tools(HoverTool(renderers=[r_blocks], tooltips=[("block", "@block (@group)"), ("status", "@status"),
                                                               ("corner angles", "@angles")]))
        self.src_outer = ColumnDataSource(dict(x=[], z=[]))
        f.line("x", "z", source=self.src_outer, color="black", line_width=1.5)
        self.src_airfoil = ColumnDataSource(dict(x=[], z=[]))
        self.r_airfoil = f.patch("x", "z", source=self.src_airfoil, fill_color="#444444", line_color="#222222")
        self.src_int = ColumnDataSource(dict(x=[], z=[], id=[]))
        f.scatter("x", "z", source=self.src_int, size=5, color="black")
        self.r_int_text = f.text("x", "z", text="id", source=self.src_int, text_font_size="10px", x_offset=3,
                                 y_offset=-3, text_color="black")
        self.src_ext = ColumnDataSource(dict(x=[], z=[], id=[]))
        r_ext = f.scatter("x", "z", source=self.src_ext, size=11, color="purple", alpha=0.8)
        f.text("x", "z", text="id", source=self.src_ext, text_font_size="11px", x_offset=6, y_offset=-6,
               text_color="purple")
        f.add_tools(HoverTool(renderers=[r_ext], tooltips=[("exterior vertex", "@id"), ("x", "@x{0.000}"),
                                                            ("z", "@z{0.000}")]))
        draw = PointDrawTool(renderers=[r_ext], add=False, description="Drag exterior vertices")
        f.add_tools(draw)
        self.src_labels = ColumnDataSource(dict(x=[], z=[], text=[]))
        f.text("x", "z", text="text", source=self.src_labels, text_font_size="12px", text_font_style="bold",
               text_color="#08519c", text_align="center", y_offset=-6)
        self.src_ext.on_change("data", self.on_drag)
        self.figure = f
        self.pane = pn.pane.Bokeh(f, sizing_mode="fixed")

    # ---------------- data ---------------- #
    def update(self, set_range=False):
        app = self.step.app
        c = app.case
        lay = app.layout()
        v = lay["vertices"]
        sl = min(self.slice, len(v) - 1)
        layer = min(sl, len(lay["active"]) - 1)
        y = c.blocks.slice_locations[sl]
        types = layer_types(c)
        xz = v[sl][:, [0, 2]]
        dz = section_params(c.wing_stations or [[0, 0, 1.0, 0, 0, 0]], y)[3]
        xs, ys, col, blk, stat, ang, grp = [], [], [], [], [], [], []
        for k in lay["active"][layer]:
            st, amin, amax = lay["status"][(layer, k, sl)]
            q = xz[BLOCKS_2D[k]]
            xs.append(list(q[:, 0]))
            ys.append(list(q[:, 1]))
            col.append(STATUS_COLOR.get(st, GROUP_COLOR[block_group(k)]))
            blk.append(k)
            stat.append(st)
            ang.append("{:.0f}-{:.0f} deg".format(amin, amax))
            grp.append(block_group(k))
        R = blocks.radius(c)
        th = np.linspace(np.pi / 2, 3 * np.pi / 2, 120)
        a = airfoil_data(c.airfoil.name, app.airfoil_file)
        ax, az = place(a.outline(), *section_params(c.wing_stations or [[0, 0, 1.0, 0, 0, 0]], y), c.cog)
        flow = types[layer] == "flow"
        labels = [(n, i, j) for n, i, j in EDGE_LABELS] + ([("e08", 4, 16)] if flow else [])
        if self.zoom == "Whole domain":
            labels = [lab for lab in labels if lab[0] not in AIRFOIL_LABELS]
        self._updating = True
        try:
            self.src_blocks.data = dict(xs=xs, ys=ys, color=col, block=blk, status=stat, angles=ang, group=grp)
            self.src_outer.data = dict(x=list(np.r_[2 * R, R * np.cos(th), 2 * R, 2 * R]),
                                       z=list(dz + np.r_[R, R * np.sin(th), -R, R]))
            self.src_airfoil.data = dict(x=list(ax), z=list(az))
            self.r_airfoil.glyph.fill_alpha = 0.0 if flow else 1.0
            self.src_int.data = dict(x=list(xz[:16, 0]), z=list(xz[:16, 1]), id=[str(i) for i in range(16)])
            self.src_ext.data = dict(x=list(xz[22:, 0]), z=list(xz[22:, 1]), id=[str(i) for i in range(22, 39)])
            self._ext_prev = (np.array(xz[22:, 0]), np.array(xz[22:, 1]), dz)
            self.src_labels.data = dict(x=[0.5 * (xz[i, 0] + xz[j, 0]) for _, i, j in labels],
                                        z=[0.5 * (xz[i, 1] + xz[j, 1]) for _, i, j in labels],
                                        text=[n for n, _, _ in labels])
            self.r_int_text.visible = self.zoom != "Whole domain"
        finally:
            self._updating = False
        self.figure.title.text = "Slice {} at y = {:.4g} m: blocks of layer {} ({}), as mshGen generates them".format(
            sl, y, layer, types[layer])
        if set_range:
            self._apply_range(R, dz, ax, az)

    def set_zoom(self, zoom):
        self.zoom = zoom
        self.update(set_range=True)

    def _apply_range(self, R, dz, ax, az):
        zoom = self.zoom
        if zoom == "Whole domain":
            cx, cz, hx, hz = 0.5 * R, dz, 1.6 * R, 1.08 * R
        else:
            chord = ax.max() - ax.min()
            if zoom == "Airfoil":
                cx, hx = 0.5 * (ax.min() + ax.max()) + 0.2 * chord, 0.8 * chord
            else:                                           # trailing edge
                cx, hx = ax.max() + 0.15 * chord, 0.45 * chord
            cz, hz = 0.5 * (az.min() + az.max()), 0.25 * chord
        ratio = (PLOT_W - 60) / (PLOT_H - 70)
        if hx / hz < ratio:
            hx = hz * ratio
        else:
            hz = hx / ratio
        self.x_range.start, self.x_range.end = cx - hx, cx + hx
        self.y_range.start, self.y_range.end = cz - hz, cz + hz
        self.r_int_text.visible = zoom != "Whole domain"

    # ---------------- drag ---------------- #
    def on_drag(self, attr, old, new):
        if self._updating or self._ext_prev is None:
            return
        px, pz, dz = self._ext_prev
        x, z = np.asarray(new.get("x", []), float), np.asarray(new.get("z", []), float)
        if len(x) != len(px):                       # a point was deleted (backspace): restore
            self.update()
            return
        moved = np.where(np.hypot(x - px, z - pz) > 1e-9 * (1 + np.abs(px).max()))[0]
        if len(moved) == 0:
            return
        c = self.step.app.case
        for i in moved:
            blocks.move_exterior(c, 22 + int(i), float(x[i]), float(z[i]) - dz)
        self.step.app.changed()
        self.step.refresh(plot=True, tables=True)


class BlocksStep:
    title = "3. Blocks"

    def __init__(self, app):
        self.app = app
        self._updating = False

        # Slices
        self.slices = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=230, selectable="checkbox",
                                           layout="fit_data_stretch", sizing_mode="stretch_width",
                                           editors=editors_readonly(["slice", "layer above", "cells", "grading"]))
        self.add_y = pn.widgets.FloatInput(name="New slice near y (m)", step=0.1, width=180)
        self.add_btn = pn.widgets.Button(name="Add slice", button_type="light", align="end")
        self.del_btn = pn.widgets.Button(name="Delete selected", button_type="light", align="end")
        self.suggest_btn = pn.widgets.Button(name="Suggest slices", button_type="light", align="end")
        self.slice_msg = pn.pane.Markdown(
            "_Edit y in the table to move a slice (root and tip are fixed). New slices are placed at the nearest "
            "cell face, so the cells do not change. Spanwise cells and grading: step 4._")

        # Chordwise divisions
        self.div_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, layout="fit_data_stretch",
                                              sizing_mode="stretch_width", editors=editors_readonly(["division"]))
        self.link = pn.widgets.Checkbox(name="same position on both sides", value=True)
        self.vert_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=300,
                                               layout="fit_data_stretch", sizing_mode="stretch_width",
                                               editors=editors_readonly(["vertex", "spline", "role"]))
        self.ext_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=300,
                                              layout="fit_data_stretch", sizing_mode="stretch_width",
                                              editors=editors_readonly(["vertex", "boundary"]))

        # Cut-outs and patches
        self.aileron = pn.widgets.MultiChoice(name="Aileron cut-out in layers (consecutive)", options=[])
        self.split = pn.widgets.Checkbox(name="Wing surface as wingUp / wingDown (SPLIT)")
        self.artificious = pn.widgets.Checkbox(name="First wing layers as separate patch 'extra' (ARTIFICIOUS)")
        self.skip = pn.widgets.IntInput(name="Layers in 'extra'", start=1, step=1, width=150)

        self.vertex_box = pn.Column(sizing_mode="stretch_width")

        # Plot
        self.plot = BlockPlot(self)
        self.slice_sel = pn.widgets.IntSlider(name="Slice", start=0, end=1, value=0, width=300)
        self.zoom = pn.widgets.RadioButtonGroup(options=["Whole domain", "Airfoil", "Trailing edge"],
                                                value="Whole domain", button_type="primary", width=400)
        self.plot_msg = pn.pane.Markdown()
        legend = pn.pane.HTML(
            "".join('<span style="background:{};padding:2px 8px;margin-right:6px;border:1px solid #999">{}</span>'.format(c, t)
                    for t, c in [("wake", GROUP_COLOR["wake"]), ("upper side", GROUP_COLOR["upper"]),
                                 ("lower side", GROUP_COLOR["lower"]), ("inside contour (flow layers)", GROUP_COLOR["interior"]),
                                 ("skewed", STATUS_COLOR["skewed"]), ("folded", STATUS_COLOR["folded"])])
            + "<br><small>Drag the purple exterior vertices with the point-draw tool (toolbar); they stay on their "
              "boundary. Hover a block for its number and corner angles.</small>")

        self.slices.on_edit(self.on_slice_edit)
        self.add_btn.on_click(self.on_add)
        self.del_btn.on_click(self.on_delete)
        self.suggest_btn.on_click(self.on_suggest)
        self.div_table.on_edit(self.on_div_edit)
        self.vert_table.on_edit(self.on_vert_edit)
        self.ext_table.on_edit(self.on_ext_edit)
        self.aileron.param.watch(self.on_aileron, "value")
        for w in (self.split, self.artificious, self.skip):
            w.param.watch(self.on_patches, "value")
        self.slice_sel.param.watch(lambda e: self.on_slice_sel(), "value")
        self.zoom.param.watch(lambda e: self.plot.set_zoom(e.new), "value")

        self.inputs = pn.Column(
            pn.pane.Markdown("### Slices"), self.slices,
            pn.Row(self.add_y, self.add_btn, self.del_btn), self.suggest_btn, self.slice_msg,
            pn.pane.Markdown("### Chordwise divisions of the airfoil"),
            pn.pane.Markdown("_Positions along the splines (0 = LE, 1 = TE, about x/c). They split the airfoil "
                             "surface into the edges e03 (TE) ... e07 (LE)._"),
            self.div_table, self.link,
            pn.Card(self.vert_table, title="All airfoil vertices (advanced)", collapsed=True),
            pn.Card(pn.pane.Markdown("_x, z in the root plane (m, origin = COG; other slices add dz). Vertices "
                                     "on the outer boundary stay on it._"), self.ext_table,
                    title="Exterior vertices (table)", collapsed=True),
            pn.pane.Markdown("### Cut-outs and patches"), self.aileron,
            pn.pane.Markdown("_Aileron cut position: fixed by mshGen per wing name (MegAWES: 70 % chord on the "
                             "suction side, cut normal tilted 20 deg); see LIMITATIONS.md item 4._"),
            self.split, pn.Row(self.artificious, self.skip),
            pn.pane.Markdown("### Block vertices"), self.vertex_box,
            width=470, sizing_mode="fixed")
        self.right = pn.Column(self.slice_sel, self.zoom, self.plot_msg, self.plot.pane, legend,
                               sizing_mode="stretch_width")

    # ---------------- load / refresh ---------------- #
    def load(self):
        self.plot.slice = 0
        self.plot.zoom = "Whole domain"
        self._updating = True
        try:
            self.slice_sel.value = 0
            self.zoom.value = "Whole domain"
        finally:
            self._updating = False
        self._first_view = True

    def view(self):
        self.refresh(plot=True, tables=True, set_range=getattr(self, "_first_view", True))
        self._first_view = False
        return [pn.Row(self.inputs, self.right, sizing_mode="stretch_width")]

    def refresh(self, plot=True, tables=True, set_range=False):
        c = self.app.case
        if tables:
            self._fill_tables()
        self.vertex_box.objects = [self.app.vertex_panel.view()]
        if plot:
            try:
                self.plot.update(set_range=set_range)
                self.plot_msg.object = ""
            except Exception as e:  # noqa: BLE001
                self.plot_msg.object = "**Cannot draw the blocks:** " + repr(e)
        self.suggest_btn.disabled = c.dim != "3D"
        self.add_btn.disabled = self.del_btn.disabled = c.dim != "3D"

    def _fill_tables(self):
        c = self.app.case
        types = layer_types(c)
        tip = blocks.tip_slice(c)
        s = c.blocks.slice_locations
        rows = []
        for k, y in enumerate(s):
            role = "root" if k == 0 else "tip" if k == tip else "far field" if k == len(s) - 1 else ""
            rows.append({"slice": "{}{}".format(k, " (" + role + ")" if role else ""), "y": y,
                         "layer above": types[k] if k < len(types) else "",
                         "cells": c.meshing.layer_resolution[k] if k < len(types) else None,
                         "grading": c.meshing.layer_grading[k] if k < len(types) else None})
        wing_layers = [k for k, t in enumerate(types) if t != "flow"]
        self._updating = True
        try:
            df = pd.DataFrame(rows)
            df["cells"] = df["cells"].astype("Int64")
            self.slices.value = df
            self.slices.selection = []
            self.div_table.value = pd.DataFrame(blocks.divisions(c))
            roles = {0: "TE", 13: "LE", 16: "", 2: "camber (division 1)", 5: "camber (division 2)",
                     8: "camber (division 3)", 12: "camber, nose", 14: "camber helper", 15: "camber helper"}
            self.vert_table.value = pd.DataFrame([{"vertex": i, "spline": {"s": "suction", "p": "pressure", "c": "camber"}[sp],
                                                   "role": roles.get(i, ""), "position": p}
                                                  for i, (p, sp) in enumerate(c.blocks.interior_points)])
            self.ext_table.value = pd.DataFrame([{"vertex": vid, "boundary": blocks.EXTERIOR_KIND[vid], "x": x, "z": z}
                                                 for vid, (x, z) in zip(range(22, 39), c.blocks.exterior_points)])
            self.aileron.options = ["layer {}".format(k) for k in wing_layers]
            self.aileron.value = ["layer {}".format(k) for k in (c.options.aileron_layers or []) if k in wing_layers]
            self.split.value, self.artificious.value = bool(c.options.split), bool(c.options.artificious)
            self.skip.value = int(c.options.skip_layer)
            self.skip.disabled = not c.options.artificious
            self.slice_sel.end = max(1, len(s) - 1)
            self.slice_sel.value = min(self.slice_sel.value, len(s) - 1)
        finally:
            self._updating = False

    def _done(self, msg=None, kind="info", tables=True):
        if msg:
            self.app.notify(kind, msg)
        self.app.changed()
        self.refresh(plot=True, tables=tables)

    # ---------------- slices ---------------- #
    def on_slice_edit(self, event):
        if self._updating:
            return
        c = self.app.case
        try:
            blocks.set_slice(c, int(event.row), float(event.value))
        except (ValueError, TypeError) as e:
            self._done(str(e), "warning")
            return
        self._done()

    def on_add(self, event):
        if self.add_y.value is None:
            return
        try:
            y = blocks.add_slice(self.app.case, float(self.add_y.value))
        except ValueError as e:
            self.app.notify("warning", str(e))
            return
        self._done("Slice added at y = {:.6g} m (nearest cell face: the cells did not change)".format(y), "success")

    def on_delete(self, event):
        sel = sorted(self.slices.selection, reverse=True)
        if not sel:
            self.app.notify("info", "Select slices with the checkboxes first")
            return
        try:
            for i in sel:
                blocks.remove_slice(self.app.case, int(i))
        except ValueError as e:
            self._done(str(e), "warning")
            return
        self._done("Removed {} slice(s); the layers were merged".format(len(sel)), "success")

    def on_suggest(self, event):
        try:
            s = blocks.suggest_slices(self.app.case)
        except ValueError as e:
            self.app.notify("warning", str(e))
            return
        self._done("Suggested slices: " + ", ".join("{:.4g}".format(y) for y in s) + " (cells: first guess, see step 4)",
                   "success")

    def on_slice_sel(self):
        if self._updating:
            return
        self.plot.slice = int(self.slice_sel.value)
        try:
            self.plot.update()
        except Exception as e:  # noqa: BLE001
            self.plot_msg.object = "**Cannot draw the blocks:** " + repr(e)

    # ---------------- vertices ---------------- #
    def on_div_edit(self, event):
        if self._updating:
            return
        col, row, val = event.column, int(event.row), float(event.value)
        try:
            if self.link.value:
                blocks.set_division(self.app.case, row, suction=val, pressure=val)
            elif col == "suction side":
                blocks.set_division(self.app.case, row, suction=val)
            else:
                blocks.set_division(self.app.case, row, pressure=val)
        except (ValueError, TypeError) as e:
            self._done("Not applied: " + str(e), "warning")
            return
        self._done()

    def on_vert_edit(self, event):
        if self._updating:
            return
        try:
            blocks.set_interior_point(self.app.case, int(event.row), float(event.value))
        except (ValueError, TypeError) as e:
            self._done("Not applied: " + str(e), "warning")
            return
        self._done()

    def on_ext_edit(self, event):
        if self._updating:
            return
        row = int(event.row)
        df = self.ext_table.value
        try:
            x, z = blocks.move_exterior(self.app.case, 22 + row, float(df.iloc[row]["x"]), float(df.iloc[row]["z"]))
        except (ValueError, TypeError) as e:
            self._done("Not applied: " + str(e), "warning")
            return
        self._done()

    # ---------------- options ---------------- #
    def on_aileron(self, event):
        if self._updating:
            return
        layers = sorted(int(v.split()[1]) for v in event.new)
        if layers and layers != list(range(layers[0], layers[-1] + 1)):
            self._done("Aileron layers must be consecutive", "warning")
            return
        self.app.case.options.aileron_layers = layers or None
        self._done()

    def on_patches(self, event):
        if self._updating:
            return
        o = self.app.case.options
        o.split, o.artificious = bool(self.split.value), bool(self.artificious.value)
        o.skip_layer = int(self.skip.value or 1)
        self.skip.disabled = not o.artificious
        self._done(tables=False)
