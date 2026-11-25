'''
step3_blocks.py: step 3 of the GUI, the blocks of the domain, in two parts

3a. Spanwise slicing (SlicingStep)
    Inputs: slices (add at a cell face / move / delete / suggest), scale the slices with the span,
            per-layer options: aileron cut-out layers, first wing layers as patch 'extra' (ARTIFICIOUS)
    Plot:   top view of the half wing with the slices and the aileron layers
3b. Airfoil blocking (BlockingStep)
    Inputs: airfoil vertices (position along the suction / pressure / camber spline) and exterior vertices, as
            tables or dragged in the plot (each stays on its spline or boundary; 22 is free); wing surface as
            wingUp / wingDown (SPLIT); block vertices panel
    Plots:  small top view with the selected slice; block layout of that slice as mshGen generates it: blocks
            coloured by status (valid / skewed / folded), block numbers and hover info, airfoil, vertices
Cells and gradings (spanwise per layer, and along the block edges) are set in step 4.
Edits of vertices make them "custom" (not recomputed automatically; the vertex panel can reset them).
'''
import json

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, PointDrawTool, Range1d
from bokeh.plotting import figure

from core import blockopt, blocks, sizing
from core.blockcheck import BLOCKS_2D
from core.estimate import layer_types
from core.planform import place, section_params
from ui.common import note, scaled
from ui.step1_geometry import airfoil_data, top_view

GROUP_COLOR = {"wake": "#c6dbef", "upper": "#9ecae1", "lower": "#a1d99b", "interior": "#fdd0a2"}
STATUS_COLOR = {"skewed": "#fd8d3c", "folded": "#de2d26", "inverted": "#a50f15"}
SPLINE_COLOR = {"suction": "#08519c", "pressure": "#006d2c", "camber": "#d94801"}
PLOT_W, PLOT_H = 690, 580
SUGGEST = "Suggest block vertices"
FIXED_VERTICES = [0, 13]                    # TE, LE
NO_STATIONS = [[0, 0, 1.0, 0, 0, 0]]


def _case_key(case):
    return json.dumps(case.to_dict(), sort_keys=True)


def block_group(k):
    if k < 13:
        return "interior"
    if k in (13, 14, 25, 26):
        return "wake"
    return "upper" if k <= 19 else "lower"


def editors_readonly(columns):
    return {c: None for c in columns}


class SliceTopView:
    '''Small top view of the half wing (flow downwards) with all slices and the selected one highlighted;
    optionally the spanwise cell faces (thin lines, step 4).'''

    def __init__(self, height=200, tools=""):
        # explicit ranges, set from the wing and the slices in update(): automatic ranges kept the extent of the case
        # shown first when another case (another aircraft) was loaded
        self.x_range, self.y_range = Range1d(0, 1), Range1d(1, 0)      # y: start > end, the flow downwards
        f = figure(width=PLOT_W, height=height, tools=tools, toolbar_location=None if not tools else "right",
                   x_range=self.x_range, y_range=self.y_range)
        f.toolbar.logo = None
        f.xaxis.axis_label, f.yaxis.axis_label = "y (m)", "x (m)"
        self.src_faces = ColumnDataSource(dict(ys=[], xs=[]))
        f.multi_line("ys", "xs", source=self.src_faces, color="#4292c6", line_width=0.6, line_alpha=0.7)
        self.src_wing = ColumnDataSource(dict(y=[], x=[]))
        f.patch("y", "x", source=self.src_wing, fill_color="#d9d9d9", line_color="#636363", fill_alpha=0.8)
        self.src_slices = ColumnDataSource(dict(ys=[], xs=[]))
        f.multi_line("ys", "xs", source=self.src_slices, color="#969696", line_dash="dashed")
        self.src_sel = ColumnDataSource(dict(y=[], x=[]))
        f.line("y", "x", source=self.src_sel, color="#de2d26", line_width=3)
        self.src_lab = ColumnDataSource(dict(y=[], x=[], text=[]))
        f.text("y", "x", text="text", source=self.src_lab, text_font_size="10px", text_align="left", x_offset=3,
               text_baseline="top", text_color="#555555")
        self.figure = f
        self.pane = scaled(f)

    def update(self, case, sl, faces=None):
        '''faces: spanwise positions of the cell faces to draw (None: none).'''
        st = case.wing_stations or NO_STATIONS
        b2 = case.half_span or 1.0
        s = case.blocks.slice_locations
        ys = np.unique(np.r_[np.linspace(0, b2, 61), np.asarray(st)[:, 1]])
        le = np.array([place(np.array([[0.0, 0.0]]), *section_params(st, y), case.cog)[0][0] for y in ys])
        te = np.array([place(np.array([[1.0, 0.0]]), *section_params(st, y), case.cog)[0][0] for y in ys])
        pad = 0.15 * (te.max() - le.min() + 1e-9)
        lo, hi = float(le.min() - pad), float(te.max() + pad)
        y0, y1 = min(0.0, float(s[0])), float(max(s[-1], b2))
        dy = 0.02 * (y1 - y0 + 1e-9)
        self.x_range.update(start=y0 - dy, end=y1 + dy, reset_start=y0 - dy, reset_end=y1 + dy)
        self.y_range.update(start=hi, end=lo, reset_start=hi, reset_end=lo)
        self.src_wing.data = dict(y=list(np.r_[ys, ys[::-1]]), x=list(np.r_[le, te[::-1]]))
        self.src_slices.data = dict(ys=[[y, y] for y in s], xs=[[lo, hi]] * len(s))
        self.src_sel.data = dict(y=[s[sl], s[sl]], x=[lo, hi])
        self.src_lab.data = dict(y=list(s), x=[lo] * len(s), text=[str(k) for k in range(len(s))])
        f = [] if faces is None else list(faces)
        self.src_faces.data = dict(ys=[[y, y] for y in f], xs=[[lo, hi]] * len(f))
        self.figure.title.text = "Top view: slice {} at y = {:.4g} m (red){}".format(
            sl, s[sl], "; blue: spanwise cell faces ({})".format(len(f) - 1) if f else "")


class BlockPlot:
    '''Block layout of one slice. Exterior vertices (kept on their boundary) and airfoil vertices (kept on their
    spline) can be dragged with the point-draw tool.'''

    def __init__(self, step):
        self.step = step
        self._updating = False
        self._prev = {}
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
        self.src_prop = ColumnDataSource(dict(xs=[], ys=[]))          # suggested blocks (dashed)
        f.multi_line("xs", "ys", source=self.src_prop, color="black", line_dash="dashed", line_width=1.6)
        self.src_outer = ColumnDataSource(dict(x=[], z=[]))
        f.line("x", "z", source=self.src_outer, color="black", line_width=1.5)
        self.src_airfoil = ColumnDataSource(dict(x=[], z=[]))
        self.r_airfoil = f.patch("x", "z", source=self.src_airfoil, fill_color="#444444", line_color="#222222")
        self.texts = []
        # fixed (TE, LE) and derived (16-21) airfoil vertices
        self.src_fix = ColumnDataSource(dict(x=[], z=[], id=[], info=[]))
        r_fix = f.scatter("x", "z", source=self.src_fix, size=6, marker="square", color="black")
        self.texts.append(f.text("x", "z", text="id", source=self.src_fix, text_font_size="10px", x_offset=4,
                                 y_offset=-4, text_color="black"))
        self.src_der = ColumnDataSource(dict(x=[], z=[], id=[], info=[]))
        r_der = f.scatter("x", "z", source=self.src_der, size=6, fill_color="white", line_color="#737373")
        self.texts.append(f.text("x", "z", text="id", source=self.src_der, text_font_size="10px", x_offset=4,
                                 y_offset=-4, text_color="#737373"))
        # movable airfoil vertices, coloured by their spline
        self.src_air = ColumnDataSource(dict(x=[], z=[], id=[], color=[], info=[]))
        r_air = f.scatter("x", "z", source=self.src_air, size=10, color="color", alpha=0.9)
        self.texts.append(f.text("x", "z", text="id", source=self.src_air, text_font_size="11px", x_offset=6,
                                 y_offset=-6, text_color="color"))
        f.add_tools(HoverTool(renderers=[r_air, r_fix, r_der], tooltips=[("vertex", "@id"), ("", "@info")]))
        # exterior vertices
        self.src_ext = ColumnDataSource(dict(x=[], z=[], id=[]))
        r_ext = f.scatter("x", "z", source=self.src_ext, size=11, color="purple", alpha=0.8)
        f.text("x", "z", text="id", source=self.src_ext, text_font_size="11px", x_offset=6, y_offset=-6,
               text_color="purple")
        f.add_tools(HoverTool(renderers=[r_ext], tooltips=[("exterior vertex", "@id"), ("x", "@x{0.000}"),
                                                            ("z", "@z{0.000}")]))
        draw = PointDrawTool(renderers=[r_ext, r_air], add=False, description="Drag vertices")
        f.add_tools(draw)
        f.toolbar.active_drag = draw
        self.src_ext.on_change("data", lambda attr, old, new: self.on_drag("ext", new))
        self.src_air.on_change("data", lambda attr, old, new: self.on_drag("air", new))
        self.figure = f
        self.pane = scaled(f)

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
        sec = section_params(c.wing_stations or NO_STATIONS, y)
        dz = sec[3]
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
        ax, az = place(a.outline(), *sec, c.cog)
        flow = types[layer] == "flow"
        pts = c.blocks.interior_points
        cut = sl in blocks.aileron_slices(c)
        air = blocks.EDITABLE_VERTICES
        der = list(blocks.DERIVED) if flow else []
        self._updating = True
        try:
            self.src_blocks.data = dict(xs=xs, ys=ys, color=col, block=blk, status=stat, angles=ang, group=grp)
            self.src_outer.data = dict(x=list(np.r_[2 * R, R * np.cos(th), 2 * R, 2 * R]),
                                       z=list(dz + np.r_[R, R * np.sin(th), -R, R]))
            self.src_airfoil.data = dict(x=list(ax), z=list(az))
            self.r_airfoil.glyph.fill_alpha = 0.0 if flow else 1.0
            self.src_fix.data = dict(x=list(xz[FIXED_VERTICES, 0]), z=list(xz[FIXED_VERTICES, 1]),
                                     id=[str(i) for i in FIXED_VERTICES], info=["trailing edge (fixed)",
                                                                                "leading edge (fixed)"])
            self.src_der.data = dict(x=list(xz[der, 0]), z=list(xz[der, 1]), id=[str(i) for i in der],
                                     info=["halfway between {} and {} (mshGen)".format(*blocks.DERIVED[i])
                                           for i in der])
            self.src_air.data = dict(
                x=list(xz[air, 0]), z=list(xz[air, 1]), id=[str(i) for i in air],
                color=[SPLINE_COLOR[blocks.SPLINE_NAME[pts[i][1]]] for i in air],
                info=["{} side, position {:.4g}".format(blocks.SPLINE_NAME[pts[i][1]], pts[i][0])
                      + (" (set by the aileron cut in this slice)" if cut and i in (4, 5, 6) else "")
                      for i in air])
            self.src_ext.data = dict(x=list(xz[22:, 0]), z=list(xz[22:, 1]), id=[str(i) for i in range(22, 39)])
            prop = getattr(self.step, "proposal", None)
            if prop:
                ps = blockopt._Slice(c, a, sl, flow)
                pz = blockopt.slice_vertices(a, prop["interior_points"], prop["exterior_points"], ps)
                quads = [BLOCKS_2D[k] + BLOCKS_2D[k][:1] for k in lay["active"][layer]]
                self.src_prop.data = dict(xs=[list(pz[q, 0]) for q in quads], ys=[list(pz[q, 1]) for q in quads])
            else:
                self.src_prop.data = dict(xs=[], ys=[])
            self._prev = {"air": (list(air), np.array(xz[air, 0]), np.array(xz[air, 1])),
                          "ext": (list(range(22, 39)), np.array(xz[22:, 0]), np.array(xz[22:, 1]))}
            self._slice_y, self._slice, self._dz = y, sl, dz
            for t in self.texts:
                t.visible = self.zoom != "Whole domain"
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

    # ---------------- drag ---------------- #
    def on_drag(self, kind, new):
        if self._updating or kind not in self._prev:
            return
        ids, px, pz = self._prev[kind]
        x, z = np.asarray(new.get("x", []), float), np.asarray(new.get("z", []), float)
        if len(x) != len(px):                       # a point was deleted (backspace): restore
            self.update()
            return
        moved = np.where(np.hypot(x - px, z - pz) > 1e-9 * (1 + np.abs(px).max()))[0]
        if len(moved) == 0:
            return
        app = self.step.app
        c = app.case
        try:
            for i in moved:
                vid = ids[int(i)]
                if kind == "ext":
                    blocks.move_exterior(c, vid, float(x[i]), float(z[i]) - self._dz)
                elif vid in (4, 5, 6) and self._slice in blocks.aileron_slices(c):
                    raise ValueError("in slice {} mshGen places vertices 4, 5, 6 at the aileron cut; move them in "
                                     "a slice without the cut".format(self._slice))
                else:
                    blocks.move_interior(c, vid, airfoil_data(c.airfoil.name, app.airfoil_file),
                                         float(x[i]), float(z[i]), self._slice_y)
        except ValueError as e:
            app.notify("warning", "Not moved: " + str(e))
            self.update()
            return
        app.changed()
        self.step.refresh(plot=True, tables=True)


class SlicingStep:
    title = "3a. Spanwise slicing"
    intro = 'Where the block vertices are placed along the span; the cells follow in step 4.'

    def __init__(self, app):
        self.app = app
        self._updating = False
        self.slices = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=260, selectable="checkbox",
                                           layout="fit_data_stretch", sizing_mode="stretch_width",
                                           editors=editors_readonly(["slice", "layer above"]))
        self.add_y = pn.widgets.FloatInput(name="New slice near y (m)", step=0.1, width=180)
        self.add_btn = pn.widgets.Button(name="Add slice", button_type="light", align="end")
        self.del_btn = pn.widgets.Button(name="Delete selected", button_type="light", align="end")
        self.suggest_btn = pn.widgets.Button(name="Suggest slices", button_type="light", align="end")
        self.scale_slices = pn.widgets.Checkbox(name="Scale the slices when the span changes", value=True)
        self.slice_msg = pn.pane.Markdown(
            "_Slices are the planes along the span where the block vertices are placed: root, planform kinks, "
            "around the tip, tip, far field. Edit y in the table to move a slice (root and tip are fixed). A new "
            "slice is placed at a cell face of the current spanwise mesh. The spanwise cells and grading of each "
            "layer are set in step 4._")
        self.aileron = pn.widgets.MultiChoice(name="Aileron cut-out in layers (consecutive)", options=[])
        self.artificious = pn.widgets.Checkbox(name="First wing layers as separate patch 'extra' (ARTIFICIOUS)")
        self.skip = pn.widgets.IntInput(name="Layers in 'extra'", start=1, step=1, width=150)
        self.dim_msg = pn.pane.Markdown()
        self.plots = pn.Column(sizing_mode="stretch_width")

        self.slices.on_edit(self.on_slice_edit)
        self.add_btn.on_click(self.on_add)
        self.del_btn.on_click(self.on_delete)
        self.suggest_btn.on_click(self.on_suggest)
        self.aileron.param.watch(self.on_aileron, "value")
        for w in (self.artificious, self.skip):
            w.param.watch(self.on_patches, "value")

        self.inputs = pn.Column(
            self.dim_msg,
            pn.pane.Markdown("### Slices"), self.slices,
            pn.Row(self.add_y, self.add_btn, self.del_btn), self.suggest_btn, self.scale_slices, self.slice_msg,
            pn.pane.Markdown("### Layers: aileron cut-out and patches"), self.aileron,
            pn.pane.Markdown("_Aileron cut position along the chord: fixed by mshGen per wing name (MegAWES: 70 % "
                             "chord on the suction side, cut normal tilted 20 deg); see LIMITATIONS.md item 4._"),
            pn.Row(self.artificious, self.skip),
            width=470, sizing_mode="fixed", margin=(5, 40, 5, 10))

    def load(self):
        pass

    def view(self):
        self.refresh()
        return [pn.Row(self.inputs, self.plots, sizing_mode="stretch_width")]

    def refresh(self, plot=True, tables=True, set_range=False):
        c = self.app.case
        if tables:
            self._fill_tables()
        three_d = c.dim == "3D"
        for w in (self.add_btn, self.del_btn, self.suggest_btn, self.add_y):
            w.disabled = not three_d
        self.dim_msg.object = "" if three_d else "_2D case: one layer, no spanwise slicing._"
        if plot:
            try:
                self.plots.objects = [top_view(c, slices=True)] if three_d and c.wing_stations else []
            except Exception as e:  # noqa: BLE001
                self.plots.objects = [note("Cannot draw the top view: " + repr(e), "danger")]

    def _fill_tables(self):
        c = self.app.case
        types = layer_types(c)
        tip = blocks.tip_slice(c)
        s = c.blocks.slice_locations
        rows = []
        for k, y in enumerate(s):
            role = "root" if k == 0 else "tip" if k == tip else "far field" if k == len(s) - 1 else ""
            rows.append({"slice": "{}{}".format(k, " (" + role + ")" if role else ""), "y": y,
                         "layer above": types[k] if k < len(types) else ""})
        wing_layers = [k for k, t in enumerate(types) if t != "flow"]
        self._updating = True
        try:
            self.slices.value = pd.DataFrame(rows)
            self.slices.selection = []
            self.aileron.options = ["layer {}".format(k) for k in wing_layers]
            self.aileron.value = ["layer {}".format(k) for k in (c.options.aileron_layers or []) if k in wing_layers]
            self.artificious.value = bool(c.options.artificious)
            self.skip.value = int(c.options.skip_layer)
            self.skip.disabled = not c.options.artificious
        finally:
            self._updating = False

    def _done(self, msg=None, kind="info", tables=True):
        if msg:
            self.app.notify(kind, msg)
        self.app.changed()
        self.refresh(plot=True, tables=tables)

    def on_slice_edit(self, event):
        if self._updating:
            return
        try:
            blocks.set_slice(self.app.case, int(event.row), float(event.value))
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
        self._done("Slice added at y = {:.6g} m (at a cell face of the spanwise mesh)".format(y), "success")

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
        self._done("Suggested slices: " + ", ".join("{:.4g}".format(y) for y in s)
                   + " (spanwise cells: a first guess, see step 4)", "success")

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
        o.artificious = bool(self.artificious.value)
        o.skip_layer = int(self.skip.value or 1)
        self.skip.disabled = not o.artificious
        self._done(tables=False)


class BlockingStep:
    title = "3b. Airfoil blocking"
    intro = 'The blocks in each slice: the airfoil and exterior block vertices, checked live.'

    def __init__(self, app):
        self.app = app
        self._updating = False

        # Vertices
        self.vert_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=390,
                                               layout="fit_data_stretch", sizing_mode="stretch_width",
                                               editors=editors_readonly(["vertex", "boundary"]))
        self.cut_msg = pn.pane.Markdown()
        self.ext_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=300,
                                              layout="fit_data_stretch", sizing_mode="stretch_width",
                                              editors=editors_readonly(["vertex", "boundary"]))
        self.split = pn.widgets.Checkbox(name="Wing surface as wingUp / wingDown (SPLIT)")
        self.vertex_box = pn.Column(sizing_mode="stretch_width")

        # Suggest block vertices (core/blockopt.py)
        self.proposal, self._proposal_key = None, None
        self.opt_move = pn.widgets.Select(name="Move", options=list(blockopt.MOVE), value=list(blockopt.MOVE)[0])
        self.opt_focus = pn.widgets.Select(name="Priority close to the airfoil", options=list(blockopt.FOCUS),
                                           value=list(blockopt.FOCUS)[0])
        self.opt_btn = pn.widgets.Button(name=SUGGEST, button_type="primary", sizing_mode="stretch_width")
        self.opt_note = pn.pane.Markdown(sizing_mode="stretch_width")
        self.opt_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, layout="fit_columns",
                                              sizing_mode="stretch_width", disabled=True, visible=False,
                                              widths={"quality": 290, "now": 65, "suggested": 100},
                                              formatters={"quality": {"type": "textarea"}})
        self.opt_apply = pn.widgets.Button(name="Apply", button_type="success", visible=False)
        self.opt_discard = pn.widgets.Button(name="Discard", button_type="light", visible=False)
        self.opt_btn.on_click(self.on_suggest)
        self.opt_apply.on_click(self.on_apply)
        self.opt_discard.on_click(lambda e: self.clear_proposal(refresh=True))

        # Plots
        self.top = SliceTopView()
        self.plot = BlockPlot(self)
        self.slice_sel = pn.widgets.IntSlider(name="Slice", start=0, end=1, value=0, width=300)
        self.zoom = pn.widgets.RadioButtonGroup(options=["Whole domain", "Airfoil", "Trailing edge"],
                                                value="Whole domain", button_type="primary", width=400)
        self.plot_msg = pn.pane.Markdown()
        swatch = '<span style="background:{};padding:2px 8px;margin-right:6px;border:1px solid #999">{}</span>'
        dot = '<span style="color:{};font-size:16px">&#9679;</span> {} &nbsp; '
        legend = pn.pane.HTML(
            "".join(swatch.format(c, t) for t, c in [
                ("wake", GROUP_COLOR["wake"]), ("upper side", GROUP_COLOR["upper"]),
                ("lower side", GROUP_COLOR["lower"]), ("inside contour (flow layers)", GROUP_COLOR["interior"]),
                ("skewed", STATUS_COLOR["skewed"]), ("folded", STATUS_COLOR["folded"])])
            + "<br>" + "".join(dot.format(c, t) for t, c in [
                ("suction side", SPLINE_COLOR["suction"]), ("pressure side", SPLINE_COLOR["pressure"]),
                ("camber line", SPLINE_COLOR["camber"]), ("exterior", "purple")])
            + "&#9632; TE / LE (fixed) &nbsp; &#9675; 16-21 (halfway, mshGen)"
            + "<br><small>Drag the coloured vertices with the point-draw tool (toolbar; zoom to <i>Airfoil</i> for "
              "the airfoil vertices). Airfoil vertices stay on their spline, exterior vertices on their boundary; "
              "22 (near-wake end) is free. Hover a block for its number and corner angles.</small>")

        self.vert_table.on_edit(self.on_vert_edit)
        self.ext_table.on_edit(self.on_ext_edit)
        self.split.param.watch(self.on_split, "value")
        self.slice_sel.param.watch(lambda e: self.on_slice_sel(), "value")
        self.zoom.param.watch(lambda e: self.plot.set_zoom(e.new), "value")

        self.inputs = pn.Column(
            pn.pane.Markdown("### Suggest block vertices"),
            pn.pane.Markdown("_Moves the vertices for orthogonal cells, with priority close to the airfoil: grid "
                             "lines normal to the wall, block corners near 90 deg, smooth expansion, wake line along "
                             "the flow. Evaluates the blocks, not the cells (about 40 s). Shown dashed in the plot; "
                             "see docs/block_optimiser.md._"),
            pn.Row(self.opt_move, self.opt_focus), self.opt_btn, self.opt_note, self.opt_table,
            pn.Row(self.opt_apply, self.opt_discard),
            pn.pane.Markdown("### Airfoil vertices"),
            pn.pane.Markdown("_Position along the spline of the boundary: 0 = LE, 1 = TE (about x/c), the same in "
                             "every slice. TE (0) and LE (13) are fixed; 14 only places 20 and 21; 16-21 lie "
                             "halfway between their neighbours (mshGen); 15 is not used by mshGen._"),
            self.vert_table, self.cut_msg,
            width=470, sizing_mode="fixed", margin=(5, 40, 5, 10))
        self.lower = pn.Row(                            # full width, below the plots
            pn.Column(pn.pane.Markdown("### Exterior vertices"),
                      pn.pane.Markdown("_x, z in the root plane (m, origin = COG; other slices add dz). They stay on "
                                       "their boundary; 22 (near-wake end) is free._"),
                      self.ext_table, width=470, sizing_mode="fixed", margin=(5, 40, 5, 10)),
            pn.Column(pn.pane.Markdown("### Block vertices and block check"), self.vertex_box,
                      pn.pane.Markdown("### Wing surface patches"), self.split, sizing_mode="stretch_width"),
            sizing_mode="stretch_width")
        self.right = pn.Column(self.slice_sel, self.top.pane, self.zoom, self.plot_msg, self.plot.pane, legend,
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
        self.clear_proposal()

    def view(self):
        self.refresh(plot=True, tables=True, set_range=getattr(self, "_first_view", True))
        self._first_view = False
        return [pn.Row(self.inputs, self.right, sizing_mode="stretch_width"),
                pn.layout.Divider(margin=(18, 0, 6, 0)), pn.pane.Markdown("## Exterior vertices, block check and patches"),
                self.lower]

    def refresh(self, plot=True, tables=True, set_range=False):
        if self.proposal is not None and self._proposal_key != _case_key(self.app.case):
            self.clear_proposal()                   # the case changed since the suggestion: outdated
            self.opt_note.object = "_The case changed: the suggestion was discarded._"
        if tables:
            self._fill_tables()
        self.vertex_box.objects = [self.app.vertex_panel.view()]
        if plot:
            self._update_plots(set_range)

    def _update_plots(self, set_range=False):
        c = self.app.case
        self.top.pane.visible = c.dim == "3D" and c.half_span is not None
        try:
            if self.top.pane.visible:
                self.top.update(c, min(self.plot.slice, len(c.blocks.slice_locations) - 1))
            self.plot.update(set_range=set_range)
            self.plot_msg.object = ""
        except Exception as e:  # noqa: BLE001
            self.plot_msg.object = "**Cannot draw the blocks:** " + repr(e)

    def _fill_tables(self):
        c = self.app.case
        s = c.blocks.slice_locations
        cut = blocks.aileron_slices(c)
        self._updating = True
        try:
            self.vert_table.value = pd.DataFrame(blocks.interior_rows(c))
            self.cut_msg.object = ("_Aileron cut: in slices {} mshGen places 4, 5 and 6 itself (on the cut); the "
                                   "positions here apply to the other slices._".format(", ".join(map(str, cut)))
                                   if cut else "")
            self.ext_table.value = pd.DataFrame([{"vertex": vid, "boundary": blocks.EXTERIOR_KIND[vid], "x": x, "z": z}
                                                 for vid, (x, z) in zip(range(22, 39), c.blocks.exterior_points)])
            self.split.value = bool(c.options.split)
            self.slice_sel.end = max(1, len(s) - 1)
            self.slice_sel.value = min(self.slice_sel.value, len(s) - 1)
        finally:
            self._updating = False

    def _done(self, msg=None, kind="info", tables=True):
        if msg:
            self.app.notify(kind, msg)
        self.app.changed()
        self.refresh(plot=True, tables=tables)

    def on_slice_sel(self):
        if self._updating:
            return
        self.plot.slice = int(self.slice_sel.value)
        self._update_plots()

    # ---------------- vertices ---------------- #
    def on_vert_edit(self, event):
        if self._updating:
            return
        vid = blocks.EDITABLE_VERTICES[int(event.row)]
        try:
            blocks.set_interior_point(self.app.case, vid, float(event.value))
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
            blocks.move_exterior(self.app.case, 22 + row, float(df.iloc[row]["x"]), float(df.iloc[row]["z"]))
        except (ValueError, TypeError) as e:
            self._done("Not applied: " + str(e), "warning")
            return
        self._done()

    def on_split(self, event):
        if self._updating:
            return
        self.app.case.options.split = bool(self.split.value)
        self._done(tables=False)

    # ---------------- suggest block vertices ---------------- #
    def clear_proposal(self, refresh=False):
        self.proposal, self._proposal_key = None, None
        self.opt_table.visible = self.opt_apply.visible = self.opt_discard.visible = False
        self.opt_note.object = ""
        if refresh:
            self._update_plots()

    def on_suggest(self, event):
        c, f = self.app.case, self.app.airfoil_file
        move, focus = self.opt_move.value, self.opt_focus.value
        key = _case_key(c)
        self.clear_proposal()
        self.opt_btn.disabled, self.opt_btn.name = True, "Searching (about 40 s) ..."
        self.opt_note.object = ""

        def job():
            a = airfoil_data(c.airfoil.name, f)
            return blockopt.suggest(c, a, f, move=move, focus=focus)

        def done(result, error):
            self.opt_btn.disabled, self.opt_btn.name = False, SUGGEST
            if error:
                self.opt_note.object = "**No suggestion:** " + str(error)
                return
            if key != _case_key(self.app.case):
                self.opt_note.object = "_The case changed during the search: run it again._"
                return
            self.proposal, self._proposal_key = result, key
            rows = [{"quality": k, "now": "{:.3g}".format(result["before"][k]),
                     "suggested": "{:.3g}".format(result["after"][k])} for k in result["before"]]
            self.opt_table.value = pd.DataFrame(rows)
            self.opt_table.visible = self.opt_apply.visible = self.opt_discard.visible = True
            self.opt_note.object = ("_Suggested blocks dashed in the plot. Smaller is better everywhere. The block "
                                    "values are over slices {}; the cell values use the current cells of step 4 "
                                    "(root and tip slice). {}._".format(
                                        result["note"].split("slices evaluated: ")[-1], result["note"].split(";")[0]))
            self._update_plots()
        self.app.run_background(job, done)

    def on_apply(self, event):
        if self.proposal is None:
            return
        sizing.apply_proposal(self.app.case, {"interior_points": self.proposal["interior_points"],
                                              "exterior_points": self.proposal["exterior_points"]})
        self.clear_proposal()
        self._done("Suggested block vertices applied. The segment lengths changed: check the chordwise cells in "
                   "step 4.", "success")
