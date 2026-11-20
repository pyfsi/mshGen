'''
view3d.py: 3D views that also work in a browser without WebGL

Plotly 3D plots need WebGL in the browser. Without it (e.g. a browser on a remote desktop or in an X
session without graphics acceleration) Plotly shows only "WebGL is not supported by your browser".
A hidden probe (WebGLProbe) asks the browser whether WebGL is available. Each View3D then shows
  - interactive: the Plotly figure (drag to rotate, scroll to zoom), needs WebGL;
  - image:       a matplotlib 3D plot of the same surfaces, rendered on the server as a PNG, with view
                 presets and azimuth / elevation / zoom sliders to turn it.
The sidebar switch "3D views" (auto / interactive / image) overrides the automatic choice.

The surfaces are given as patches: (name, points (N x 3), faces (list of vertex index lists), colour).
'''
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import panel as pn
import param
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from panel.reactive import ReactiveHTML

MODES = ["auto", "interactive", "image"]
IMAGE_WIDTH = 1100
PROBE_TIMEOUT = 6.0          # s: no answer from the probe -> treat as "no WebGL" (images always work)
LIGHT = np.array([-0.4, -0.5, 1.0]) / np.linalg.norm([-0.4, -0.5, 1.0])

# view presets: azimuth (deg, camera direction from +x towards +y), elevation (deg), zoom
PRESETS = {"3D": (-125, 25, 1.0),
           "Top (x-y)": (0, 90, 1.0),
           "Front (y-z)": (0, 0, 1.0),
           "Side (x-z)": (-90, 0, 1.0)}

_RENDER = ThreadPoolExecutor(max_workers=1)           # renders, separate from the app's save jobs
_MPL_LOCK = threading.Lock()


class WebGLProbe(ReactiveHTML):
    '''Invisible element: reports whether the browser can create a WebGL context.'''
    webgl = param.Boolean(default=None, allow_None=True)

    _template = '<div id="probe" style="width:0;height:0;overflow:hidden"></div>'
    _scripts = {"render": """
        let ok = false;
        try {
          const c = document.createElement('canvas');
          ok = !!(window.WebGLRenderingContext && (c.getContext('webgl') || c.getContext('experimental-webgl')));
        } catch (e) { ok = false; }
        data.webgl = ok;
    """}


def grid_patch(name, X, Y, Z, color):
    '''Patch of a structured surface (arrays n x m): quads between neighbouring grid points.'''
    n, m = X.shape
    pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    idx = np.arange(n * m).reshape(n, m)
    faces = np.stack((idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]), axis=-1).reshape(-1, 4)
    return (name, pts, faces, color)


def _shade(color, normals):
    '''Face colours: base colour times ambient + diffuse (two-sided) light.'''
    from matplotlib.colors import to_rgb
    base = np.array(to_rgb(color))
    k = 0.5 + 0.5 * np.abs(normals @ LIGHT)
    return np.clip(base[None, :] * k[:, None], 0, 1)


def render_mpl(patches, az, el, zoom, width, height, title="", edges=False, max_edges=60000,
               labels=("x (m)", "y (m)", "z (m)")):
    '''PNG of the patches seen from azimuth/elevation (deg), orthographic projection, equal axes.'''
    polys, colors, legend = [], [], []
    for name, pts, faces, color in patches:
        pts = np.asarray(pts, dtype=float)
        if len(faces) == 0:
            continue
        quads = [pts[list(f)] for f in faces]
        normals = []
        for q in quads:
            nrm = np.cross(q[2] - q[0], q[-1] - q[1]) if len(q) > 3 else np.cross(q[1] - q[0], q[2] - q[0])
            ln = np.linalg.norm(nrm)
            normals.append(nrm / ln if ln > 0 else nrm)
        polys += quads
        colors.append(_shade(color, np.array(normals)))
        legend.append(Patch(facecolor=color, label=name))
    allpts = np.vstack(polys) if polys else np.zeros((1, 3))
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    span = np.maximum(hi - lo, 1e-3 * max(hi - lo))

    with _MPL_LOCK:                           # mplot3d is not thread-safe in all versions
        fig = Figure(figsize=(width / 100, height / 100), dpi=150)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        # one collection: matplotlib sorts faces by depth only within a collection
        draw_edges = edges and sum(len(p) for p in polys) <= max_edges
        fc = np.vstack(colors) if colors else np.zeros((0, 3))
        # without cell edges, faces get a thin edge in their own colour (no light seams between faces)
        coll = Poly3DCollection(polys, facecolors=fc, edgecolors=(0, 0, 0, 0.18) if draw_edges else fc,
                                linewidths=0.12 if draw_edges else 0.3)
        coll.set_clip_on(False)               # mplot3d clips to the unzoomed axes box (zoom > 1 cuts the wing)
        ax.add_collection3d(coll)
        ax.set_xlim(lo[0], lo[0] + span[0])
        ax.set_ylim(lo[1], lo[1] + span[1])
        ax.set_zlim(lo[2], lo[2] + span[2])
        ax.set_box_aspect(span, zoom=1.0)    # the zoom is applied after the fit (below)
        ax.set_proj_type("ortho")
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(5))
        axes = (ax.xaxis, ax.yaxis, ax.zaxis)
        axes[int(np.argsort(span)[0])].set_major_locator(MaxNLocator(2))      # the thin direction: few ticks
        a, e = np.radians(az), np.radians(el)
        view = np.abs([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        if view.max() > 0.97:                 # looking along an axis: its ticks only clutter the picture
            hidden = axes[int(np.argmax(view))]
            hidden.set_ticks([])
            hidden.label.set_visible(False)
        ax.tick_params(labelsize=8)
        # A thin wing fills only a strip of the 3D axes: zoom so that the drawing fills the frame (the zoom
        # slider multiplies this), then crop to the drawing with the title and the legend above it.
        W_in, H_in = fig.get_size_inches()
        header = 0.4 if title or len(legend) > 1 else 0.0
        box = _drawn_box(fig)
        if box is not None:
            x0, y0, x1, y1 = box
            fit = min(0.94 * min(W_in / (x1 - x0), (H_in - header) / (y1 - y0)), 8.0)
            if abs(fit * zoom - 1) > 0.05:    # zoom > 1: the picture is larger than the frame and cut at its edges
                ax.set_box_aspect(span, zoom=fit * zoom)
                ax.set_position([0, 0, 1, 1 - header / H_in])
                box = _drawn_box(fig)
        x0, y0, x1, y1 = box if box is not None else (0, 0, W_in, H_in - header)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W_in), min(y1, H_in - header)
        if title:
            fig.text(x0 / W_in, (y1 + 0.08) / H_in, title, fontsize=11, ha="left", va="bottom")
        if len(legend) > 1:
            fig.legend(handles=legend, loc="lower right", bbox_to_anchor=(x1 / W_in, (y1 + 0.02) / H_in),
                       ncol=len(legend), fontsize=9, frameon=False, handlelength=1.2, columnspacing=1.0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches=Bbox([[x0, y0], [x1, y1 + header]]))
    return buf.getvalue()


def _drawn_box(fig, pad_px=6):
    '''(x0, y0, x1, y1) in inches of the non-white part of the drawn figure, None if empty.'''
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    rows, cols = np.where((rgb < 250).any(axis=2))
    if not len(rows):
        return None
    H, dpi = rgb.shape[0], fig.dpi
    return ((cols.min() - pad_px) / dpi, (H - rows.max() - pad_px) / dpi,
            (cols.max() + pad_px) / dpi, (H - rows.min() + pad_px) / dpi)


class View3D:
    '''A 3D view: interactive Plotly figure (WebGL) or matplotlib image, see the module docstring.'''

    def __init__(self, app, fig, patches, height=500, slot="main", width=IMAGE_WIDTH, title="", edges=False):
        '''fig: Plotly figure (interactive); patches: the same surfaces for the matplotlib image;
        slot: one live view per slot (a new view of the same slot replaces the old one);
        width: pixel width of the image (about the width of the column it is shown in).'''
        self.app, self.fig, self.patches, self.height, self.width = app, fig, patches, height, width
        self.title, self.edges = title, edges
        self._shown, self._updating, self._job, self._cache = None, False, 0, {}
        self.preset = pn.widgets.RadioButtonGroup(options=list(PRESETS), value="3D", button_type="light",
                                                  width=420, sizing_mode="fixed")
        az, el, zoom = PRESETS["3D"]
        self.az = pn.widgets.IntSlider(name="Azimuth (deg)", start=-180, end=180, step=5, value=az, width=200)
        self.el = pn.widgets.IntSlider(name="Elevation (deg)", start=-90, end=90, step=5, value=el, width=200)
        self.zoom = pn.widgets.FloatSlider(name="Zoom", start=0.5, end=4.0, step=0.1, value=zoom, width=200)
        self.status = pn.pane.Markdown("", styles={"color": "#666"})
        self.image = pn.pane.PNG(None, sizing_mode="scale_width", max_width=width)
        self.controls = pn.Column(
            pn.Row(self.preset, self.status),
            pn.Row(self.az, self.el, self.zoom, sizing_mode="stretch_width"), sizing_mode="stretch_width")
        self.panel = pn.Column(sizing_mode="stretch_width", min_height=height)
        self.preset.param.watch(self.on_preset, "value")
        for w in (self.az, self.el, self.zoom):
            w.param.watch(self.on_camera, "value_throttled")
        app.view3d.views[slot] = self
        self.update()

    # ---------------- mode ---------------- #
    def update(self):
        mode = self.app.view3d.effective()
        if mode == self._shown:
            return
        self._shown = mode
        if mode is None:
            self.panel.objects = [pn.pane.Markdown("_Preparing the 3D view ..._")]
        elif mode == "interactive":
            self.panel.objects = [pn.pane.Plotly(self.fig, sizing_mode="stretch_width", height=self.height)]
        else:
            self.panel.objects = [self.controls, self.image]
            self.render()

    # ---------------- image mode ---------------- #
    def on_preset(self, event):
        if self._updating:
            return
        az, el, zoom = PRESETS[event.new]
        self._updating = True
        try:
            self.az.value, self.el.value, self.zoom.value = az, el, zoom
        finally:
            self._updating = False
        self.render()

    def on_camera(self, event):
        if not self._updating:
            self.render()

    def render(self):
        self._job += 1
        job = self._job
        key = (self.az.value, self.el.value, round(self.zoom.value, 2))
        if key in self._cache:
            self.image.object = self._cache[key]
            return
        args = (self.patches, *key, self.width, self.height, self.title, self.edges)
        self.status.object = "_drawing ..._"

        def done(png, error):
            if job != self._job:                     # a newer view was requested meanwhile
                return
            if error:
                self.status.object = "Cannot draw the 3D view: " + repr(error)
                return
            self._cache[key] = png
            self.image.object = png
            self.status.object = "_matplotlib image (no WebGL in this browser or 3D views set to image)_"
        self.app.run_background(lambda: render_mpl(*args), done, executor=_RENDER)


class View3DSettings:
    '''App part: probe, sidebar switch, registry of the live views.'''

    def __init__(self, app, initial="auto"):
        self.app = app
        self.probe = WebGLProbe(width=1, height=1, margin=0, sizing_mode="fixed")
        self.mode = pn.widgets.RadioButtonGroup(options=MODES, value=initial if initial in MODES else "auto",
                                                button_type="light")
        self.info = pn.pane.Markdown("", styles={"font-size": "12px", "color": "#555"})
        self.views = {}                       # slot -> live View3D
        self.started = None
        self.timed_out = False
        self.probe.param.watch(lambda e: self.changed(), "webgl")
        self.mode.param.watch(lambda e: self.changed(), "value")
        self.set_info()

    def widgets(self):
        return [pn.pane.Markdown("**3D views**"), self.mode, self.info, self.probe]

    def effective(self):
        '''"interactive", "image" or None (auto, browser not known yet).'''
        if self.mode.value != "auto":
            return self.mode.value
        if self.probe.webgl is None:
            return "image" if self.timed_out else None
        return "interactive" if self.probe.webgl else "image"

    def poll(self, now):
        if self.started is None:
            self.started = now
        if self.probe.webgl is None and not self.timed_out and now - self.started > PROBE_TIMEOUT:
            self.timed_out = True
            self.changed()

    def set_info(self):
        w = self.probe.webgl
        browser = ("browser has WebGL" if w else "browser has no WebGL" if w is False
                   else "no answer from the browser" if self.timed_out else "checking the browser ...")
        self.info.object = "{}: 3D views are **{}**".format(browser, {
            "interactive": "interactive (drag to rotate)", "image": "matplotlib images (turn with the sliders)",
            None: "..."}[self.effective()])

    def changed(self):
        self.set_info()
        for v in list(self.views.values()):
            v.update()
