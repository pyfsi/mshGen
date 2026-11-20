'''
view3d.py: 3D views (Plotly) that also work in a browser without WebGL

Plotly 3D plots need WebGL in the browser. Without it (e.g. a browser on a remote desktop or in an X
session without graphics acceleration) Plotly shows only "WebGL is not supported by your browser".
A hidden probe (WebGLProbe) asks the browser whether WebGL is available. Each View3D then shows
  - interactive: the Plotly figure (drag to rotate, scroll to zoom), needs WebGL;
  - image:       the same figure rendered on the server (kaleido) as a PNG, with view presets and
                 azimuth / elevation / distance sliders to turn it.
The sidebar switch "3D views" (auto / interactive / image) overrides the automatic choice.
'''
import hashlib
import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import panel as pn
import param
from panel.reactive import ReactiveHTML

MODES = ["auto", "interactive", "image"]
IMAGE_WIDTH = 1100
PROBE_TIMEOUT = 6.0          # s: no answer from the probe -> treat as "no WebGL" (images always work)

# view presets: azimuth (deg, from +x towards +y), elevation (deg), distance (Plotly eye units)
PRESETS = {"3D": (-145, 30, 3.6),
           "Top (x-y)": (0, 90, 2.4),
           "Front (y-z)": (0, 0, 2.4),
           "Side (x-z)": (-90, 0, 2.4)}

_RENDER = ThreadPoolExecutor(max_workers=1)           # kaleido renders, separate from the app's save jobs
_PNG_CACHE, _PNG_LOCK = OrderedDict(), threading.Lock()


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


def camera(az, el, dist):
    '''Plotly scene camera for azimuth/elevation (deg) and distance.'''
    a, e = math.radians(az), math.radians(el)
    eye = dict(x=dist * math.cos(e) * math.cos(a), y=dist * math.cos(e) * math.sin(a), z=dist * math.sin(e))
    if abs(el) > 85:          # looking straight down/up: x (flow) points down the screen, y to the right
        up = dict(x=-math.cos(a) * math.copysign(1, el), y=-math.sin(a) * math.copysign(1, el), z=0)
    else:
        up = dict(x=0, y=0, z=1)
    return dict(eye=eye, up=up, center=dict(x=0, y=0, z=0))


def render_png(fig, az, el, dist, width, height):
    '''PNG of the Plotly figure seen from (az, el, dist); cached (kaleido needs a few seconds).'''
    spec = fig.to_json()
    key = (hashlib.md5(spec.encode()).hexdigest(), az, el, round(dist, 3), width, height)
    with _PNG_LOCK:
        if key in _PNG_CACHE:
            _PNG_CACHE.move_to_end(key)
            return _PNG_CACHE[key]
    f = fig.__class__(fig)                     # copy: the interactive figure keeps its own camera
    f.update_layout(scene_camera=camera(az, el, dist), width=width, height=height)
    title = f.layout.title.text
    if title:
        f.update_layout(title_text=title.replace("(drag to rotate)", "").strip())
    png = f.to_image(format="png", width=width, height=height)
    with _PNG_LOCK:
        _PNG_CACHE[key] = png
        while len(_PNG_CACHE) > 40:
            _PNG_CACHE.popitem(last=False)
    return png


class View3D:
    '''A 3D view of a Plotly figure: interactive (WebGL) or server-rendered image, see the module docstring.'''

    def __init__(self, app, fig, height=500, slot="main", width=IMAGE_WIDTH):
        '''slot: one live view per slot (a new view of the same slot replaces the old one);
        width: pixel width of the server-rendered image (about the width of the column it is shown in).'''
        self.app, self.fig, self.height, self.width = app, fig, height, width
        self._shown, self._updating, self._job = None, False, 0
        self.preset = pn.widgets.RadioButtonGroup(options=list(PRESETS), value="3D", button_type="light",
                                                  width=420, sizing_mode="fixed")
        az, el, dist = PRESETS["3D"]
        self.az = pn.widgets.IntSlider(name="Azimuth (deg)", start=-180, end=180, step=5, value=az, width=200)
        self.el = pn.widgets.IntSlider(name="Elevation (deg)", start=-90, end=90, step=5, value=el, width=200)
        self.dist = pn.widgets.FloatSlider(name="Distance", start=0.6, end=8.0, step=0.1, value=dist, width=200)
        self.status = pn.pane.Markdown("", styles={"color": "#666"})
        self.image = pn.pane.PNG(None, sizing_mode="scale_width", max_width=width)
        self.controls = pn.Column(
            pn.Row(self.preset, self.status),
            pn.Row(self.az, self.el, self.dist, sizing_mode="stretch_width"), sizing_mode="stretch_width")
        self.panel = pn.Column(sizing_mode="stretch_width", min_height=height)
        self.preset.param.watch(self.on_preset, "value")
        for w in (self.az, self.el, self.dist):
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
        az, el, dist = PRESETS[event.new]
        self._updating = True
        try:
            self.az.value, self.el.value, self.dist.value = az, el, dist
        finally:
            self._updating = False
        self.render()

    def on_camera(self, event):
        if not self._updating:
            self.render()

    def render(self):
        self._job += 1
        job = self._job
        az, el, dist = self.az.value, self.el.value, self.dist.value
        fig, height, width = self.fig, self.height, self.width
        self.status.object = "_rendering on the server ..._"

        def done(png, error):
            if job != self._job:                     # a newer view was requested meanwhile
                return
            if error:
                self.status.object = "Cannot render the 3D view: " + repr(error)
                return
            self.image.object = png
            self.status.object = "_static image (no WebGL in this browser or 3D views set to image)_"
        self.app.run_background(lambda: render_png(fig, az, el, dist, width, height), done, executor=_RENDER)


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
            "interactive": "interactive (drag to rotate)", "image": "images rendered on the server",
            None: "..."}[self.effective()])

    def changed(self):
        self.set_info()
        for v in list(self.views.values()):
            v.update()
