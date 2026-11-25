'''
record.py: records the tutorial video of the mshGen GUI (GUI/docs/mshGen_GUI_tutorial.mp4)

A headless Firefox drives the real GUI through a complete session: create a wing case and go through all six
steps, using or pointing at every input and option. Each frame shows the GUI with the explained element
outlined and a pop-up message next to it; frames are held long enough to read the message. Long computations
(Make mesh, the mesh optimiser, the build, the test run) are shown sped up: a frame every few seconds, shown for
half a second, with the elapsed time. The video is kept below 10 minutes (MAX_VIDEO). No audio.
The 3D views (Plotly, WebGL) cannot be drawn by headless Firefox: the same figures are rendered with Plotly's
static export (kaleido) and placed over them.

Usage (Anaconda module loaded; needs OpenFOAM and ParaView for the build in step 5 and the test run in step 6,
and the Claude Code CLI for the AI summary and review; about 30 minutes):
    python GUI/tools/tutorial/record.py [--out GUI/docs/mshGen_GUI_tutorial.mp4] [--frames DIR] [--until CHAPTER]
--until stops before the chapter that starts with CHAPTER (a quick check of the first steps).
The tutorial case cases/tutorial_wing is created and deleted again.
Before the next recording: the changes listed in GUI/PLAN.md, 'Next tutorial recording'.
'''
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GUI = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, GUI)
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from browser import Browser  # noqa: E402

CASE = "tutorial_wing"
W, H = 1600, 1000          # browser window
FRAME = (1600, 914)         # frame size = the page area of the window (every frame is padded to it)
HEADER = 58                 # height of the GUI's header bar (the pasted 3D figures stay below it)
ORANGE = (230, 85, 13)
HOLD = (2.0, 0.30, 3.5, 11.0)   # hold time of a frame: base + per word of the pop-up, at least, at most (s)
FAST = 0.5                  # s per frame of a sped-up computation
MAX_VIDEO = 585.0           # s: longer videos are shortened by scaling the hold times (at least 3 s per frame)
MAKE = "Make mesh (slices, block vertices, cells)"
UNDO, REDO = "↶ Undo", "Redo ↷"


def q(s):
    return json.dumps(s)


# JavaScript expressions that return the element to explain
def W_(title):
    return "T.el(T.find({}))".format(q(title))


def L_(label):
    return "T.el(T.find({}, 'labels'))".format(q(label))


def HAS(text, n=0):
    return "T.el(T.find({}, 'has', {}))".format(q(text), n)


def TABS(title):
    return "T.el(T.find({}, 'tabs'))".format(q(title))


def BTN(label):
    return "T.button({})".format(q(label))


def TYPED(t, n=0):
    return "T.el(T.typed({}, {}))".format(q(t), n)


def CARD(title):
    return "T.el(T.cardview({}))".format(q(title))


def font(size, bold=False):
    import matplotlib
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(path, size)


class StopRecording(Exception):
    '''--until: the chapter to stop before was reached.'''


class Recorder:

    def __init__(self, frames_dir, until=None):
        self.dir = frames_dir
        os.makedirs(frames_dir, exist_ok=True)
        self.frames = []            # (file, seconds)
        self.chapters = []          # (title, frame index)
        self.missing = []           # frames whose element was not found (no outline)
        self.until = until
        self.b = None
        self.fig3d = None           # Plotly figure drawn over the (WebGL) 3D view when it is on screen

    # ---------------- frames ---------------- #
    def _save(self, img, seconds):
        if img.size != FRAME:                         # all frames the same size (the video needs it)
            canvas = Image.new("RGB", FRAME, "white")
            img = img.copy()
            img.thumbnail(FRAME)
            canvas.paste(img, ((FRAME[0] - img.width) // 2, (FRAME[1] - img.height) // 2))
            img = canvas
        path = os.path.join(self.dir, "frame{:03d}.png".format(len(self.frames)))
        img.save(path)
        self.frames.append((path, seconds))
        print("frame {:3d} ({:4.1f} s)".format(len(self.frames) - 1, seconds), flush=True)

    @staticmethod
    def hold(text):
        words = len(text.replace("<b>", "").replace("</b>", "").split())
        base, per, lo, hi = HOLD
        return max(lo, min(hi, base + per * words))

    def chapter(self, title):
        if self.until and title.startswith(self.until):
            raise StopRecording(title)
        self.chapters.append((title, len(self.frames)))
        print("== " + title, flush=True)

    def card(self, title, lines, seconds=None, sub=None):
        img = Image.new("RGB", FRAME, (44, 110, 155))
        d = ImageDraw.Draw(img)
        d.text((110, 100), title, font=font(60, True), fill="white")
        if sub:
            d.text((110, 190), sub, font=font(30), fill=(220, 235, 245))
        y = 270
        for line in lines:
            mono = line.startswith("$ ") or line.startswith("  ")
            d.text((130 if mono else 110, y), line, font=font(28 if not mono else 26), fill="white")
            y += 48
        self._save(img, seconds or self.hold(" ".join([title] + lines)))

    def shot(self, title, body, target=None, where=None, click=None, seconds=None, also=(), composite=None):
        '''Outline target (and the elements in also), show the pop-up, screenshot.'''
        if target and not self.b.T("!!({})".format(target)):
            self.missing.append(title)
            print("WARNING: element not found for '{}'".format(title), flush=True)
        js = ["T.clear();", "const e = {};".format(target or "null"), "if (e) {T.scroll(e); }"]
        self.b.T("(function(){" + " ".join(js) + " return true})()")
        time.sleep(0.8)
        marks = [target] + list(also)
        call = ("(function(){{const e = {t}; {m} T.pop({ti}, {bo}, e, {wh}); {c} return true}})()".format(
            t=target or "null", m=" ".join("T.mark({});".format(x) for x in marks if x), ti=q(title), bo=q(body),
            wh=q(where) if where else "null", c="T.dot({});".format(click) if click else ""))
        self.b.T(call)
        time.sleep(0.5)
        img = Image.open(io.BytesIO(self.b.screenshot())).convert("RGB")
        if composite:
            composite(img)
        elif self.fig3d is not None:
            self.overlay_plotly(img, self.fig3d)
        self._save(img, seconds or self.hold(title + " " + body))
        self.b.T("T.clear()")

    def timelapse(self, title, body, target, done, every, where=None, before=None, max_frames=20, timeout=3600):
        '''A running computation sped up: a frame every `every` s, shown FAST s each, with the elapsed time in the
        title, until done (a JavaScript condition) is true; the last frame shows the finished state. Beyond
        max_frames only the last frame is taken.'''
        t0, n = time.time(), 0
        while True:
            finished = bool(self.b.T(done))
            if n < max_frames or finished:
                if before:
                    self.b.T(before)
                el = int(time.time() - t0)
                self.shot("{} · sped up · {:d}:{:02d}".format(title, el // 60, el % 60), body, target, where=where,
                          seconds=FAST)
                n += 1
            if finished:
                return
            if time.time() - t0 > timeout:
                raise TimeoutError(title)
            time.sleep(every)

    def overlay_plotly(self, img, fig):
        '''Draw a Plotly figure (static export) over the first visible Plotly view in the screenshot.'''
        r = self.b.T("(function(){const v = T.typed('Plotly'); return v ? T.rect(v.el) : null})()")
        if not r:
            return
        x, y, w, h = (int(round(v)) for v in r)
        y0, y1 = max(y, HEADER), min(y + h, img.height)
        if y1 - y0 < 20 or w < 50:
            return
        key = (id(fig), w, h)
        if getattr(self, "_png_key", None) != key:              # kaleido takes seconds: cache per figure/size
            self._png = Image.open(io.BytesIO(fig.to_image(format="png", width=w, height=h))).convert("RGB")
            self._png_key = key
        img.paste(self._png.crop((0, y0 - y, w, y1 - y)), (x, y0))

    # ---------------- browser helpers ---------------- #
    def do(self, js, wait=1.5):
        self.b.T(js)
        time.sleep(wait)

    def wait_for(self, cond, timeout=600, every=1.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.b.T(cond):
                    return True
            except RuntimeError:
                pass
            time.sleep(every)
        raise TimeoutError("waited for " + cond)


# ---------------------------------------------------------------------------------------------- #
# Scenario
# ---------------------------------------------------------------------------------------------- #

def ai_result(r, button, ok, fails, tries=2):
    '''Wait for the AI text after a click on button: ok appears, or one of fails (the GUI gives up after the CLI
    timeout of 15 minutes, core/aisummary.CLI_TIMEOUT). The texts are searched in the rendered page: a Markdown
    heading "### What is good" is found as "What is good". A failed call is tried again once. True if ok appeared.'''
    cond = " || ".join("!!T.find({}, 'has')".format(q(t)) for t in (ok,) + tuple(fails))
    for k in range(tries):
        if k:
            print("WARNING: no AI text, trying again", flush=True)
            r.do("T.click({})".format(q(button)), 5)
        try:
            r.wait_for(cond, 1000, 5)
        except TimeoutError:
            pass
        if r.b.T("!!T.find({}, 'has')".format(q(ok))):
            return True
    r.missing.append(button + ": no AI text")
    return False


def scenario(r, url, login_png):
    from core import airfoils, paths
    from core import simulation as sm
    from core.case import Case
    from core.planform import Planform
    from core.results import wall_surfaces
    from ui.build import BuildStep
    from ui.step1_geometry import wing_figure
    case_dir = os.path.join(paths.CASES_DIR, CASE)

    r.chapter("Introduction")
    r.card("mshGen GUI", [
        "A complete session: a new wing, from the geometry to a checked mesh and a test run.",
        "",
        "  1. Geometry        airfoil, planform, mesh origin",
        "  2. Domain          C-grid size, far field, Make mesh",
        "  3a. Slicing        slices along the span, layers",
        "  3b. Blocking       block vertices, block check, block optimiser",
        "  4. Meshing         targets, mesh optimiser, cells and gradings",
        "  5. Build & check   blockMesh, checkMesh, flagged cells, AI summary, images",
        "  6. Test run        OpenFOAM: convergence, y+, Cp, tip vortex, AI review",
        "",
        "Long computations are shown sped up."],
        seconds=11, sub="Structured C-grid meshes of wings with mshGen and blockMesh")
    r.card("Start the GUI", [
        "On the cluster, in a terminal:",
        "$ mshGen/GUI/run_gui.sh            (optional: port, default 5006)",
        "Open http://localhost:5006/app in your browser; VS Code forwards the port (PORTS tab).",
        "",
        "Log in with your user name and the password in",
        "  ~/.config/mshgen_gui/users.json   (made on the first start; only you can read it)",
        "",
        "Direct link to a case and a step:  /app?case=MegAWES_wing&step=3b"], seconds=9)
    if login_png:
        img = Image.open(login_png).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle((420, 700, 1180, 870), fill="white", outline=ORANGE, width=4)
        d.text((445, 720), "Login page", font=font(28, True), fill=ORANGE)
        d.text((445, 765), "user name: your login name", font=font(24), fill="black")
        d.text((445, 805), "password: from ~/.config/mshgen_gui/users.json", font=font(24), fill="black")
        r._save(img, 4)

    b = r.b
    # interactive 3D views: headless Firefox has no WebGL, the Plotly boxes are overlaid with static exports
    b.open(url + "?case=MegAWES_wing&step=1&view3d=interactive", wait=25)

    # ---------------- window ---------------- #
    r.chapter("The window")
    r.shot("The window", "Left: the <b>sidebar</b> with the case, Save / Revert, the status, the steps and New case. "
           "Right: the current step: inputs left, live plots right, more sections below.")
    r.shot("Header", "A fact or quote of the day, <b>Undo</b> / <b>Redo</b> of every edit, and the light or dark "
           "theme.", BTN(UNDO), also=[BTN(REDO)], where="below")
    r.shot("Case and status", "Every case is a folder in mshGen/cases with its settings in case.json. <b>Save</b> "
           "writes it, <b>Revert</b> reloads it; below: the estimated number of cells.", W_("Case"),
           also=[BTN("Save"), BTN("Revert"), HAS("Cells (estimate)")])
    r.shot("Steps", "Go back and forth at any time. Each step shows its state: ✔ done, ⚠ attention, "
           "✖ error, ○ to do, ▶ running.", L_("1. Geometry"))

    # ---------------- new case ---------------- #
    r.chapter("New case")
    r.do("T.set('Name', {})".format(q(CASE)), 1)
    r.do("T.set('Airfoil', 'Mrev-v2')", 1)
    r.shot("New case", "<b>New wing</b>: a library airfoil with a default planform; <b>From example</b>: an mshGen "
           "example script. Here: tutorial_wing with Mrev-v2, the MegAWES airfoil.", BTN("Create case"),
           click=BTN("Create case"), also=[W_("Name"), W_("Airfoil"), L_("New wing")])
    r.do("T.click('Create case')", 2)
    r.wait_for("T.find('Case') && T.find('Case').model.value === {}".format(q(CASE)), 120)
    time.sleep(6)
    b.inject()

    # ---------------- step 1 ---------------- #
    outline = airfoils.AirfoilData("Mrev-v2", airfoils.library_file("Mrev-v2")).outline()
    r.fig3d = wing_figure(Case.load(case_dir), outline)
    r.chapter("Step 1: Geometry")
    r.shot("Step 1 · Case type and airfoil", "A <b>3D</b> half wing (or a 2D section) and an airfoil of mshGen's "
           "library; the card tells what mshGen does with it.", L_("3D"),
           also=[W_("Airfoil (mshGen library)"), CARD("What mshGen does for this airfoil")], where="right")
    r.shot("Step 1 · The airfoil", "As mshGen reads it: the points, the removed point, the splines and the camber "
           "line. All plots: zoom with the wheel, click a legend entry to hide it.", HAS("as mshGen reads it"),
           where="left")
    for k, v in (("Half-span (m)", 6.0), ("Root chord (m)", 1.2), ("Tip chord (m)", 0.6),
                 ("Constant-chord part up to y (m)", 1.0), ("LE sweep outboard (deg, + backwards)", 5.0),
                 ("Twist at the root (deg, + nose up)", 2.0), ("Twist at the tip (deg)", -1.0),
                 ("Dihedral outboard (deg, + tip up)", 3.0)):
        r.do("T.set({}, {})".format(q(k), v), 1.2)
    time.sleep(3)
    case = Case.load(case_dir)
    case.wing.stations = Planform(half_span=6.0, root_chord=1.2, tip_chord=0.6, kink_y=1.0, le_sweep_deg=5.0,
                                  twist_root_deg=2.0, twist_tip_deg=-1.0, dihedral_deg=3.0).to_stations()
    r.fig3d = wing_figure(case, outline)
    r.shot("Step 1 · Planform", "The parameters make the planform: half-span 6 m, chords 1.2 / 0.6 m, constant "
           "chord up to 1 m, 5° sweep, twist 2 / −1°, 3° dihedral. Above: the mesh origin (COG).",
           TABS("Parameters"), also=[W_("COG x (m)")], where="right")
    r.shot("Step 1 · Top and front view", "The planform with the <b>near-wake line</b> (every trailing edge must "
           "stay upstream of it), and the heights along the span: dihedral and twist.",
           HAS("Top view of the half wing"), also=[HAS("Front view")], where="left")
    r.do("T.tab('Table')", 2)
    r.shot("Step 1 · Planform table", "Or edit the table: y, chord, twist, dx (sweep), dz (dihedral), linear "
           "between the rows; insert or delete rows.", TABS("Table"), where="right")
    r.do("T.tab('Parameters')", 1)
    r.shot("Step 1 · Derived values", "Span, area, aspect ratio, taper ratio, the <b>mean aerodynamic chord</b> and "
           "its position, sweep and dihedral: exact for the planform.", HAS("Derived values"), where="above")
    r.shot("Step 1 · 3D view", "The half wing in 3D: drag to rotate, scroll to zoom.", TYPED("Plotly"),
           where="left")
    r.fig3d = None

    # ---------------- step 2 ---------------- #
    r.chapter("Step 2: Domain and Make mesh")
    r.do("T.radio('2. Domain')", 5)
    b.inject()
    r.shot("Step 2 · Domain", "The C-grid radius R in root chords (the domain reaches 2R downstream) and the "
           "far-field plane beyond the tip; the plots show the domain and its patches.",
           W_("C-grid radius R / root chord (mesh_scale)"),
           also=[W_("Far field (symTip) at y (m)"), HAS("Root plane: C-domain")], where="right")
    r.shot("Step 2 · Make mesh", "One click for steps 3 and 4: it suggests the slices and the block vertices and "
           "optimises the cells for the targets of step 4. Each part can also be done by hand.", BTN(MAKE),
           click=BTN(MAKE), where="right")
    r.do("T.click({})".format(q(MAKE)), 2)
    r.wait_for("!T.button({})".format(q(MAKE)), 60)                # running: the button says so
    r.timelapse("Step 2 · Making the mesh", "Slices, block vertices (the optimiser of step 3b), then the cells (the "
                "optimiser of step 4).", HAS("Blocking and meshing"), "!!T.button({})".format(q(MAKE)), every=5,
                where="right")
    time.sleep(3)
    r.shot("Step 2 · Mesh made", "Steps 3 and 4 are done: see their marks. <b>Undo</b> would take it all back in "
           "one step.", L_("3a. Spanwise slicing"), also=[BTN(UNDO)], where="right")

    # ---------------- step 3a ---------------- #
    r.chapter("Step 3a: Spanwise slicing")
    r.do("T.radio('3a. Spanwise slicing')", 5)
    b.inject()
    r.shot("Step 3a · Slices", "The planes along the span where the block vertices are placed: root, kink, around "
           "the tip, far field (here from Make mesh). Edit y in the table.", TYPED("Tabulator", 0),
           also=[HAS("dashed: slices")], where="right")
    r.do("T.set('New slice near y (m)', 3.0)", 1)
    r.shot("Step 3a · Add, delete, suggest", "<b>Add slice</b> near y = 3 m (on a cell face), <b>Delete "
           "selected</b> slices, or <b>Suggest slices</b> again.", BTN("Add slice"), click=BTN("Add slice"),
           also=[W_("New slice near y (m)"), BTN("Delete selected"), BTN("Suggest slices")], where="right")
    r.do("T.click('Add slice')", 4)
    r.shot("Step 3a · Layers", "Options: an <b>aileron cut-out</b> in wing layers, the first wing layers as patch "
           "'extra' (ARTIFICIOUS), and slices that scale with the span.",
           W_("Aileron cut-out in layers (consecutive)"),
           also=[W_("First wing layers as separate patch 'extra' (ARTIFICIOUS)"),
                 W_("Scale the slices when the span changes")], where="right")

    # ---------------- step 3b ---------------- #
    r.chapter("Step 3b: Airfoil blocking")
    r.do("T.radio('3b. Airfoil blocking')", 6)
    b.inject()
    r.shot("Step 3b · Blocks of a slice", "Exactly as mshGen generates them: wake, upper and lower side; skewed "
           "blocks orange, folded red. Hover a block for its corner angles.", HAS("blocks of layer"), where="left")
    r.do("T.radio('Airfoil')", 3)
    r.do("T.drag(7, 0.15, 0.1)", 4)
    r.shot("Step 3b · Airfoil vertices", "On the suction side, the pressure side or the camber line, at a position "
           "from 0 (LE) to 1 (TE): edit the table or drag them in the plot (here vertex 7).",
           TYPED("Tabulator", 0), also=[HAS("blocks of layer")], where="right")
    r.do("T.click({})".format(q(UNDO)), 5)
    b.inject()
    r.shot("Undo", "<b>Undo</b> takes any edit back, here the drag; <b>Redo</b> repeats it.", BTN(UNDO),
           also=[HAS("blocks of layer")], where="below")
    r.do("(function(){const v = T.find('Slice'); if (v) v.model.value = v.model.end - 1; return !!v})()", 4)
    r.shot("Step 3b · Other slices", "The slider selects the slice (red in the top view). Beyond the tip, blocks "
           "inside the airfoil contour fill the space behind the tip.", W_("Slice"),
           also=[HAS("Top view: slice")], where="left")
    r.do("T.set('Slice', 0)", 3)
    r.do("T.click('Suggest block vertices')", 3)
    r.wait_for("!!T.find('Suggested blocks dashed', 'has') || !!T.find('No suggestion', 'has')", 600, 2)
    time.sleep(3)
    r.shot("Step 3b · Suggest block vertices", "The block optimiser: grid lines normal to the wall, corners near "
           "90°, smooth expansion. The suggestion is dashed, with a before / after table: <b>Apply</b> or "
           "<b>Discard</b>.", BTN("Suggest block vertices"), also=[TYPED("Tabulator", 0), HAS("blocks of layer")],
           where="right")
    r.do("T.click('Apply')", 4)
    r.do("T.check('Wing surface as wingUp / wingDown (SPLIT)', true)", 3)
    r.shot("Step 3b · Exterior vertices and block check", "Below: the exterior vertices (on their boundary; drag "
           "them too), the block check of all slices, and <b>SPLIT</b>: the wing surface as wingUp / wingDown.",
           W_("Wing surface as wingUp / wingDown (SPLIT)"),
           also=[HAS("Block vertices and block check"), HAS("Exterior vertices, block check and patches")],
           where="left")

    # ---------------- step 4 ---------------- #
    r.chapter("Step 4: Meshing")
    r.do("T.radio('4. Meshing')", 7)
    b.inject()
    r.shot("Step 4 · Level and targets", "<b>COARSE</b>, MEDIUM or FINE loads the targets: y+, chordwise and "
           "spanwise cells, growth ratio, max cells; <b>Reset</b> restores them. U and ν set the y+.",
           W_("Mesh level"), also=[BTN("Reset to the level presets"), W_("Target y+ (first cell centre)"),
                                   W_("Free-stream velocity U (m/s)")], where="right")
    r.do("T.set('Free-stream velocity U (m/s)', 40)", 3)
    r.do("T.set('Target y+ (first cell centre)', 30)", 3)
    r.shot("Step 4 · Optimise mesh", "Here: y+ 30 at U = 40 m/s. <b>Optimise mesh</b> computes the cells and "
           "gradings of all edges and layers for these targets.", BTN("Optimise mesh"),
           click=BTN("Optimise mesh"), where="right")
    r.do("T.click('Optimise mesh')", 2)
    r.wait_for("!!T.button('Optimising ...') || !!T.find('Optimisation result', 'has')", 60)
    r.timelapse("Step 4 · Optimising", "Sizing for the targets, the outer gradings on the exact block mesh, the y+ "
                "correction.", "(T.button('Optimising ...') || T.button('Optimise mesh'))",
                "!!T.find('Optimisation result', 'has')", every=4, where="right")
    time.sleep(6)
    b.inject()
    r.shot("Step 4 · Result", "Before and after against the targets: y+, cells, growth, LE / TE and wake "
           "resolution, non-orthogonality. <b>Undo</b> under the button restores the cells of before.",
           HAS("Optimisation result"), where="right")
    r.do("T.set('Colour by', 'aspect ratio, 3D (with spanwise size)')", 3)
    r.do("T.radio('Leading edge')", 3)
    r.shot("Step 4 · Mesh preview", "The mesh of a slice exactly as blockMesh builds it, coloured by cell size, "
           "aspect ratio (here 3D, at the leading edge), size ratio or corner angle; hover a cell.",
           HAS("Slice 0 at y"), also=[W_("Colour by")], where="left")
    r.do("T.set('Colour by', 'cell size (m)')", 2)
    r.do("T.radio('Airfoil')", 2)
    r.shot("Step 4 · Edges and layers", "Cells and gradings of every block edge and spanwise layer, editable; the "
           "plot shows each edge with its cell nodes (select a row to highlight it).", TYPED("Tabulator", 0),
           also=[HAS("edges and cell nodes"), TYPED("Tabulator", 1)], where="right")
    r.shot("Step 4 · Detailed analysis", "Every slice in one table, with warnings for checkMesh limits and the "
           "targets; the spanwise cell size and the cells along the airfoil.", HAS("All slices"),
           also=[HAS("Spanwise cell size")], where="left")

    # ---------------- step 5 ---------------- #
    r.chapter("Step 5: Build & check")
    r.do("T.radio('5. Build & check')", 6)
    b.inject()
    r.do("T.check('export to Fluent (foamMeshToFluent)', true)", 1)
    r.shot("Step 5 · Build", "<b>Save and build</b>: saves the case, checks the blocks and runs mshGen, blockMesh, "
           "checkMesh, the images (on by default) and here the Fluent export.",
           BTN("Save and build mesh (Allrun.pre)"), click=BTN("Save and build mesh (Allrun.pre)"),
           also=[W_("also make images (ParaView views)"), W_("export to Fluent (foamMeshToFluent)")], where="right")
    r.do("T.click('Save and build mesh (Allrun.pre)')", 10)
    r.wait_for("!!T.find('$ ./Allrun.pre', 'has')", 120)
    r.timelapse("Step 5 · Building", "The log of the build, live: mshGen, blockMesh, checkMesh, the Fluent export "
                "and the images.", HAS("$ ./Allrun.pre"), "!!T.find('== exit code', 'has')", every=12, where="below",
                before="T.scrollEnd('$ ./Allrun.pre')")
    time.sleep(10)
    b.inject()
    r.wait_for("!!T.find('Top view (flow downwards)', 'has')", 300)
    time.sleep(3)
    r.shot("Step 5 · checkMesh and flagged cells", "The checkMesh summary and the cells that fail it or have a "
           "lower quality: a table and a top and side view (whole domain or the wing).",
           HAS("Top view (flow downwards)"), also=[HAS("Side view (looking along the span)")], where="above")
    r.shot("Step 5 · Fluent mesh", "The zones of fluentInterface/tutorial_wing.msh; set <b>inlet</b> to "
           "velocity-inlet in Fluent.", HAS("pressure-outlet"), where="right")
    r.do("T.click('Write AI summary')", 4)
    r.shot("Step 5 · AI summary", "<b>Write AI summary</b>: Claude reviews the mesh through Claude Code with your "
           "login (no API key); only technical facts are sent, shown below the button.", BTN("Writing ..."),
           where="right")
    if ai_result(r, "Write AI summary", "What is good", ("No summary", "Failed:")):
        time.sleep(4)
        b.inject()
        r.shot("Step 5 · The summary", "What is good, a modelling approach, improvements, assumptions and a crazy "
               "idea; saved as ai_summary.md.", HAS("What is good"), where="left")
    r.do("T.click('Show 3D view of the walls')", 3)
    r.wait_for("!!T.typed('Plotly')", 120)
    time.sleep(3)
    r.fig3d = BuildStep.figure_3d(wall_surfaces(case_dir), True)
    r.shot("Step 5 · 3D view of the walls", "The wall patches of the built mesh with cell edges: drag to rotate.",
           TYPED("Plotly"), where="left")
    r.fig3d = None
    r.wait_for("!!T.find('Images (last build with images)', 'has')", 120)
    r.shot("Step 5 · Images", "ParaView views of the build: root plane, leading and trailing edge, wing, tip, cuts, "
           "and the mesh inside the wing contour beyond the tip.",
           HAS("14_interior_wing_mesh_beyond_tip.png", 1), where="left")    # the caption: the log names it first

    # ---------------- step 6 ---------------- #
    ncpu = os.cpu_count() or 1
    cores = max(2, min(16, ncpu - int(round(sm.busy_processes())) - 2))     # never more than the free cores
    cores_title = "Cores (this machine has {})".format(ncpu)
    r.chapter("Step 6: Test run")
    r.do("T.radio('6. Test run')", 6)
    b.inject()
    r.do("T.set({}, {})".format(q(cores_title), cores), 2)
    r.do("T.set('Iterations', 60)", 2)
    r.shot("Step 6 · Inputs", "A short simpleFoam run on this mesh: cores, velocity, angle of attack, turbulence "
           "model, iterations, section planes. Here: {} cores, 60 iterations.".format(cores), W_(cores_title),
           also=[W_("Angle of attack alpha (deg)"), W_("Turbulence model"), W_("Iterations")], where="right")
    r.do("T.card('Settings of the test run', true)", 3)
    r.shot("Step 6 · Machine load and set-up", "The busy processes of this machine (a warning before sharing cores), "
           "and the whole set-up: boundary conditions, schemes, solvers, reference values.", HAS("Machine now"),
           also=[CARD("Settings of the test run")], where="below")
    r.do("T.card('Settings of the test run', false)", 2)
    r.shot("Step 6 · Run", "<b>Save and run</b>: decomposePar, simpleFoam in parallel and reconstructPar, in run/ "
           "of the case. <b>Stop</b> writes the current iteration and ends.", BTN("Save and run (Allrun)"),
           click=BTN("Save and run (Allrun)"), also=[BTN("Stop (write and end)")], where="right")
    r.do("T.click('Save and run (Allrun)')", 5)
    r.wait_for("!!T.find('· running', 'has')", 300, 2)
    r.timelapse("Step 6 · Running", "The residuals and the force coefficients while the solver runs.",
                HAS("Convergence"), "!!T.find('finished (exit code', 'has')", every=4, where="above")
    r.wait_for("!!T.find('Reversed flow at the wall', 'has')", 600, 2)
    time.sleep(8)
    b.inject()
    r.shot("Step 6 · Convergence", "The initial residuals and Cl, Cd, Cm; the range of the coefficients leaves out "
           "the first iterations. Hover for values.", HAS("Convergence"), where="above")
    r.shot("Step 6 · y+", "y+ of the solution per wall patch next to the prediction of step 4, and along the "
           "airfoil at the section.", HAS("y+ of the solution and the prediction"), where="left")
    r.shot("Step 6 · Pressure", "Cp along both sides at a section (choose it; red x: reversed flow) and the share of "
           "the wing with reversed flow.", W_("Section"), also=[HAS("Reversed flow at the wall")], where="right")
    r.shot("Step 6 · The wing from above", "Cp on both sides of the wing, reversed flow dotted, the section planes "
           "dashed.", HAS("The wing seen from above"), where="above")
    r.shot("Step 6 · Contours", "Cp and the velocity in the section plane around the airfoil.",
           HAS("Contours of pressure and velocity"), where="left")
    r.shot("Step 6 · Tip vortex", "Streamwise vorticity and the in-plane velocity in two crossflow planes behind "
           "the wing.", HAS("Tip vortex"), where="left")
    r.do("T.click('Write AI review')", 5)
    review = ("<b>Write AI review</b>: Claude judges the convergence, the plausibility of the solution and the "
              "effects of the mesh, and recommends changes; saved as run/ai_review.md.")
    if ai_result(r, "Write AI review", "Recommendations", ("No review", "Failed:")):
        time.sleep(5)
        b.inject()
        r.shot("Step 6 · AI review", review, HAS("Plausibility of the solution"), where="left")
    else:
        r.shot("Step 6 · AI review", review, BTN("Write AI review"), where="right")

    r.chapter("Files and next steps")
    r.card("The case folder", [
        "cases/tutorial_wing/",
        "  case.json          all settings (source of truth)",
        "  airfoil.txt, wing.txt, settings/*.csv, generate_mesh.py, images.json",
        "  Allrun.pre         rebuilds everything without the GUI:  ./Allrun.pre [--images]",
        "  constant/polyMesh  OpenFOAM mesh;  fluentInterface/tutorial_wing.msh  Fluent mesh",
        "  images/            ParaView views;  ai_summary.md",
        "  run/               test run of step 6 (Allrun, logs, results, ai_review.md)",
        "",
        "Command line: python GUI/mshgen_case.py list | new | import | write | blocks | build | check",
        "Documentation: GUI/README.md, GUI/PLAN.md, LIMITATIONS.md"], seconds=10)


def assemble(r, out, fps=10):
    '''Frames + hold times -> H.264 MP4 with chapters (ffmpeg from imageio-ffmpeg); longer than MAX_VIDEO: the hold
    times of the normal frames are scaled down (at least 3 s; sped-up frames stay).'''
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    total = sum(sec for _, sec in r.frames)
    if total > MAX_VIDEO:
        fixed = sum(sec for _, sec in r.frames if sec <= 1.0)
        k = (MAX_VIDEO - fixed) / (total - fixed)
        r.frames = [(p, sec if sec <= 1.0 else max(3.0, sec * k)) for p, sec in r.frames]
        print("hold times scaled by {:.3f}: {:.0f} s -> {:.0f} s".format(k, total, sum(s for _, s in r.frames)))
    lst = os.path.join(r.dir, "frames.txt")
    with open(lst, "w") as f:
        for path, sec in r.frames:
            f.write("file '{}'\nduration {:.2f}\n".format(path, sec))
        f.write("file '{}'\n".format(r.frames[-1][0]))
    starts, t = [], 0.0
    for _, sec in r.frames:
        starts.append(t)
        t += sec
    meta = os.path.join(r.dir, "chapters.txt")
    with open(meta, "w") as f:
        f.write(";FFMETADATA1\ntitle=mshGen GUI tutorial\n")
        for i, (title, k) in enumerate(r.chapters):
            end = starts[r.chapters[i + 1][1]] if i + 1 < len(r.chapters) else t
            f.write("[CHAPTER]\nTIMEBASE=1/1000\nSTART={}\nEND={}\ntitle={}\n".format(
                int(starts[k] * 1000), int(end * 1000), title))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", meta,
                    "-map_metadata", "1", "-vf", "fps={},format=yuv420p".format(fps), "-c:v", "libx264",
                    "-preset", "slow", "-crf", "24", "-movflags", "+faststart", out], check=True)
    chapters = ["{:d}:{:02d}  {}".format(int(starts[k]) // 60, int(starts[k]) % 60, title) for title, k in r.chapters]
    return t, chapters


def login_screenshot(port, workdir):
    '''Screenshot of the login page (server with basic authentication and a throwaway password file).'''
    pw = os.path.join(workdir, "users.json")
    with open(pw, "w") as f:
        json.dump({"user": "tutorial"}, f)
    log = open(os.path.join(workdir, "panel_login.log"), "w")
    srv = subprocess.Popen(["panel", "serve", "app.py", "--port", str(port), "--address", "127.0.0.1",
                            "--allow-websocket-origin", "127.0.0.1:{}".format(port), "--basic-auth", pw,
                            "--cookie-secret", "tutorialtutorialtutorial"], cwd=GUI, stdout=log, stderr=log)
    try:
        time.sleep(15)
        with Browser(W, H, workdir) as b:
            b.cmd("WebDriver:Navigate", {"url": "http://127.0.0.1:{}/app".format(port)})
            time.sleep(4)
            path = os.path.join(workdir, "login.png")
            with open(path, "wb") as f:
                f.write(b.screenshot())
            return path
    except Exception as e:  # noqa: BLE001
        print("no login screenshot:", e)
        return None
    finally:
        srv.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(GUI, "docs", "mshGen_GUI_tutorial.mp4"))
    ap.add_argument("--frames", default=None, help="keep the frames in this folder")
    ap.add_argument("--until", default=None, help="stop before the chapter that starts with this text")
    a = ap.parse_args()
    from core import paths
    case_dir = os.path.join(paths.CASES_DIR, CASE)
    shutil.rmtree(case_dir, ignore_errors=True)
    work = tempfile.mkdtemp(prefix="mshgen_tutorial_")
    frames = a.frames or os.path.join(work, "frames")
    port = 5930 + os.getpid() % 40
    login = login_screenshot(port + 1, work)
    log = open(os.path.join(work, "panel.log"), "w")
    srv = subprocess.Popen(["panel", "serve", "app.py", "--port", str(port), "--address", "127.0.0.1",
                            "--allow-websocket-origin", "127.0.0.1:{}".format(port)], cwd=GUI, stdout=log, stderr=log)
    r = Recorder(frames, a.until)
    try:
        time.sleep(15)
        r.b = Browser(W, H, work)
        try:
            scenario(r, "http://127.0.0.1:{}/app".format(port), login)
        except StopRecording as e:
            print("stopped before", e)
        total, chapters = assemble(r, a.out)
        with open(os.path.splitext(a.out)[0] + "_chapters.txt", "w") as f:
            f.write("\n".join(chapters) + "\n")
        print("written {} ({:d}:{:02d}, {} frames, {:.1f} MB)".format(a.out, int(total) // 60, int(total) % 60,
                                                                       len(r.frames), os.path.getsize(a.out) / 1e6))
        print("\n".join(chapters))
        if r.missing:
            print("elements not found (no outline): " + "; ".join(r.missing))
    finally:
        if r.b:
            r.b.close()
        srv.kill()
        shutil.rmtree(case_dir, ignore_errors=True)
        if not a.frames:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
