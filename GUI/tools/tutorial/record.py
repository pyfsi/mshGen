'''
record.py: records the tutorial video of the mshGen GUI (GUI/docs/mshGen_GUI_tutorial.mp4)

A headless Firefox drives the real GUI through a complete session: create a wing case and go through all five
steps, using or pointing at every input and option. Each frame shows the GUI with the explained element
outlined and a pop-up message next to it; frames are held long enough to read the message. No audio.
The 3D views (Plotly, WebGL) cannot be drawn by headless Firefox: the same figures are rendered with Plotly's
static export (kaleido) and placed over them.

Usage (Anaconda module loaded; needs OpenFOAM and ParaView for the build in step 5, about 15 minutes):
    python GUI/tools/tutorial/record.py [--out GUI/docs/mshGen_GUI_tutorial.mp4] [--frames DIR]
The tutorial case cases/tutorial_wing is created and deleted again.
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


def PROPOSE(tab):
    return "T.buttonIn({}, 'Propose')".format(q(tab))


def CARD(title):
    return "T.el(T.cardview({}))".format(q(title))


def font(size, bold=False):
    import matplotlib
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(path, size)


class Recorder:

    def __init__(self, frames_dir):
        self.dir = frames_dir
        os.makedirs(frames_dir, exist_ok=True)
        self.frames = []            # (file, seconds)
        self.chapters = []          # (title, frame index)
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
        return max(4.0, min(16.0, 2.5 + 0.36 * words))

    def chapter(self, title):
        self.chapters.append((title, len(self.frames)))

    def card(self, title, lines, seconds=None, sub=None):
        img = Image.new("RGB", FRAME, (44, 110, 155))
        d = ImageDraw.Draw(img)
        d.text((110, 110), title, font=font(60, True), fill="white")
        if sub:
            d.text((110, 200), sub, font=font(30), fill=(220, 235, 245))
        y = 290
        for line in lines:
            mono = line.startswith("$ ") or line.startswith("  ")
            d.text((130 if mono else 110, y), line, font=font(28 if not mono else 26), fill="white")
            y += 50
        self._save(img, seconds or self.hold(" ".join([title] + lines)))

    def shot(self, title, body, target=None, where=None, click=None, seconds=None, also=(), composite=None):
        '''Outline target (and the elements in also), show the pop-up, screenshot.'''
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

def scenario(r, url, login_png):
    from core import paths, sizing
    from core.case import Case
    from core.planform import Planform, section_params

    r.chapter("Introduction")
    r.card("mshGen GUI", [
        "This video shows every input and option of the GUI, in a complete session:",
        "a new wing case, from the geometry to a checked OpenFOAM and Fluent mesh.",
        "",
        "  1. Geometry      airfoil, planform, mesh origin",
        "  2. Domain        size of the C-grid, far field, flow conditions",
        "  3. Blocks        slices, chordwise divisions, exterior vertices",
        "  4. Meshing       cells and gradings, sizing helpers, quality optimiser",
        "  5. Build & check blockMesh, checkMesh, y+, Fluent export, images"],
        seconds=12, sub="Structured C-grid meshes of wings with mshGen and blockMesh")
    r.card("Start the GUI", [
        "On the cluster, in a terminal:",
        "$ mshGen/GUI/run_gui.sh            (optional: port, default 5006)",
        "Open http://localhost:5006/app in your browser; VS Code forwards the port",
        "(PORTS tab). The server listens on this machine only.",
        "",
        "Log in with your user name and the password stored in",
        "  ~/.config/mshgen_gui/users.json   (created on the first start, only you can read it)",
        "",
        "Direct links: /app?case=MegAWES_wing&step=3 opens a case at a step."], seconds=14)
    if login_png:
        img = Image.open(login_png).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle((420, 700, 1180, 870), fill="white", outline=ORANGE, width=4)
        d.text((445, 720), "Login page", font=font(28, True), fill=ORANGE)
        d.text((445, 765), "user name: your login name", font=font(24), fill="black")
        d.text((445, 805), "password: from ~/.config/mshgen_gui/users.json", font=font(24), fill="black")
        r._save(img, 6)

    b = r.b
    # interactive 3D views: headless Firefox has no WebGL, the Plotly boxes are overlaid with static exports
    b.open(url + "?case=MegAWES_wing&step=1&view3d=interactive", wait=25)

    # ---------------- sidebar ---------------- #
    r.chapter("The window and the sidebar")
    r.shot("The window", "Left: the <b>sidebar</b> with the case, Save/Revert, the status, the steps and "
           "<b>New case</b>. Right: the current step, with the inputs on the left and live plots on the right.")
    r.shot("Case", "Choose a case. Every case is a folder in <b>mshGen/cases</b> with its settings in "
           "<b>case.json</b> and everything needed to rebuild the mesh, also without the GUI.", W_("Case"))
    r.shot("Save and Revert", "All edits are kept in memory first. <b>Save</b> writes case.json and all case files "
           "(and recomputes the block vertices if the geometry changed). <b>Revert</b> reloads the saved case.",
           BTN("Save"), also=[BTN("Revert")])
    r.shot("Status", "Unsaved changes, the <b>estimated number of cells</b> (exact), where the block vertices come "
           "from, the <b>block check</b> (no folded blocks) and problems of the settings.",
           HAS("Cells (estimate)"))
    r.shot("Steps", "The five steps. You can go back and forth at any time; every step works on the same case.",
           L_("1. Geometry"))
    r.chapter("New case")
    r.shot("New case", "<b>New wing</b>: an airfoil of the mshGen library with a default planform. "
           "<b>From example</b>: import any script of mshGen/examples as a case.", L_("New wing"))
    r.do("T.set('Name', 'tutorial_wing')", 1)
    r.do("T.set('Airfoil', 'Mrev-v2')", 1)
    r.shot("Create a wing case", "Name <b>tutorial_wing</b>, airfoil <b>Mrev-v2</b> (the MegAWES airfoil). "
           "<b>Create case</b> computes mshGen's default block vertices and opens the case.",
           BTN("Create case"), click=BTN("Create case"), also=[W_("Name")])
    r.do("T.click('Create case')", 2)
    r.wait_for("T.find('Case') && T.find('Case').model.value === 'tutorial_wing'", 120)
    time.sleep(6)
    b.inject()

    # ---------------- step 1 ---------------- #
    from core import airfoils
    from ui.step1_geometry import wing_figure
    outline = airfoils.AirfoilData("Mrev-v2", airfoils.library_file("Mrev-v2")).outline()
    r.fig3d = wing_figure(Case.load(os.path.join(paths.CASES_DIR, CASE)), outline)
    r.chapter("Step 1: Geometry")
    r.shot("Step 1 · Case type", "<b>3D</b>: a half wing with symmetry at the root. <b>2D</b>: an airfoil section "
           "(one cell thick, chord 1).", L_("3D"))
    r.shot("Step 1 · Airfoil", "The airfoils of the mshGen library. Only these can be used (mshGen selects its "
           "built-in behaviour by the airfoil name; see LIMITATIONS.md).", W_("Airfoil (mshGen library)"))
    r.do("T.card('What mshGen does for this airfoil', true)", 2)
    r.shot("Step 1 · What mshGen does", "For the selected airfoil: point clean-up, camber line, default vertex "
           "positions and for which C-grid sizes default resolutions exist.",
           CARD("What mshGen does for this airfoil"))
    r.do("T.card('What mshGen does for this airfoil', false)", 1)
    r.shot("Step 1 · Airfoil as mshGen reads it", "The points used, the point mshGen removes (red x), the "
           "spline of each side and the camber line. All plots: zoom with the mouse wheel, pan by dragging, "
           "click a legend entry to hide it.", HAS("as mshGen reads it"), where="left")
    r.shot("Step 1 · Mesh origin (COG)", "The origin of the mesh (centre of gravity): the root section is placed "
           "relative to it. Shown in the root-section plot.", W_("COG x (m)"), also=[W_("COG z (m)")])
    r.shot("Step 1 · Planform parameters", "A straight-tapered wing with an optional constant-chord part: "
           "half-span, root and tip chord, LE sweep, twist at root and tip, dihedral. They generate the "
           "planform table.", TABS("Parameters"), where="right")
    for k, v in (("Half-span (m)", 6.0), ("Root chord (m)", 1.2), ("Tip chord (m)", 0.6),
                 ("Constant-chord part up to y (m)", 1.0), ("LE sweep outboard (deg, + backwards)", 5.0),
                 ("Twist at the root (deg, + nose up)", 2.0), ("Twist at the tip (deg)", -1.0),
                 ("Dihedral outboard (deg, + tip up)", 3.0)):
        r.do("T.set({}, {})".format(q(k), v), 1.2)
    time.sleep(3)
    case = Case.load(os.path.join(paths.CASES_DIR, CASE))
    p = Planform(half_span=6.0, root_chord=1.2, tip_chord=0.6, kink_y=1.0, le_sweep_deg=5.0,
                 twist_root_deg=2.0, twist_tip_deg=-1.0, dihedral_deg=3.0)
    case.wing.stations = p.to_stations()
    r.fig3d = wing_figure(case, outline)
    r.shot("Step 1 · A new planform", "Half-span 6 m, chords 1.2 / 0.6 m, constant chord up to 1 m, 5 deg sweep, "
           "twist 2 / -1 deg, 3 deg dihedral. Plots and the table follow immediately.",
           TABS("Parameters"), where="right")
    r.shot("Step 1 · Top view", "The planform with the <b>slices</b> (dashed) and the <b>near-wake line</b>: "
           "mshGen's exterior vertices are the same in every slice, so every trailing edge must stay upstream "
           "of it (limits sweep and twist).", HAS("Top view of the half wing"), where="left")
    r.shot("Step 1 · Front view", "Height of the leading and trailing edge along the span: dihedral and twist.",
           HAS("Front view"), where="left")
    r.shot("Step 1 · 3D view", "The half wing in 3D: drag to rotate, scroll to zoom.", TYPED("Plotly"),
           where="left")
    r.shot("Step 1 · Derived values", "Span, area, aspect ratio, taper ratio, mean aerodynamic chord, sweep, twist "
           "and tip height of the planform.", HAS("half-span (m)"), where="left")
    r.do("T.tab('Table')", 2)
    r.shot("Step 1 · Planform table", "Or edit the table directly (double-click a cell): station, y, chord, twist "
           "(beta), dx (upstream shift: sweep) and dz (dihedral). <b>Insert row</b> / <b>Delete selected rows</b>. "
           "Values are interpolated linearly between rows.", TABS("Table"), where="right")
    r.do("T.tab('Parameters')", 1)
    r.shot("Step 1 · Scale the slices", "When the span changes, the slices are scaled with it, so the tip slice "
           "stays at the tip.", W_("Scale the slices when the span changes"))
    r.shot("Step 1 · Block vertices", "The root chord changed, so mshGen's default block vertices no longer fit: "
           "they are recomputed when you save, or now with the button. Option: also reset the edge resolution "
           "to mshGen's defaults.", HAS("since the block vertices were computed"),
           also=[BTN("Recompute block vertices (mshGen defaults)")], where="right")
    r.do("T.click('Recompute block vertices (mshGen defaults)')", 3)
    r.wait_for("!!T.find('all blocks valid', 'has')", 60)
    r.shot("Step 1 · Block check", "The block check runs after every change (0.1 s): every block in every slice "
           "must be valid, otherwise blockMesh would fail. Skewed blocks are shown as warnings.",
           HAS("all blocks valid"), where="right")
    r.shot("Save", "Save the case: case.json and all files of cases/tutorial_wing are written.", BTN("Save"),
           click=BTN("Save"))
    r.do("T.click('Save')", 8)

    # ---------------- step 2 ---------------- #
    r.fig3d = None
    r.chapter("Step 2: Domain")
    r.do("T.radio('2. Domain')", 5)
    b.inject()
    r.shot("Step 2 · C-grid radius", "Radius R of the C-shaped domain in root chords. The domain reaches 2R "
           "downstream (fixed by mshGen).", W_("C-grid radius R / root chord (mesh_scale)"))
    r.shot("Step 2 · Far field", "Position of the far-field symmetry plane (symTip) beyond the tip.",
           W_("Far field (symTip) at y (m)"))
    r.shot("Step 2 · Root plane", "The C-domain at the root with the root section; inlet: C-front, top and "
           "bottom; outlet in red.", HAS("Root plane: C-domain"), where="left")
    r.shot("Step 2 · Domain from above", "Planform and patches (symRoot, symTip, inlet, outlet).",
           HAS("Domain seen from above"), where="left")

    # ---------------- step 3 ---------------- #
    r.chapter("Step 3: Blocks")
    r.do("T.radio('3. Blocks')", 5)
    b.inject()
    r.shot("Step 3 · Slices", "Planes along the span where the block vertices are placed: root, planform kinks, "
           "around the tip, tip, far field. Edit y in the table to move a slice (root and tip are fixed).",
           TYPED("Tabulator", 0), where="right")
    r.do("T.set('New slice near y (m)', 2.0)", 1)
    r.shot("Step 3 · Add a slice", "<b>Add slice</b> near y = 2 m: it is placed at the nearest cell face, so the "
           "cells do not change. <b>Delete selected</b> merges two layers; <b>Suggest slices</b> proposes root, "
           "kinks, tip region and far field.", BTN("Add slice"), click=BTN("Add slice"),
           also=[BTN("Delete selected"), BTN("Suggest slices"), W_("New slice near y (m)")])
    r.do("T.click('Add slice')", 4)
    r.shot("Step 3 · Slice added", "The new slice and its layers (type, cells, grading).", TYPED("Tabulator", 0),
           where="right")
    r.shot("Step 3 · Chordwise divisions", "Where the airfoil surface is divided into the edges e03 (TE) ... e07 "
           "(LE), as positions along the splines (about x/c), on both sides or separately.",
           TYPED("Tabulator", 1), also=[W_("same position on both sides")], where="right")
    r.shot("Step 3 · More vertices", "<b>All airfoil vertices</b> (advanced) and the <b>exterior vertices</b> as "
           "tables.", CARD("All airfoil vertices (advanced)"), also=[CARD("Exterior vertices (table)")],
           where="right")
    r.do("T.check('Wing surface as wingUp / wingDown (SPLIT)', true)", 2)
    r.shot("Step 3 · Cut-outs and patches", "<b>Aileron cut-out</b> in selected layers; <b>SPLIT</b>: wing surface "
           "as wingUp / wingDown (e.g. for FSI); <b>ARTIFICIOUS</b>: the first wing layers as a separate patch.",
           W_("Aileron cut-out in layers (consecutive)"),
           also=[W_("Wing surface as wingUp / wingDown (SPLIT)"),
                 W_("First wing layers as separate patch 'extra' (ARTIFICIOUS)")], where="right")
    r.shot("Step 3 · Block layout", "The blocks of a slice exactly as mshGen generates them: wake, upper and "
           "lower side; skewed blocks orange, folded ones red; edge names e00-e10. Hover a block for its number "
           "and corner angles.", HAS("blocks of layer"), where="left")
    r.do("T.radio('Airfoil')", 3)
    r.shot("Step 3 · Zoom", "Zoom presets: whole domain, airfoil, trailing edge (or the mouse wheel).",
           L_("Airfoil"), where="below")
    r.do("T.set('Slice', 4)", 3)
    r.shot("Step 3 · Other slices", "The slider selects the slice: at the tip and beyond, the blocks inside the "
           "airfoil contour fill the space behind the tip.", W_("Slice"), where="below")
    r.do("T.radio('Whole domain')", 2)
    r.do("T.set('Slice', 0)", 3)
    r.do("T.drag(27, -0.4, 0)", 4)
    r.shot("Step 3 · Drag exterior vertices", "Move the purple exterior vertices with the point-draw tool of the "
           "toolbar: they stay on their boundary (here vertex 27 moved along the top). Moved vertices are custom: "
           "they are no longer recomputed automatically.", HAS("blocks of layer"), where="left")
    r.do("T.click('Recompute block vertices (mshGen defaults)')", 3)
    r.wait_for("!!T.find('all blocks valid', 'has')", 60)
    r.shot("Step 3 · Back to the defaults", "The block-vertex panel resets the vertices to mshGen's defaults.",
           HAS("Block vertices are mshGen"), where="right")

    # ---------------- step 4 ---------------- #
    r.chapter("Step 4: Meshing - helpers")
    r.do("T.radio('4. Meshing')", 6)
    b.inject()
    r.shot("Step 4 · Level", "The level of mshGen's default resolution (starting point); only levels mshGen has "
           "for this airfoil and R/c are offered. Choosing one loads its default cells and gradings.",
           W_("mshGen default settings level (mesh_level)"), where="right")
    r.do("T.set('Free-stream velocity U (m/s)', 40)", 3)
    r.shot("Step 4 · Flow conditions", "Velocity U = 40 m/s and viscosity nu: the Reynolds number and a first "
           "estimate of the wall-normal first cell and its y+ (flat-plate skin friction).",
           W_("Free-stream velocity U (m/s)"), also=[W_("Kinematic viscosity nu (m2/s)"), HAS("y+ at the first cell centre")],
           where="right")
    r.shot("Step 4 · Helpers", "Compute cells and gradings from target sizes. Each helper shows the old and new "
           "values and the predicted result (measured on the block mesh) before you apply it.",
           CARD("Helpers: cells and gradings from target sizes"), where="right")
    r.do("T.set('Target y+ (first cell centre)', 30)", 1)
    r.do("T.radio('limit growth ratio')", 1)
    r.shot("Step 4 · Wall y+", "Target y+ at the first cell centre (median or maximum over the wing), keeping the "
           "wall-normal cells or limiting the growth ratio. Sets e00 (e09/e10 follow).",
           W_("Target y+ (first cell centre)"), also=[L_("limit growth ratio"), W_("max growth ratio")],
           click=PROPOSE("Wall y+"), where="right")
    r.do(PROPOSE("Wall y+") + ".click()", 2)
    r.wait_for("!!T.button('Apply proposal')", 180)
    time.sleep(2)
    r.shot("Step 4 · Proposal", "The change (old -> new) and the prediction: cells, y+ on the wing and the tip "
           "face, size jumps, aspect ratio. <b>Apply proposal</b> or <b>Discard</b>.",
           BTN("Apply proposal"), click=BTN("Apply proposal"), where="right")
    r.do("T.click('Apply proposal')", 6)
    for tab_name, text, fields in (
            ("Chordwise", "Cell size at the TE and LE, the largest cell and the growth ratio: cells and gradings "
                          "of e03-e07, continuous at the divisions.",
             ["cell at the TE (m)", "cell at the LE (m)", "largest cell (m)"]),
            ("Wake", "First wake cell at the TE, largest cell and growth: e02 (near wake) and e01 (to the outlet).",
             ["first wake cell at the TE (m)", "largest wake cell (m)"])):
        r.do("T.tab({})".format(q(tab_name)), 2)
        r.shot("Step 4 · " + tab_name, text, W_(fields[0]), also=[W_(f) for f in fields[1:]],
               click=PROPOSE(tab_name), where="right")
        r.do(PROPOSE(tab_name) + ".click()", 2)
        r.wait_for("!!T.button('Apply proposal')", 180)
        time.sleep(1)
        r.do("T.click('Apply proposal')", 6)
    r.do("T.tab('Spanwise')", 2)
    case = Case.load(os.path.join(paths.CASES_DIR, CASE))
    case.flow.U = 40.0
    tip = sizing.first_height_for_yplus(case, 30.0, section_params(p.to_stations(), 6.0)[0])
    r.do("T.set('cell at the tip (m)', {})".format(tip), 1)
    r.shot("Step 4 · Spanwise", "Cell size on the wing, at the tip (its height sets the y+ of the tip face; here "
           "for y+ 30) and beyond the tip: cells and grading of every layer, continuous at the slices.",
           W_("cell at the tip (m)"), also=[W_("cell on the wing (m)"), W_("largest cell beyond the tip (m)")],
           click=PROPOSE("Spanwise"), where="right")
    r.do(PROPOSE("Spanwise") + ".click()", 2)
    r.wait_for("!!T.button('Apply proposal')", 180)
    time.sleep(1)
    r.do("T.click('Apply proposal')", 6)
    r.do("T.tab('Cell budget')", 2)
    r.shot("Step 4 · Cell budget", "Scales all cell counts to a target total, optionally keeping the first wall "
           "cell and the tip cell (y+).", W_("Target number of cells"),
           also=[W_("keep the first wall cell and the tip cell (y+)")], where="right")
    r.chapter("Step 4: Meshing - quality optimiser")
    r.do("T.tab('Optimise quality')", 2)
    r.do("T.set('Evaluations (about 0.4 s each)', 100)", 1)
    r.shot("Step 4 · Optimise quality", "Improves non-orthogonality, skewness and size jumps at a <b>fixed cell "
           "count</b> (y+ unchanged). Choose what it may change, the priority and the number of evaluations. "
           "Nothing gets worse than now.", L_("wake gradings (e01, e02)"),
           also=[W_("Priority: size jumps (0) ... orthogonality (1)"), W_("Evaluations (about 0.4 s each)")],
           click=PROPOSE("Optimise quality"), where="right")
    r.do(PROPOSE("Optimise quality") + ".click()", 15)
    r.shot("Step 4 · Optimiser running", "The progress: evaluations and the objective (smaller is better).",
           HAS("evaluation"), where="right")
    r.wait_for("!!T.button('Apply proposal')", 600)
    time.sleep(2)
    r.shot("Step 4 · Optimiser result", "The changes and the mesh quality now and with the proposal "
           "(non-orthogonality as checkMesh, skewness, size ratios).", HAS("Mesh quality"), where="right")
    r.do("T.click('Apply proposal')", 6)

    r.chapter("Step 4: Meshing - settings and preview")
    r.shot("Step 4 · Edges", "Cells and gradings of e00-e10 (grad1: airfoil side, grad2: outer side), editable "
           "like a spreadsheet.", TYPED("Tabulator", 0), where="right")
    r.shot("Step 4 · Spanwise layers", "Cells and grading per layer, first and last cell, and the size jump to "
           "the next layer (warning above 1.3); the estimated number of cells below.", TYPED("Tabulator", 1),
           also=[HAS("Estimated cells")], where="right")
    r.shot("Step 4 · Mesh preview", "The 2D mesh of a slice, computed with blockMesh's own interpolation: it is "
           "the real mesh. Choose the slice; zoom with the presets or the mouse wheel; hover a cell for its "
           "values.", HAS("Slice 0 at y"), also=[W_("Slice"), L_("Leading edge")], where="left")
    r.do("T.set('Colour by', 'aspect ratio, 3D (with spanwise size)')", 3)
    r.do("T.radio('Leading edge')", 3)
    r.shot("Step 4 · Colour by", "Cell size, aspect ratio (in the slice or 3D with the spanwise size), size ratio "
           "to the neighbours or the smallest corner angle. Here: 3D aspect ratio at the leading edge.",
           W_("Colour by"), also=[HAS("Slice 0 at y")], where="left")
    r.do("T.set('Colour by', 'cell size (m)')", 2)
    r.do("T.radio('Airfoil')", 3)
    r.shot("Step 4 · Update", "The preview updates after every change (option), or with <b>Update preview</b>.",
           W_("update the preview after every change"), also=[BTN("Update preview")], where="left")
    r.shot("Step 4 · Measured sizes", "Summary of the slice and the measured cell sizes per edge parameter, "
           "e.g. the first cell at the wall.", HAS("cells in the slice"), also=[HAS("first cell at the wall (height)")],
           where="left")
    r.shot("Step 4 · Along the airfoil", "Chordwise cell length on both sides with the largest jump between "
           "neighbours, and the first wall-normal cell height along the surface.",
           HAS("Cells along the airfoil surface"), also=[HAS("First wall-normal cell height")], where="left")
    r.shot("Step 4 · Distributions", "Wall-normal and wake cell sizes, and the spanwise cell size with the slices.",
           HAS("Wall-normal and wake cell sizes"), also=[HAS("Spanwise cell size")], where="left")

    # ---------------- step 5 ---------------- #
    r.chapter("Step 5: Build & check")
    r.do("T.radio('5. Build & check')", 5)
    b.inject()
    r.do("T.check('also make images (mshGen plots + ParaView views)', true)", 1)
    r.do("T.card('Environment modules', true)", 2)
    r.shot("Step 5 · Options", "Images (mshGen plots and 13 ParaView views), export to Fluent, and the modules "
           "of Python, OpenFOAM and ParaView on this cluster.",
           W_("also make images (mshGen plots + ParaView views)"),
           also=[W_("export to Fluent (foamMeshToFluent)"), CARD("Environment modules")], where="right")
    r.do("T.card('Environment modules', false)", 1)
    r.shot("Step 5 · Build", "<b>Save and build mesh</b>: saves the case, runs the block check and Allrun.pre "
           "(mshGen, blockMesh, checkMesh, foamMeshToFluent, images).",
           BTN("Save and build mesh (Allrun.pre)"), click=BTN("Save and build mesh (Allrun.pre)"), where="right")
    r.do("T.click('Save and build mesh (Allrun.pre)')", 45)
    r.shot("Step 5 · Live log", "The log of the build while it runs.", HAS("$ ./Allrun.pre"), where="right")
    r.wait_for("!!T.find('== exit code', 'has')", 1500, 5)
    time.sleep(12)
    b.inject()
    r.shot("Step 5 · Finished", "The build is done.", HAS("== exit code"), where="right")
    r.shot("Step 5 · checkMesh", "The checkMesh summary: cells, cell types, aspect ratio, non-orthogonality, "
           "skewness, failed checks.", HAS("max_non_orthogonality"), where="right")
    r.wait_for("!!T.find('y+ on the walls', 'has')", 120)
    r.shot("Step 5 · y+ on the walls", "y+ measured on the built mesh per wall patch (wall distance of the first "
           "cells): about 30, as set with the helpers.", HAS("y+ on the walls"), where="right")
    r.shot("Step 5 · Patches and Fluent zones", "Patch types and faces, and the zones of the Fluent mesh: "
           "<b>inlet</b> is written as pressure-outlet, set it to velocity-inlet in Fluent.",
           HAS("pressure-outlet"), where="right")
    r.do("T.click('Show 3D view of the walls')", 3)
    r.wait_for("!!T.typed('Plotly')", 120)
    time.sleep(3)
    from core.results import wall_surfaces
    from ui.build import BuildStep
    fig3 = BuildStep.figure_3d(wall_surfaces(os.path.join(paths.CASES_DIR, CASE)), True)
    r.fig3d = fig3
    r.shot("Step 5 · 3D view of the walls", "The wall patches of the built mesh with cell edges: drag to rotate.",
           TYPED("Plotly"), where="left")
    r.wait_for("!!T.find('Images (last build with images)', 'has')", 120)
    r.shot("Step 5 · Images", "The images of the build: mshGen's block plots and ParaView views (root plane, "
           "leading and trailing edge, first cells, 3D wing, planform, tip, cuts).",
           HAS("Images (last build with images)"), where="left")

    r.chapter("Files and next steps")
    r.card("The case folder", [
        "cases/tutorial_wing/",
        "  case.json          all settings (source of truth)",
        "  airfoil.txt, wing.txt, settings/*.csv, generate_mesh.py, images.json",
        "  Allrun.pre         rebuilds everything without the GUI:  ./Allrun.pre [--images]",
        "  constant/polyMesh  OpenFOAM mesh;  fluentInterface/tutorial_wing.msh  Fluent mesh",
        "",
        "Command line: python GUI/mshgen_case.py list | new | import | write | blocks | build | check",
        "Documentation: GUI/README.md, GUI/PLAN.md, LIMITATIONS.md"], seconds=14)


def assemble(r, out, fps=10):
    '''Frames + hold times -> H.264 MP4 with chapters (ffmpeg from imageio-ffmpeg).'''
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
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
    r = Recorder(frames)
    try:
        time.sleep(15)
        r.b = Browser(W, H, work)
        scenario(r, "http://127.0.0.1:{}/app".format(port), login)
        total, chapters = assemble(r, a.out)
        with open(os.path.splitext(a.out)[0] + "_chapters.txt", "w") as f:
            f.write("\n".join(chapters) + "\n")
        print("written {} ({:.0f} s, {} frames, {:.1f} MB)".format(a.out, total, len(r.frames),
                                                                     os.path.getsize(a.out) / 1e6))
        print("\n".join(chapters))
    finally:
        if r.b:
            r.b.close()
        srv.kill()
        shutil.rmtree(case_dir, ignore_errors=True)
        if not a.frames:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
