'''
helpers.py: sizing helpers of step 4 (M4)

Each tab computes a proposal from target sizes (core/sizing.py). The proposal is shown as old -> new values
and its predicted result (measured on the actual block mesh of a copy of the case, next to the current
values) before it is applied: wall y+, chordwise sizes, wake, spanwise sizes, cell budget.
'''
import numpy as np
import panel as pn

from core import estimate, preview, sizing
from ui.common import note, table


class HelpersPanel:

    def __init__(self, step):
        self.step = step
        self.app = step.app
        self.proposal = None
        self._defaults_for = None
        f = pn.widgets.FloatInput
        # wall
        self.yplus = f(name="Target y+ (first cell centre)", value=50.0, step=1.0, start=0.1, width=200)
        self.criterion = pn.widgets.Select(name="for the", options=["median", "max"], value="median", width=120)
        self.wall_mode = pn.widgets.RadioButtonGroup(options=["keep wall-normal cells", "limit growth ratio"],
                                                     value="keep wall-normal cells", button_type="light")
        self.wall_growth = f(name="max growth ratio", value=1.15, step=0.01, start=1.01, width=150)
        self.wall_info = pn.pane.Markdown()
        # chordwise
        self.d_te = f(name="cell at the TE (m)", step=0.001, start=1e-6, width=150)
        self.d_le = f(name="cell at the LE (m)", step=0.001, start=1e-6, width=150)
        self.d_max = f(name="largest cell (m)", step=0.01, start=1e-6, width=150)
        self.c_growth = f(name="max growth ratio", value=1.15, step=0.01, start=1.01, width=150)
        # wake
        self.w_first = f(name="first wake cell at the TE (m)", step=0.001, start=1e-6, width=200)
        self.w_max = f(name="largest wake cell (m)", step=0.1, start=1e-6, width=150)
        self.w_growth = f(name="max growth ratio", value=1.15, step=0.01, start=1.01, width=150)
        # spanwise
        self.s_root = f(name="cell on the wing (m)", step=0.01, start=1e-6, width=150)
        self.s_tip = f(name="cell at the tip (m)", step=1e-4, start=1e-7, width=150)
        self.s_far = f(name="largest cell beyond the tip (m)", step=0.1, start=1e-6, width=200)
        self.s_growth = f(name="max growth ratio", value=1.15, step=0.01, start=1.01, width=150)
        self.s_info = pn.pane.Markdown()
        # budget
        self.target = pn.widgets.IntInput(name="Target number of cells", step=100000, start=1000, width=200)
        self.keep_yplus = pn.widgets.Checkbox(name="keep the first wall cell and the tip cell (y+)", value=True)
        # quality optimiser
        from core.quality import GROUPS
        self.opt_groups = pn.widgets.CheckBoxGroup(name="Variables", options=GROUPS, inline=False,
                                                   value=["grad1", "grad2", "wake", "span"])
        self.opt_weight = pn.widgets.FloatSlider(name="Priority: size jumps (0) ... orthogonality (1)", start=0.0,
                                                 end=1.0, step=0.1, value=0.5)
        self.opt_evals = pn.widgets.IntSlider(name="Evaluations (about 0.4 s each)", start=50, end=600, step=50,
                                              value=200)
        self.opt_progress = pn.pane.Markdown()
        self._progress = None

        self.buttons = {}
        tabs = []
        for name, widgets, fn in [
            ("Wall y+", [pn.Row(self.yplus, self.criterion), self.wall_mode, self.wall_growth, self.wall_info],
             self.propose_wall),
            ("Chordwise", [pn.pane.Markdown("_Cell sizes along the airfoil (e03-e07, grad1); sizes grow by at most "
                                            "the ratio from the TE and LE up to the largest cell._"),
                           pn.Row(self.d_te, self.d_le), pn.Row(self.d_max, self.c_growth)], self.propose_chordwise),
            ("Wake", [pn.pane.Markdown("_Cells along the wake line: near wake (e02, 0.5 root chord) and far wake "
                                       "(e01, to the outlet), growing from the TE cell._"),
                      self.w_first, pn.Row(self.w_max, self.w_growth)], self.propose_wake),
            ("Spanwise", [pn.pane.Markdown("_Cells and grading of every layer: refined to the tip cell at the tip "
                                           "slice (its height sets the y+ of the tip face), sizes continuous at "
                                           "the slices._"),
                          pn.Row(self.s_root, self.s_tip), pn.Row(self.s_far, self.s_growth), self.s_info],
             self.propose_spanwise),
            ("Cell budget", [pn.pane.Markdown("_Scales all cell counts (chordwise, wall-normal, wake, spanwise) "
                                              "to the target._"), self.target, self.keep_yplus], self.propose_budget),
            ("Optimise quality", [pn.pane.Markdown(
                "_Improves non-orthogonality, skewness and size jumps at a **fixed cell count** (counts, wall and "
                "tip first cells unchanged), measured on the exact block mesh of the root, tip and far-field "
                "slices. No quality value gets worse than now. Vertices moved by the optimiser become custom._"),
                self.opt_groups, self.opt_weight, self.opt_evals, self.opt_progress], self.propose_optimise),
        ]:
            btn = pn.widgets.Button(name="Propose", button_type="primary", width=120)
            btn.on_click(lambda e, fn=fn: fn())
            self.buttons[name] = btn
            tabs.append((name, pn.Column(*widgets, btn, sizing_mode="stretch_width")))
        self.tabs = pn.Tabs(*tabs, dynamic=False)
        self.apply_btn = pn.widgets.Button(name="Apply proposal", button_type="success", width=160)
        self.discard_btn = pn.widgets.Button(name="Discard", button_type="light", width=100)
        self.apply_btn.on_click(lambda e: self.apply())
        self.discard_btn.on_click(lambda e: self.show(None))
        self.result = pn.Column(sizing_mode="stretch_width")
        self.layout = pn.Card(self.tabs, self.result, title="Helpers: cells and gradings from target sizes",
                              collapsed=False, sizing_mode="stretch_width")

    # ---------------- defaults from the current mesh ---------------- #
    def load(self):
        self.proposal = None
        self._defaults_for = None
        self.result.objects = []

    def refresh(self):
        '''Fill the inputs from the current mesh (once per case), show current y+.'''
        c = self.app.case
        self.tabs[3].visible = c.dim == "3D"
        try:
            model = self.app.model()
            w = sizing.wall_yplus(c, self.app.airfoil_file, model)
            st = sizing.yplus_stats(w)
        except Exception as e:  # noqa: BLE001
            self.wall_info.object = "Cannot measure y+: " + repr(e)
            return
        self.wall_info.object = ("_Now (U = {:g} m/s, nu = {:g} m2/s, flat-plate u_tau with the local chord): "
                                 .format(c.flow.U, c.flow.nu)
                                 + ", ".join("{} {:.3g}".format(k, v) for k, v in st.items()) + "._")
        if self._defaults_for == c.name:
            return
        self._defaults_for = c.name
        blocks = preview.slice_mesh(c, self.app.airfoil_file, 0, model)
        surf = preview.surface_cells(blocks)
        if surf:
            up = surf["upper"]["length"]
            self.d_te.value = float(round(up[-1], 5))
            self.d_le.value = float(round(min(up[0], surf["lower"]["length"][0]), 5))
            self.d_max.value = float(round(max(up.max(), surf["lower"]["length"].max()), 4))
            self.w_first.value = float(round(up[-1], 5))
        if len(st):
            self.yplus.value = float(round(st["y+ median"], 1))
        stats = {r["edge"] + r["where"]: r for r in preview.edge_stats(blocks)}
        far = [r for k, r in stats.items() if k.startswith("e01")]
        self.w_max.value = float(round(far[0]["largest cell (m)"], 3)) if far else 1.0
        if c.dim == "3D":
            span = estimate.spanwise_table(c)
            self.s_root.value = float(round(span[0]["first cell (m)"], 4))
            tip = w["tip"][0] if w["tip"] else float(np.median(w["all"])) if len(w["all"]) else 1e-3
            self.s_tip.value = float(tip)
            self.s_far.value = float(round(max(r["last cell (m)"] for r in span), 3))
        self.target.value = int(estimate.cell_count(c))

    # ---------------- proposals ---------------- #
    def _run(self, name, fn):
        c, af = self.app.case, self.app.airfoil_file
        btn = self.buttons[name]
        btn.disabled, btn.name = True, "Computing ..."

        def job():
            prop = fn()
            return prop, sizing.predict(c, af), sizing.predict(c, af, prop)

        def done(result, error):
            btn.disabled, btn.name = False, "Propose"
            if error:
                self.app.notify("warning", "No proposal: " + str(error))
                return
            self.show(*result, title=name)
        self.app.run_background(job, done)

    def propose_wall(self):
        c, af = self.app.case, self.app.airfoil_file
        growth = self.wall_growth.value if self.wall_mode.value == "limit growth ratio" else None
        self._run("Wall y+", lambda: sizing.wall_proposal(c, af, self.app.model(), float(self.yplus.value),
                                                          self.criterion.value, growth_max=growth))

    def propose_chordwise(self):
        c = self.app.case
        self._run("Chordwise", lambda: sizing.chordwise_proposal(c, self.app.model(), float(self.d_te.value),
                                                                 float(self.d_le.value), float(self.d_max.value),
                                                                 float(self.c_growth.value)))

    def propose_wake(self):
        c = self.app.case
        self._run("Wake", lambda: sizing.wake_proposal(c, self.app.model(), float(self.w_first.value),
                                                       float(self.w_growth.value), float(self.w_max.value)))

    def propose_spanwise(self):
        c = self.app.case
        self._run("Spanwise", lambda: sizing.spanwise_proposal(c, float(self.s_root.value), float(self.s_tip.value),
                                                               float(self.s_far.value), float(self.s_growth.value)))

    def propose_budget(self):
        c, af = self.app.case, self.app.airfoil_file
        self._run("Cell budget", lambda: sizing.budget_proposal(c, af, self.app.model(), int(self.target.value),
                                                                keep_yplus=self.keep_yplus.value))

    def propose_optimise(self):
        from core.quality import optimise
        c, af = self.app.case, self.app.airfoil_file
        groups, w, n = list(self.opt_groups.value), float(self.opt_weight.value), int(self.opt_evals.value)
        if not groups:
            self.app.notify("warning", "Select at least one group of variables")
            return
        self._progress = {"text": "starting ..."}

        def progress(k, total, j, j0):
            self._progress = {"text": "evaluation {} of at most {}: objective {:.4f} (start {:.4f})".format(k, total, j, j0)}
        self._run("Optimise quality", lambda: optimise(c, af, groups, weight_orth=w, max_evals=n, progress=progress))

    def poll(self):
        '''Called periodically by the app: progress of the optimiser.'''
        if self._progress is not None:
            self.opt_progress.object = "_" + self._progress["text"] + "_"
            if not self.buttons["Optimise quality"].disabled:
                self._progress = None

    # ---------------- show / apply ---------------- #
    def show(self, prop, now=None, after=None, title=""):
        self.proposal = prop
        if prop is None:
            self.result.objects = []
            return
        rows = sizing.changes(self.app.case, prop)
        keys = list(dict.fromkeys(list(now) + list(after)))
        pred = [{"result": k, "now": now.get(k, ""), "with the proposal": after.get(k, "")} for k in keys]
        extra = []
        if "before" in prop:
            qk = [k for k in prop["before"] if k in prop["after"]]
            extra = [pn.pane.Markdown("**Mesh quality** (root, tip and far-field slices; non-orthogonality as "
                                      "checkMesh in the slice plane)"),
                     table([{"quality": k, "now": prop["before"][k], "with the proposal": prop["after"][k]}
                            for k in qk])]
        self.result.objects = [
            pn.pane.Markdown("**Proposal ({})**: {}".format(title, prop.get("note", ""))),
            table(rows) if rows else pn.pane.Markdown("_No changes._")] + extra + [
            pn.pane.Markdown("**Predicted** (measured on the block mesh of the proposal)"), table(pred),
            pn.Row(self.apply_btn, self.discard_btn)]

    def apply(self):
        if not self.proposal:
            return
        sizing.apply_proposal(self.app.case, self.proposal)
        self.app.notify("success", "Proposal applied (not saved yet)")
        self.show(None)
        self.step.after_helper()
