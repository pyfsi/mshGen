'''
vertexpanel.py: block vertices and block check, shown in steps 1-3

- where the vertices come from (mshGen defaults for this geometry / outdated / custom), recompute button
- mshGen default availability for airfoil, R/c and level
- exterior vertices off their boundary (e.g. custom vertices after changing R/c)
- block check: inverted/folded blocks (errors), skewed blocks (warnings)
'''
import panel as pn

from core import blockcheck, blocks, geometry
from ui.common import note


class VertexPanel:

    def __init__(self, app):
        self.app = app
        self._vertex_key = None
        self.recompute_btn = pn.widgets.Button(name="Recompute block vertices (mshGen defaults)", button_type="warning")
        self.reset_edges = pn.widgets.Checkbox(name="also reset edge resolution/grading to mshGen defaults")
        self.recompute_btn.on_click(self.on_recompute)
        self.layout = pn.Column(sizing_mode="stretch_width")

    def load(self):
        self._vertex_key = None

    def view(self):
        self.refresh()
        return self.layout

    def on_recompute(self, event):
        c, f, reset = self.app.case, self.app.airfoil_file, self.reset_edges.value
        self.recompute_btn.disabled = True
        self.recompute_btn.name = "Recomputing (mshGen, a few seconds) ..."

        def done(result, error):
            self.recompute_btn.disabled = False
            self.recompute_btn.name = "Recompute block vertices (mshGen defaults)"
            if error:
                self.app.notify("error", "Recompute failed: " + str(error))
            else:
                self.app.notify("success", "Block vertices recomputed" + (" and edges reset" if result else ""))
            self.app.changed(refresh_step=True)
        self.app.run_background(lambda: geometry.recompute_vertices(c, f, reset_edges=reset), done)

    def refresh(self):
        c, f = self.app.case, self.app.airfoil_file
        if c is None:
            return
        state = geometry.vertex_state(c, f)
        key = (state, c.airfoil.name, c.domain.mesh_scale, c.domain.mesh_level)
        if key != self._vertex_key:     # default of the checkbox; the user's choice is kept otherwise
            self.reset_edges.value = geometry.edge_defaults_changed(c)
            self._vertex_key = key
        items = []
        if state == "ok":
            items.append(note("Block vertices are mshGen's defaults for this geometry.", "success"))
        elif state == "outdated":
            items += [note("The geometry, R/c or level changed since the block vertices were computed. They are "
                           "recomputed (mshGen defaults) when you save the case, or now:", "warning"),
                      self.recompute_btn, self.reset_edges]
        else:
            items += [note("The block vertices were set by hand (custom) and are not recomputed automatically. "
                           "To use mshGen's defaults for this geometry:", "info"),
                      self.recompute_btn, self.reset_edges]
        try:
            geometry.check_level(c)
        except ValueError as e:
            items.append(note(str(e), "danger"))
        if state != "outdated":
            off = blocks.exterior_problems(c)
            if off:
                items.append(note("Exterior vertices {} are not on the outer boundary for R = {:.4g} m (R/c changed?). "
                                  "Recompute the vertices or move them in step 3.".format(
                                      ", ".join(map(str, off)), blocks.radius(c)), "danger"))
        items.append(self.block_check_view(state))
        self.layout.objects = items

    def block_check_view(self, state):
        if state == "outdated":
            self.app.block_problems = None
            return pn.pane.Markdown("_Block check: runs after the vertices are recomputed._")
        try:
            lay = self.app.layout()
        except Exception as e:  # noqa: BLE001
            self.app.block_problems = None
            return note("Block check failed: " + repr(e), "danger")
        c = self.app.case
        problems, warnings = blockcheck.problems_of(c, lay), blockcheck.warnings_of(c, lay)
        self.app.block_problems = problems

        def lst(items, n=6):
            return "\n".join("- " + p for p in items[:n]) + ("" if len(items) <= n else "\n- ... ({} more)".format(len(items) - n))
        out = []
        if problems:
            out.append(note("**Block check: invalid blocks** (blockMesh would fail). Reduce sweep/twist/taper, move "
                            "slices or vertices; trailing edges must stay upstream of the near-wake line.\n\n"
                            + lst(problems), "danger"))
        if warnings:
            out.append(note("**Skewed blocks** (corner below {:.0f} or above {:.0f} deg: poor cells, high "
                            "non-orthogonality):\n\n".format(blockcheck.SKEW_MIN_ANGLE, blockcheck.SKEW_MAX_ANGLE)
                            + lst(warnings), "warning"))
        if not problems and not warnings:
            out.append(note("Block check: all blocks valid in all slices.", "success"))
        return pn.Column(*out, sizing_mode="stretch_width")
