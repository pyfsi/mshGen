'''
app.py: mshGen GUI (Panel, in the browser). Start it with GUI/run_gui.sh.

The app holds the case being edited (in memory); "Save" writes case.json and all case files
(recomputing the block vertices first if the geometry changed). Steps:
  1. Geometry                         ui/step1_geometry.py
  2. Domain, 3. Blocks, 4. Meshing    ui/step2_domain.py, ui/step3_blocks.py, ui/step4_meshing.py
  5. Build & check                    ui/build.py
See GUI/PLAN.md.
'''
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import panel as pn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import airfoils, blockcheck, estimate, geometry, paths  # noqa: E402
from core.case import Case  # noqa: E402
from core.writer import write_case  # noqa: E402
from ui.build import BuildStep  # noqa: E402
from ui.step4_meshing import MeshingStep  # noqa: E402
from ui.step1_geometry import GeometryStep  # noqa: E402
from ui.step2_domain import DomainStep  # noqa: E402
from ui.step3_blocks import BlocksStep  # noqa: E402
from ui.vertexpanel import VertexPanel  # noqa: E402

pn.extension("tabulator", "plotly", notifications=True, sizing_mode="stretch_width")


def _json(case):
    return json.dumps(case.to_dict(), sort_keys=True)


class App:

    def __init__(self):
        self.case, self.folder, self.saved_json, self.airfoil_file = None, None, None, None
        self.dim_memory = {}
        self.block_problems = None          # result of the last block check (None: not checked)
        self.jobs = []
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Sidebar: case
        self.case_select = pn.widgets.Select(name="Case", options=paths.list_cases())
        self.save_btn = pn.widgets.Button(name="Save", button_type="primary")
        self.revert_btn = pn.widgets.Button(name="Revert", button_type="light")
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Sidebar: new case
        self.new_mode = pn.widgets.RadioButtonGroup(options=["New wing", "From example"], value="New wing")
        self.new_name = pn.widgets.TextInput(name="Name", placeholder="e.g. my_wing")
        self.new_airfoil = pn.widgets.Select(name="Airfoil", options=airfoils.names_with_defaults())
        self.example_select = pn.widgets.Select(name="Example script", options=paths.list_examples(), visible=False)
        self.create_btn = pn.widgets.Button(name="Create case", button_type="success")

        self._layout_cache = (None, None)
        self._model_cache = (None, None)
        self.vertex_panel = VertexPanel(self)
        self.steps = [GeometryStep(self), DomainStep(self), BlocksStep(self), MeshingStep(self), BuildStep(self)]
        self.step = pn.widgets.RadioBoxGroup(name="Step", options=[s.title for s in self.steps],
                                             value=self.steps[0].title)
        self.main = pn.Column()

        self.case_select.param.watch(lambda e: self.load_case(e.new), "value")
        self.save_btn.on_click(lambda e: self.save())
        self.revert_btn.on_click(lambda e: self.load_case(self.case.name) if self.case else None)
        self.step.param.watch(lambda e: self.show_step(), "value")
        self.new_mode.param.watch(self.on_new_mode, "value")
        self.create_btn.on_click(self.on_create)
        pn.state.add_periodic_callback(self.poll, period=500)

        self.template = pn.template.FastListTemplate(
            title="mshGen GUI",
            sidebar=[self.case_select, pn.Row(self.save_btn, self.revert_btn), self.status, pn.layout.Divider(),
                     pn.pane.Markdown("**Steps**"), self.step, pn.layout.Divider(),
                     pn.pane.Markdown("**New case**"), self.new_mode, self.new_name, self.new_airfoil,
                     self.example_select, self.create_btn],
            main=[self.main], accent_base_color="#2c6e9b", header_background="#2c6e9b", sidebar_width=330)
        # Optional URL parameters: /app?case=<name>&step=<1-5>
        args = pn.state.session_args or {}
        want_case = args.get("case", [b""])[0].decode()
        want_step = args.get("step", [b""])[0].decode()
        if want_step.isdigit() and 1 <= int(want_step) <= len(self.steps):
            self.step.value = self.steps[int(want_step) - 1].title
        if want_case in self.case_select.options:
            self.case_select.value = want_case          # triggers load_case
            if self.case is None:
                self.load_case(want_case)
        elif self.case_select.options:
            self.load_case(self.case_select.value)
        else:
            self.main.objects = [pn.pane.Markdown("No cases yet: create one (left).")]

    # ---------------- helpers for the steps ---------------- #
    def notify(self, kind, msg):
        n = pn.state.notifications
        if n is None:
            print("[" + kind + "] " + msg)
        else:
            getattr(n, kind)(msg, duration=8000 if kind in ("error", "warning") else 4000)

    def run_background(self, fn, done):
        '''Run fn() in a worker thread; done(result, error) is called in this session afterwards.'''
        self.jobs.append((self.executor.submit(fn), done))

    def poll(self):
        for job in [j for j in self.jobs if j[0].done()]:
            self.jobs.remove(job)
            fut, done = job
            err = fut.exception()
            done(None if err else fut.result(), err)
        self.steps[4].poll()
        self.steps[3].helpers.poll()

    def set_airfoil_file(self, filename):
        self.airfoil_file = filename

    def changed(self, refresh_step=False):
        '''Called after every edit: status; with refresh_step also the current step (e.g. after a recompute).'''
        self.update_status()
        if refresh_step:
            step = self.current_step()
            if hasattr(step, "refresh"):
                step.refresh()
            else:
                self.show_step()

    def current_step(self):
        return next(s for s in self.steps if s.title == self.step.value)

    def layout(self):
        '''Block layout of the current case (vertices of all slices, block status); cached per case state.'''
        key = _json(self.case) + "|" + str(self.airfoil_file)
        if self._layout_cache[0] != key:
            self._layout_cache = (key, blockcheck.layout(self.case, self.airfoil_file,
                                                         blockcheck.slice_vertices(self.case, self.airfoil_file,
                                                                                   self.model())))
        return self._layout_cache[1]

    def model(self):
        '''mshGen Mesh object (vertices, blocks, edges) of the current case; cached per case state.'''
        from core.model import mshgen_mesh
        key = _json(self.case) + "|" + str(self.airfoil_file)
        if self._model_cache[0] != key:
            self._model_cache = (key, mshgen_mesh(self.case, self.airfoil_file, upto="edges"))
        return self._model_cache[1]

    def is_dirty(self):
        return self.case is not None and _json(self.case) != self.saved_json

    def update_status(self):
        c = self.case
        if c is None:
            self.status.object = ""
            return
        problems = c.validate()
        state = geometry.vertex_state(c, self.airfoil_file)
        lines = ["**{}**{}".format(c.name, "  – _unsaved changes_" if self.is_dirty() else ""),
                 "", "Cells (estimate): **{:,}**".format(estimate.cell_count(c)),
                 "", "Block vertices: " + {"ok": "mshGen defaults", "outdated": "**recomputed on save**",
                                           "custom": "custom"}[state],
                 "", "Block check: " + ("not run" if self.block_problems is None else
                                        "OK" if not self.block_problems else
                                        "**{} invalid blocks**".format(len(self.block_problems))),
                 "", "Folder: `cases/" + c.name + "`"]
        lines += (["", "Problems:"] + ["- " + p for p in problems]) if problems else ["", "Settings valid"]
        self.status.object = "\n".join(lines)

    def refresh_view(self, step):
        if self.step.value == step.title:
            self.show_step()
        self.update_status()

    # ---------------- case handling ---------------- #
    def load_case(self, name):
        if not name:
            return
        if self.is_dirty() and self.case.name != name:
            self.notify("warning", "Unsaved changes of " + self.case.name + " were discarded")
        folder = os.path.join(paths.CASES_DIR, name)
        try:
            case = Case.load(folder)
        except Exception as e:  # noqa: BLE001
            self.main.objects = [pn.pane.Alert("Cannot read case.json: " + str(e), alert_type="danger")]
            return
        self.case, self.folder, self.saved_json = case, folder, _json(case)
        self.airfoil_file = os.path.join(folder, case.airfoil.file)
        self.dim_memory = {}
        self.vertex_panel.load()
        for s in self.steps:
            s.load()
        self.update_status()
        self.show_step()

    def save(self, then=None):
        '''Save in the background: recompute outdated vertices, copy the airfoil, write case.json and files.'''
        case, folder, airfoil_file = self.case, self.folder, self.airfoil_file
        self.save_btn.disabled = True

        def job():
            msgs = []
            if geometry.vertex_state(case, airfoil_file) == "outdated":
                reset = geometry.recompute_vertices(case, airfoil_file)
                msgs.append("block vertices recomputed" + (" and edges reset to mshGen defaults" if reset else ""))
            target = os.path.join(folder, case.airfoil.file)
            if os.path.abspath(airfoil_file) != os.path.abspath(target):
                shutil.copyfile(airfoil_file, target)
            problems = case.validate()
            block_problems = None
            if problems:
                case.save(folder)
            else:
                write_case(folder, case)
                block_problems = blockcheck.check(case, target)
            return msgs, problems, block_problems

        def done(result, error):
            self.save_btn.disabled = False
            if error:
                self.notify("error", "Save failed: " + str(error))
                if then:
                    then(False)
                return
            msgs, problems, block_problems = result
            self.saved_json = _json(self.case)
            self.airfoil_file = os.path.join(folder, case.airfoil.file)
            if problems:
                self.notify("warning", "case.json saved, case files not written: " + "; ".join(problems))
            elif block_problems:
                self.notify("warning", "Saved, but {} blocks are invalid (blockMesh would fail): {}".format(
                    len(block_problems), block_problems[0]))
            else:
                self.notify("success", "Saved cases/" + case.name + (" (" + ", ".join(msgs) + ")" if msgs else ""))
            self.changed(refresh_step=True)
            if then:
                then(not problems and not block_problems)
        self.run_background(job, done)

    # ---------------- new case ---------------- #
    def on_new_mode(self, event):
        self.new_airfoil.visible = event.new == "New wing"
        self.example_select.visible = event.new == "From example"

    def on_create(self, event):
        name = self.new_name.value.strip()
        if not name:
            self.notify("warning", "Enter a name for the new case")
            return
        if os.path.exists(os.path.join(paths.CASES_DIR, name)):
            self.notify("warning", "cases/" + name + " already exists")
            return
        self.create_btn.disabled = True
        if self.new_mode.value == "New wing":
            airfoil = self.new_airfoil.value
            fn = lambda: geometry.new_wing_case(name, airfoil)  # noqa: E731
        else:
            from core.importer import case_from_example
            example = os.path.join(paths.EXAMPLES_DIR, self.example_select.value)
            fn = lambda: case_from_example(example, name)  # noqa: E731

        def done(result, error):
            self.create_btn.disabled = False
            if error:
                self.notify("error", "Could not create the case: " + str(error))
                return
            self.notify("success", "Created cases/" + name)
            self.new_name.value = ""
            self.case_select.options = paths.list_cases()
            self.case_select.value = name
        self.run_background(fn, done)

    # ---------------- steps ---------------- #
    def show_step(self):
        if self.case is None:
            return
        step = self.current_step()
        try:
            content = step.view()
        except Exception as e:  # noqa: BLE001
            content = [pn.pane.Alert("Cannot show this step: " + repr(e), alert_type="danger")]
        self.main.objects = [pn.pane.Markdown("## " + step.title)] + content


App().template.servable()
