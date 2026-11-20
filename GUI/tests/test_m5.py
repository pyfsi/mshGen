'''
test_m5.py: M5 test, the full chain driven through the GUI (the Panel app, without a browser)

Creates cases/zz_test_m5 (deleted afterwards) and, step by step as a user would:
  1. new wing (Mrev-v2), planform parameters (half-span, chords, sweep, twist, dihedral)
  2. flow conditions (U, nu)
  3. wing surface split (wingUp/wingDown), an extra slice
  4. helpers: wall y+ 30 (growth <= 1.15), wake, chordwise, spanwise (tip cell for y+ 30)
  5. save and build with Fluent export and images; then: cells = estimate, y+ measured on the mesh,
     Fluent zones, images, 3D view of the walls
Needs OpenFOAM and ParaView (modules in case.json). Run: python GUI/tests/test_m5.py
'''
import glob
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GUI = os.path.dirname(HERE)
sys.path.insert(0, GUI)
os.environ.setdefault("MPLBACKEND", "Agg")

NAME = "zz_test_m5"


def load_app():
    src = open(os.path.join(GUI, "app.py")).read().replace("App().template.servable()", "")
    g = {"__file__": os.path.join(GUI, "app.py"), "__name__": "test_m5"}
    exec(compile(src, os.path.join(GUI, "app.py"), "exec"), g)
    return g


def main():
    g = load_app()
    from core import estimate, paths, sizing
    from core.planform import section_params
    folder = os.path.join(paths.CASES_DIR, NAME)
    if os.path.exists(folder):
        shutil.rmtree(folder)
    a = g["App"]()

    def wait(cond=lambda: not a.jobs, timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            a.poll()
            if cond():
                a.poll()
                return
            time.sleep(0.2)
        raise TimeoutError("waited too long")

    try:
        # 1. new wing + planform
        a.new_mode.value = "New wing"
        a.new_name.value = NAME
        a.new_airfoil.value = "Mrev-v2"
        a.on_create(None)
        wait()
        assert a.case.name == NAME, a.case.name
        geo = a.steps[0]
        a.step.value = geo.title
        for k, v in (("half_span", 6.0), ("root_chord", 1.2), ("tip_chord", 0.6), ("kink_y", 1.0),
                     ("le_sweep_deg", 5.0), ("twist_root_deg", 2.0), ("twist_tip_deg", -1.0), ("dihedral_deg", 3.0)):
            geo.params[k].value = v
        print("1 planform:", a.case.wing.stations, "| slices", [round(s, 3) for s in a.case.blocks.slice_locations])
        # 2. flow
        mesh = a.steps[3]
        a.step.value = mesh.title
        mesh.U.value, mesh.nu.value = 40.0, 1.5e-5
        print("2 flow: U", a.case.flow.U, "nu", a.case.flow.nu)
        # 3. blocks: split, extra slice; save (recomputes the vertices for the new root chord)
        blk = a.steps[2]
        a.step.value = blk.title
        blk.split.value = True
        blk.add_y.value = 2.0
        blk.on_add(None)
        a.save()
        wait()
        assert not a.is_dirty() and a.block_problems == [], a.block_problems
        print("3 blocks: slices", [round(s, 3) for s in a.case.blocks.slice_locations], "| split", a.case.options.split,
              "| vertices", a.case.blocks.vertex_source)
        # 4. helpers
        mesh = a.steps[3]
        a.step.value = mesh.title
        H = mesh.helpers
        H.yplus.value, H.wall_mode.value = 30.0, "limit growth ratio"
        H.propose_wall()
        wait()
        H.apply()
        H.propose_wake()
        wait()
        H.apply()
        H.propose_chordwise()
        wait()
        H.apply()
        c = a.case
        H.s_tip.value = sizing.first_height_for_yplus(c, 30.0, section_params(c.wing_stations, c.half_span)[0])
        H.s_far.value = 1.0
        H.propose_spanwise()
        wait()
        H.apply()
        pred = sizing.predict(c, a.airfoil_file)
        print("4 helpers: predicted", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in pred.items()})
        # 5. build with Fluent and images
        b = a.steps[4]
        a.step.value = b.title
        b.images_chk.value = True
        b.fluent_chk.value = True
        b.on_build(None)
        wait(lambda: not a.jobs and b.thread is None and b.log_lines and b.log_lines[-1].startswith("=="),
             timeout=1800)
        wait()
        log = "\n".join(b.log_lines)
        assert "== exit code 0" in log, log[-2000:]
        from core.runner import checkmesh_summary
        from core.results import fluent_zones, measured_yplus
        s = checkmesh_summary(folder)
        assert s["cells"] == estimate.cell_count(a.case) == pred["cells"], (s["cells"], pred["cells"])
        y = measured_yplus(folder, a.case)
        zones = {z["name"]: z["type"] for z in fluent_zones(os.path.join(folder, "fluentInterface", NAME + ".msh"))}
        imgs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(folder, "images", "*.png")))
        print("5 build: {:,} cells, max non-orth {:.1f}, max skew {:.2f}, {}".format(
            s["cells"], s["max_non_orthogonality"], s["max_skewness"],
            "mesh OK" if s["mesh_ok"] else "failed: " + "; ".join(s["warnings"])))
        for k, v in y.items():
            print("  y+ {:9s} median {:.1f} (min {:.1f}, max {:.1f})".format(k, v["y+ median"], v["y+ min"], v["y+ max"]))
            assert 24 <= v["y+ median"] <= 36, (k, v)
        print("  Fluent zones:", zones)
        assert zones.get("wingUp") == "wall" and zones.get("wingDown") == "wall" and zones.get("symRoot") == "symmetry"
        print("  images:", len(imgs), imgs[:3], "...")
        assert len([i for i in imgs if not i.startswith("mshgen")]) == 13
        b.show_3d()
        wait()
        # no browser: the WebGL probe never answers -> server-rendered image (after the probe timeout)
        v = a.view3d.views["build"]
        assert a.view3d.effective() == "image", a.view3d.effective()
        png = v.image.object
        assert isinstance(png, bytes) and png[:8] == b"\x89PNG\r\n\x1a\n", type(png)
        v.preset.value = "Top (x-y)"
        wait()
        assert v.image.object != png and (v.az.value, v.el.value) == (0, 90)
        print("  3D view as image: {} kB, top view re-rendered".format(len(png) // 1000))
        a.view3d.mode.value = "interactive"
        assert type(v.panel.objects[0]).__name__ == "Plotly", v.panel.objects
        traces = v.panel.objects[0].object.data
        print("  3D view interactive:", len(traces), "traces:", sorted({t.name for t in traces}))
        print("OK    full chain through the GUI")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
