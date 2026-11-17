'''
visualize_mesh.py: ParaView images of any mshGen case (offscreen, pvbatch), driven by images.json

Usage (from the case folder, after blockMesh; Allrun.pre --images does this):
    module purge; module load ParaView/5.9.1-foss-2021b-mpi
    pvbatch --force-offscreen-rendering <mshGen>/GUI/tools/visualize_mesh.py images.json
images.json is written by the GUI (GUI/core/views.py): patch names, root section, R, tip, far field.
Images in ./images:
    01 root plane, 02 root airfoil, 03 leading edge, 04 trailing edge, 05 first cells at the leading edge
    3D cases also: 06/07 wing (suction/pressure side), 08 planform, 09 tip, 10-13 cut at constant x
    (spanwise cells near the tip, tip zoom, first cells at the tip, whole domain)
'''
import json
import os
import sys

from paraview.simple import (CreateView, GetSources, Hide, OpenFOAMReader, Render, SaveScreenshot, Show,
                             Slice)

spec = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "images.json"))
os.makedirs("images", exist_ok=True)
if not os.path.exists(spec["foam"]):
    open(spec["foam"], "w").close()

UP, DOWN, TIP, OTHER = [0.27, 0.51, 0.71], [0.90, 0.55, 0.20], [0.60, 0.60, 0.60], [0.80, 0.30, 0.30]
EDGE = [0.15, 0.15, 0.15]

view = CreateView("RenderView")
view.ViewSize = [1600, 1000]
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 1
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass


def reader(regions):
    r = OpenFOAMReader(FileName=spec["foam"])
    r.MeshRegions = regions
    r.CellArrays = []
    return r


def show(src, color, edges=True, line_width=1.0):
    d = Show(src, view)
    d.ColorArrayName = ["POINTS", ""]
    d.Representation = "Surface With Edges" if edges else "Surface"
    d.AmbientColor = color
    d.DiffuseColor = color
    d.EdgeColor = EDGE
    d.LineWidth = line_width
    d.Ambient = 0.3
    d.Diffuse = 0.8
    return d


def bounds(src):
    src.UpdatePipeline()
    return src.GetDataInformation().GetBounds()


def camera_2d(normal, up, center, height):
    '''Orthographic camera looking along -normal, showing a window of the given height.'''
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = center
    view.CameraPosition = [c + 100 * n for c, n in zip(center, normal)]
    view.CameraViewUp = up
    view.CameraParallelScale = height / 2


def camera_3d(focal, offset, up):
    view.CameraParallelProjection = 0
    view.CameraFocalPoint = focal
    view.CameraPosition = [f + o for f, o in zip(focal, offset)]
    view.CameraViewUp = up
    view.CameraViewAngle = 30


def save(name):
    Render(view)
    SaveScreenshot("images/" + name + ".png", view, ImageResolution=view.ViewSize)
    print("written images/" + name + ".png")


def hide_all():
    for s in GetSources().values():
        Hide(s, view)


walls = spec["walls"]
groups = [(walls["up"], UP), (walls["down"], DOWN), (walls["tip"], TIP), (walls["other"], OTHER)]
root, R, dz0 = spec["root"], spec["R"], spec["dz0"]
le, te, c0, h = root["le"], root["te"], root["chord"], root["first_cell"]
y0 = 0.0 if spec["dim"] == "3D" else 0.0

# ---------------------- Root plane ---------------------- #
plane = reader([spec["root_plane"]])
show(plane, [0.97, 0.97, 0.97])
camera_2d([0, -1, 0], [0, 0, 1], [0.5 * R, y0, dz0], 2.1 * R)
save("01_root_plane_overview")
camera_2d([0, -1, 0], [0, 0, 1], [0.5 * (le[0] + te[0]) + 0.1 * c0, y0, 0.5 * (le[1] + te[1])], 0.8 * c0)
save("02_root_airfoil")
camera_2d([0, -1, 0], [0, 0, 1], [le[0] + 0.02 * c0, y0, le[1]], 0.08 * c0)
save("03_root_leading_edge")
camera_2d([0, -1, 0], [0, 0, 1], [te[0], y0, te[1]], 0.03 * c0)
save("04_root_trailing_edge")
camera_2d([0, -1, 0], [0, 0, 1], [le[0] + 6 * h, y0, le[1]], 24 * h)
save("05_root_leading_edge_first_cells")

if spec["dim"] == "3D":
    sp = spec["span"]
    b2, far, xcut = sp["half_span"], sp["far"], sp["xcut"]
    xa, xb = sp["planform_x"]
    # ---------------------- Wing surface ---------------------- #
    hide_all()
    readers = []
    for names, col in groups:
        if names:
            r = reader(names)
            show(r, col)
            readers.append((r, col))
    focal = [0.5 * (xa + xb), 0.5 * b2, dz0]
    camera_3d(focal, [-1.06 * b2, -1.18 * b2, 0.96 * b2], [0.3, 0.3, 0.9])
    save("06_wing_3d_suction_side")
    camera_3d(focal, [-1.06 * b2, -1.18 * b2, -0.96 * b2], [-0.3, -0.3, 0.9])
    save("07_wing_3d_pressure_side")
    camera_2d([0, 0, 1], [-1, 0, 0], [0.5 * (xa + xb), 0.5 * b2, 0], 1.08 * b2)
    view.CameraViewUp = [-1, 0, 0]
    save("08_wing_planform_top")
    if walls["tip"]:
        tb = bounds(reader(walls["tip"]))
        tc = [(tb[0] + tb[1]) / 2, (tb[2] + tb[3]) / 2, (tb[4] + tb[5]) / 2]
        ct = sp["tip_chord"]
        camera_3d(tc, [-1.4 * ct, 1.65 * ct, 1.2 * ct], [0, 0, 1])
        save("09_wing_tip_3d")
    # ---------------------- Cut at constant x ---------------------- #
    hide_all()
    cut = Slice(Input=reader(["internalMesh"]))
    cut.SliceType = "Plane"
    cut.SliceType.Origin = [xcut, 0, 0]
    cut.SliceType.Normal = [1, 0, 0]
    cut.Triangulatetheslice = 0
    show(cut, [0.97, 0.97, 0.97])
    for r, col in readers:
        pc = Slice(Input=r)
        pc.SliceType = "Plane"
        pc.SliceType.Origin = [xcut, 0, 0]
        pc.SliceType.Normal = [1, 0, 0]
        d = Show(pc, view)
        d.ColorArrayName = ["POINTS", ""]
        d.AmbientColor = col
        d.DiffuseColor = col
        d.LineWidth = 4
    zt = sp["tip_z_mid"]
    tag = "x{:.3g}".format(xcut)
    camera_2d([1, 0, 0], [0, 0, 1], [xcut, b2, zt], 0.33 * b2)
    save("10_cut_{}_tip_region".format(tag))
    camera_2d([1, 0, 0], [0, 0, 1], [xcut, b2, zt], 0.28 * sp["tip_chord"])
    save("11_cut_{}_tip_zoom".format(tag))
    camera_2d([1, 0, 0], [0, 0, 1], [xcut, b2, sp["tip_z_upper"]], 40 * sp["tip_first_cell"])
    save("12_cut_{}_tip_first_cells".format(tag))
    camera_2d([1, 0, 0], [0, 0, 1], [xcut, 0.5 * far, dz0], 2.15 * R)
    save("13_cut_{}_whole_domain".format(tag))
