'''
generate_mesh_MegAWES_WING_Niels_split_yplus50.py
author: Thomas Haas / Niels Pynaert
        (based on generate_mesh_MegAWES_WING.py (slices) and generate_mesh_MegAWES_WING_Niels_nohole.py (wing, settings))
date: 16.11.2026
description: Generate files for blockMesh utility
             Clean MegAWES wing (no aileron cut-out), wing surface split into suction (wingUp)
             and pressure (wingDown) side, first cell for y+ ~ 50 on wingUp, wingDown and wingTip.

y+ target:
  - Flow conditions as in the MegAWES OpenFOAM cases: U = 60 m/s, nu = 1e-5 m2/s (Re_c = 2.7e7 at root)
  - Flat-plate estimate Cf = (2 log10(Re_c) - 0.65)^-2.3 -> u_tau ~ 2.0-2.1 m/s along the span
  - y+ = 50 at the first cell centre -> first cell height ~ 5e-4 m
  - Wall-normal (default_resolution_grading_Mrev-v2_3D_WING_YP50_R00500C_COARSE.csv, e00):
    64 cells, expansion ratio 5800 (growth ~ 1.15), same as the tuned "WING" settings
  - Chordwise: as "WING", except e04 (trailing-edge segment -> former aileron hinge): 11 cells, grading 2
    instead of 16 cells refined towards the hinge, so the cell size varies smoothly (neighbour ratio <= 1.14)
  - Spanwise: slices of the clean-wing example, the layers on both sides of the tip graded
    to a first cell of 5e-4 m at the tip (growth ~ 1.15). Spanwise cell size:
    0.465 (root) -> 0.30 (y=18.9) -> 5e-4 (tip) -> 0.30 (y=23.5) -> 1.03 m (far field)
  - Local y+ on the wing varies with the pressure distribution (higher near the suction peak)
'''

# Imports
import os
import sys
from matplotlib import pyplot as plt

# mshGen location, derived from this script (<...>/mshGen/examples/<script>.py)
path_mshGen = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meshGeneration"))
sys.path.insert(0, os.path.dirname(os.path.dirname(path_mshGen)))
from mshGen.meshGeneration.mesh_class import Mesh

# ---------------------- Initialization ---------------------- #

# Initialization
path_cwd = os.getcwd()
plt.close("all")

# ---------------------- User settings ---------------------- #
'''
User settings
'''

# Airfoil name (Choose from "NACA0012", "Mrev-v2", "Mrev-v2_aileron")
airfoil_name = "Mrev-v2"

# Airfoil file (Add airfoil file to mshGen/meshGeneration/airfoils/)
filename1 = path_mshGen + "/airfoils/Mrev-v2.txt"

# Wing name (User-specific)
wing_name = "WING_YP50"  # Own settings files (y+ ~ 50), keeps the tuned "WING" files untouched

# Wing file (Add wing file to mshGen/meshGeneration/wings/)
filename2 = path_mshGen + "/wings/MegAWES_wing.txt"

# Case dimension (Choose from "2D" or "3D"):
dim = "3D"

# Mesh origin (COG):
COG = [1.6729, 0, 0.2294]

# Mesh parameters:
mesh_scale = 5.         # Extents of C-grid (Choose from: 5)
mesh_level = "COARSE"   # Mesh refinement level (Choose from: "COARSE")
slice_locations = [0., 4.65, 18.935, 21.235, 23.535, 42.47]  # root | taper | tip-2.3 m | tip | tip+2.3 m | far field
layer_resolution = [10, 38, 47, 47, 32]
layer_grading = [1., 0.642, 1/598., 598., 3.43]                 # tip layers: 0.30 m <-> 5e-4 m at the tip
layer_aileron = None    # Clean wing: no aileron cut-out
skip_layer = 1 #not skiping if ARTIFICIOUS = False

# Output folder for default files:
foldername1 = path_mshGen + "/default_files"

# Output folder for BlockMesh files:
foldername2 = path_cwd + "/output_files"
os.makedirs(foldername2, exist_ok=True)

# Flags
WRITE=True
PRINT=True
PLOT= False
DEFAULT= False # True: Write default files; False: Read default files

# optional flags to splt surfaces, used to match surface for FSI
SPLIT = True            # Wing surface as wingUp (suction side) / wingDown (pressure side)
ARTIFICIOUS = False

# Create inputs of Mesh instance (No user action needed)
casename = airfoil_name+"_"+dim+"_"+wing_name+"_R"+"{:05.0f}".format(1e2*mesh_scale)+"C_"+mesh_level
names = [casename, airfoil_name, wing_name]
files = [filename1, filename2, foldername1, foldername2]
params = [dim, COG, slice_locations, layer_resolution, layer_grading, layer_aileron, mesh_scale, mesh_level,skip_layer]
flags = [WRITE, PRINT, PLOT, DEFAULT,SPLIT,ARTIFICIOUS]

# ---------------------- Create OpenFOAM mesh ---------------------- #
'''
Create OpenFOAM mesh
'''

# Create mesh
mesh = Mesh(names, files, params, flags)

# Build mesh
mesh.initialize_mesh()                                              # Initialize mesh
mesh.generate_default()                                             # Generate mesh.default
mesh.generate_vertices()                                            # Generate mesh.vertices
mesh.generate_blocks()                                              # Generate mesh.blocks
mesh.generate_edges()                                               # Generate mesh.edges
mesh.generate_interfaces()                                          # Generate mesh.interfaces
mesh.generate_boundaries()                                          # Generate mesh.boundaries
if PLOT:
    plt.show()

print("END")
# ---------------------- END ---------------------- #
