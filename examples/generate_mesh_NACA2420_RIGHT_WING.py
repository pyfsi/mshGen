'''
generate_mesh_NACA2420_RIGHT_WING.py
author: Thomas Haas (example by Daniele, merged from mshGen_Daniele)
updated: 22.04.2025
description: Generate files for blockMesh utility (NACA2420 right wing, 3D, with flow domain)
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

# Airfoil name (Choose from "NACA0012", "NACA2420", "Mrev-v2", "Mrev-v2_aileron", "AP2")
airfoil_name = "NACA2420"

# Airfoil file (Add airfoil file to mshGen/meshGeneration/airfoils/)
filename1 = path_mshGen + "/airfoils/NACA2420.txt"

# Wing name (User-specific)
wing_name = "right_wing"

# Wing file (Add wing file to mshGen/meshGeneration/wings/)
filename2 = path_mshGen + "/wings/right_wing.txt"

# Case dimension (Choose from "2D" or "3D"):
dim = "3D"

# Mesh origin (COG): quarter chord of the 0.1 m chord
COG = [0.1*0.25, 0, 0]

# Mesh parameters:
mesh_scale = 5          # Extents of C-grid
mesh_level = "COARSE"   # Mesh refinement level (Choose from: "COARSE", "MEDIUM", "FINE")
slice_locations = [0.0, 1.0, 1.1, 1.2]
layer_resolution = [30, 16, 16]
layer_grading = [1/1.5, 1/2.4, 2.4]
layer_aileron = None    # None, a layer index, or a list of consecutive layer indices
skip_layer = 1          # Not skipping if ARTIFICIOUS = False

# Output folder for default files:
foldername1 = path_mshGen + "/default_files"

# Output folder for BlockMesh files:
foldername2 = path_cwd + "/output_files"
os.makedirs(foldername2, exist_ok=True)

# Flags
WRITE = True
PRINT = True
PLOT = False
DEFAULT = True  # True: Write default files; False: Read default files

# Optional flags to split surfaces, used to match surface for FSI
SPLIT = False
ARTIFICIOUS = False

# Create inputs of Mesh instance (No user action needed)
casename = airfoil_name+"_"+dim+"_"+wing_name+"_R"+"{:05.0f}".format(1e2*mesh_scale)+"C_"+mesh_level
names = [casename, airfoil_name, wing_name]
files = [filename1, filename2, foldername1, foldername2]
params = [dim, COG, slice_locations, layer_resolution, layer_grading, layer_aileron, mesh_scale, mesh_level, skip_layer]
flags = [WRITE, PRINT, PLOT, DEFAULT, SPLIT, ARTIFICIOUS]

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
