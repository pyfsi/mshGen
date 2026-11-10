'''
generate_mesh_NACA0012_RUDDER_UP.py
author: Thomas Haas (ported from mshGen_v1)
date: 22.09.2023
description: Generate files for blockMesh utility (MegAWES rudder, NACA0012, 3D, R = 1c, flow domain up to y = 2.25)
'''

# Imports
from mshGen.meshGeneration.mesh_class import Mesh
from matplotlib import pyplot as plt
import os

# ---------------------- Initialization ---------------------- #

# Initialization
path_cwd = os.getcwd()
plt.close("all")

# ---------------------- User settings ---------------------- #
'''
User settings
'''

# Path mshGen/meshGeneration
path_mshGen = "/cfdfile1/data/fm/niels/awesim/software/mshGen/meshGeneration"

# Airfoil name (Choose from "NACA0012", "Mrev-v2", "Mrev-v2_aileron")
airfoil_name = "NACA0012"

# Airfoil file (Add airfoil file to './airfoils/' directory)
filename1 = path_cwd + "/airfoils/NACA0012.txt"

# Wing name (User-specific)
wing_name = "RUDDER"

# Wing file (Add wing file to './wings/' directory)
filename2 = path_cwd + "/wings/MegAWES_rudder.txt"

# Case dimension (Choose from "2D" or "3D"):
dim = "3D"

# Mesh origin (COG):
COG = [0.7, 0., 0.]

# Mesh parameters:
mesh_scale = 1          # Extents of C-grid
mesh_level = "COARSE"   # Mesh refinement level (Choose from "COARSE", "MEDIUM", "FINE")
slice_locations = [0., 1.5, 2.25]
layer_resolution = [10, 10]
layer_grading = [1, 1]
layer_aileron = None    # None, a layer index, or a list of consecutive layer indices
skip_layer = 1          # Not skipping if ARTIFICIOUS = False

# Output folder for default files:
foldername1 = path_mshGen + "/default_files"

# Output folder for BlockMesh files:
foldername2 = path_cwd + "/output_files"

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
