'''
generate_mesh_MegAWES_WING.py
author: Thomas Haas
updated: 23.08.2024
description: Generate files for blockMesh utility
'''

# Imports
from mshGen.meshGeneration.mesh_class import Mesh
from matplotlib import pyplot as plt
import os

# ---------------------- Initialization ---------------------- #

# Initialization
path = os.getcwd()
plt.close("all")

# ---------------------- User settings ---------------------- #
'''
User settings
'''

# Path (current working directory)
path = "/mshGen/meshGeneration"

# Airfoil name (Choose from "NACA0012", "Mrev-v2", "Mrev-v2_aileron")
airfoil_name = "Mrev-v2"

# Airfoil file (Add airfoil file to './airfoils/' directory)
filename1 = path + "/airfoils/Mrev-v2.txt"

# Wing name (User-specific)
wing_name = "WING"

# Wing file (Add wing file to './wings/' directory)
filename2 = path + "/wings/MegAWES_basic_wing.txt"

# Case dimension (Choose from "2D" or "3D"):
dim = "3D"

# Mesh origin (COG):
COG = [1.6729, 0, 0.2294]

# Mesh parameters:
mesh_scale = 5.         # Extents of C-grid (Choose from: 5)
mesh_level = "COARSE"   # Mesh refinement level (Choose from: "COARSE", "MEDIUM", "FINE")
slice_locations = [0., 4.65, 18.235, 21.235, 24.235, 42.470]
layer_resolution = [10, 36, 10, 10, 32]
layer_grading = [1., 0.667, 1., 1., 3.2]
layer_aileron = None

# Output folder for default files:
foldername1 = path + "/default_files"

# Output folder for BlockMesh files:
foldername2 = path + "/output_files"

# Flags
WRITE=True
PRINT=True
PLOT=False
DEFAULT=True # True: Write default files; False: Read default files

# Create inputs of Mesh instance (No user action needed)
casename = airfoil_name+"_"+dim+"_"+wing_name+"_R"+"{:05.0f}".format(1e2*mesh_scale)+"C_"+mesh_level
names = [casename, airfoil_name, wing_name]
files = [filename1, filename2, foldername1, foldername2]
params = [dim, COG, slice_locations, layer_resolution, layer_grading, layer_aileron, mesh_scale, mesh_level]
flags = [WRITE, PRINT, PLOT, DEFAULT]

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
