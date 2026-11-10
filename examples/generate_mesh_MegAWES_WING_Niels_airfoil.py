'''
generate_section_3Dmesh.py
author: Thomas Haas
date: 22.09.2023
description: Generate files for blockMesh utility
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
airfoil_name = "Mrev-v2"

# Airfoil file (Add airfoil file to './airfoils/' directory)
filename1 = path_cwd + "/airfoils/Mrev-v2.txt"

# Wing name (User-specific)
wing_name = "WING"

# Wing file (Add wing file to './wings/' directory)
filename2 = None

# Case dimension (Choose from "2D" or "3D"):
dim = "2D"

# Mesh origin (COG):
COG = [0, 0, 0]

La = 0.4 #distance between aileron and wing (spanwise)

# Mesh parameters:
mesh_scale = 5.         # Extents of C-grid (Choose from: 5)
mesh_level = "COARSE"   # Mesh refinement level (Choose from: "COARSE")
slice_locations = [0., 1.]
layer_resolution = [1]
layer_grading = [1.]
layer_aileron = None
skip_layer = 1 #not skiping if ARTIFICIOUS = False

# Output folder for default files:
foldername1 = path_mshGen + "/default_files"

# Output folder for BlockMesh files:
foldername2 = path_cwd + "/output_files"

# Flags
WRITE=True
PRINT=True
PLOT= False
DEFAULT= False# True: Write default files; False: Read default files

# optional flags to splt surfaces, used to match surface for FSI
SPLIT = False
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
