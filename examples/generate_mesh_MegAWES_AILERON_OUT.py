'''
generate_mesh_MegAWES_AILERON_OUT.py
author: Thomas Haas (ported from mshGen_v0)
date: 22.09.2023
description: Generate files for blockMesh utility (MegAWES outer aileron, Mrev-v2_aileron, 3D)
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

# Airfoil name
airfoil_name = "Mrev-v2_aileron"

# Airfoil file (Add airfoil file to mshGen/meshGeneration/airfoils/)
filename1 = path_mshGen + "/airfoils/Mrev-v2_aileron.txt"

# Wing name (User-specific)
wing_name = "AILERON_OUT"

# Wing file (Add wing file to mshGen/meshGeneration/wings/)
filename2 = path_mshGen + "/wings/MegAWES_aileron_out.txt"

# Case dimension (Choose from "2D" or "3D"):
dim = "3D"

# Mesh origin (COG):
COG = [0.179, 0, 0]

# Mesh parameters:
mesh_scale = 1.5        # Extents of C-grid
mesh_level = "COARSE"   # Mesh refinement level
slice_locations = [0, 3.535, 3.885]
layer_resolution = [60, 10]
layer_grading = [1, 1]
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
DEFAULT = False  # Read the tuned default files (default_*_Mrev-v2_aileron_3D_AILERON_OUT.csv)

# Optional flags to split surfaces, used to match surface for FSI
SPLIT = False
ARTIFICIOUS = False

# Create inputs of Mesh instance (No user action needed)
# Note: case name without "_R...C_<level>" suffix, matching the existing default files of this aileron
casename = airfoil_name+"_"+dim+"_"+wing_name
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
