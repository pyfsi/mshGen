'''
generate_mesh_NACA0012_RUDDER_DOWN.py
author: Thomas Haas (ported from mshGen_v1)
date: 22.09.2023
description: Generate files for blockMesh utility (MegAWES rudder, NACA0012, 3D, R = 1c, flow domain up to y = 2.0)
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
airfoil_name = "NACA0012"

# Airfoil file (Add airfoil file to mshGen/meshGeneration/airfoils/)
filename1 = path_mshGen + "/airfoils/NACA0012.txt"

# Wing name (User-specific)
wing_name = "RUDDER"

# Wing file (Add wing file to mshGen/meshGeneration/wings/)
filename2 = path_mshGen + "/wings/MegAWES_rudder.txt"

# Case dimension (Choose from "2D" or "3D"):
dim = "3D"

# Mesh origin (COG):
COG = [0.7, 0., 0.]

# Mesh parameters:
mesh_scale = 1          # Extents of C-grid
mesh_level = "COARSE"   # Mesh refinement level (Choose from "COARSE", "MEDIUM", "FINE")
slice_locations = [0., 1.5, 2.]
layer_resolution = [10, 10]
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
DEFAULT = True  # True: Write default files; False: Read default files
                # (mshGen_v1 read default files from an earlier run with DEFAULT=False; those were not kept)

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
