'''
generate_section_3Dmesh.py
author: Thomas Haas
date: 22.09.2023
description: Generate files for blockMesh utility
'''

# Imports
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.interpolate import splprep, splev
# from meshGeneration.generate_default_airfoil_files import AirfoilSection, interpolate_wing_shape, interpolate_lifting_surface_shape
# import meshGeneration.mesh_functions as mf
from meshGeneration.mesh_class import Mesh
# from meshGeneration.generate_default_files import
# from meshGeneration.generate_default_files import main as generate_default
# from meshGeneration.mesh_functions import vertex, block, edge
# from naca0012_mesh.mesh_functions import vertex, block, edge
import meshGeneration.aero_functions as af
import meshGeneration.mesh_functions as mf
from importlib import reload
reload(af)
reload(mf)

# ---------------------- Initialization ---------------------- #

# Initialization
path = os.getcwd()
plt.close("all")

# ---------------------- User settings ---------------------- #
'''
User settings
'''

# Case names:
old_airfoil_name = "Mrev-v2"
new_airfoil_name = "Mrev_aileron"

# Airfoil files:
# filename1 = os.getcwd() + "/NACA0012.txt"
filename1 = os.getcwd() + "/Mrev-v2.txt"
filename2 = os.getcwd() + "/Mrev-v2_aileron.txt"

# Get airfoil data of original airfoil
s = af.AirfoilSection(old_airfoil_name, filename1, None, [0, 0, 0], y=0)
s.raw_data.plot()
fig_num = plt.get_fignums()
fig = plt.figure(fig_num[-1])
ax = plt.figure(fig_num[-1]).axes[0]

# Find location of aileron hinge on "camber" spline
N1 = 1001
x, z = splev(np.linspace(0, 1, N1), s.raw_data.tck_camber)
m = np.argmin(np.abs(x - 0.75))
p0 = [x[m], z[m]]
ax.plot(p0[0], p0[1], '*', color="blue", markersize=15., label='Hinge')
print("p0: x_hinge = " + "{:.3f}".format(p0[0]) + ", z_hinge = " + "{:.3f}".format(p0[1]))

# Compute radius
N2 = 101
x, z = splev(np.linspace(0, 1, N1), s.raw_data.tck_upper)
m1 = np.argmin(np.abs(x - 0.75))
p1 = [x[m1], z[m1]]
R1 = np.linalg.norm(np.array(p1)-np.array(p0))
ax.plot(p1[0], p1[1], 's', color="blue", markersize=5.)
x, z = splev(np.linspace(0, 1, N1), s.raw_data.tck_lower)
m2 = np.argmin(np.abs(x - 0.75))
p2 = [x[m2], z[m2]]
ax.plot(p2[0], p2[1], 's', color="blue", markersize=5.)
R2 = np.linalg.norm(np.array(p2)-np.array(p0))
R = np.max([R1,R2])
x_circle = p0[0]+R*np.cos(np.linspace(0,2*np.pi,N2))
z_circle = p0[1]+R*np.sin(np.linspace(0,2*np.pi,N2))
ax.plot(x_circle, z_circle, '-', color="gray")

# Extract new suction side
x, z = splev(np.linspace((m1+1)/(N1-1), 1, N1), s.raw_data.tck_upper)
x_upper = np.concatenate((x_circle[int((N2-1)/2):int(1*(N2-1)/4)-1:-1], x))
z_upper = np.concatenate((z_circle[int((N2-1)/2):int(1*(N2-1)/4)-1:-1], z))
ax.plot(x_upper, z_upper, 'b--')

# Extract new pressure side
x, z = splev(np.linspace((m2+1)/(N1-1), 1, N1), s.raw_data.tck_lower)
x_lower = np.concatenate((x_circle[int((N2-1)/2):int(3*(N2-1)/4)+1], x))
z_lower = np.concatenate((z_circle[int((N2-1)/2):int(3*(N2-1)/4)+1], z))
ax.plot(x_lower, z_lower, 'b--')

# Gather all points
x = np.concatenate((x_upper, x_lower))
z = np.concatenate((z_upper, z_lower))
ax.plot(x, z, 'go', markersize=5.)

#===========================#

# Compute new leading edge
d = [np.linalg.norm([x[i]-1, z[i]-0]) for i in range(len(x))]
m = np.argmax(d)
ax.plot(x[m], z[m], '*', color='orange', markersize=15.)
ax.plot([x[m], 1], [z[m], 0], '--', color='orange')
p_rot = [x[m], z[m]]

# Rotate
xr = []
zr = []
angle = np.arctan(z[m]/(1-x[m]))
for k in range(len(x)):
    vec = mf.Ry(-angle) @ [x[k]-x[m], 0, z[k]-z[m]]
    xr.append(vec[0])
    zr.append(vec[2])

# Scale
c0 = np.max(d)
xs = np.array([x/c0 for x in xr])
zs = np.array([z/c0 for z in zr])

fig, ax = plt.subplots()
ax.plot(xs, zs, "-", color='gray')
ax.set_aspect("equal")

idx_upper = np.where(zs > +1e-6)[0]
x0 = np.concatenate(([0], xs[idx_upper], [1]))
z0 = np.concatenate(([0], zs[idx_upper], [0]))

idx_lower = np.where(zs < -1e-6)[0]
x1 = np.concatenate(([0], xs[idx_lower], [1]))
z1 = np.concatenate(([0], zs[idx_lower], [0]))

x_interp = np.linspace(0, 1, 21)
x_interp = np.insert(x_interp, 1, 0.01)
z0_interp = np.interp(x_interp, x0, z0)
z1_interp = np.interp(x_interp, x1, z1)

ax.plot(x_interp, z0_interp, "or")
ax.plot(x_interp, z1_interp, "ob")

# # Write to file
# x = np.concatenate((x_interp, x_interp))
# z = np.concatenate((z0_interp, z1_interp))
# with open(filename2, "w") as file:
#     for v1, v2 in zip(x, z):
#         values = ['{:>12.8f}'.format(val) for val in [v1,v2]]
#         print(' '.join(values), file=file)

# Hinge location in
x0 = p0
xr = mf.Ry(-angle) @ [x0[0]-p_rot[0], 0, x0[1]-p_rot[1]]
xs = xr/c0
ax.plot(xs[0], xs[-1], '*g')

print("END")
#
# # Extract new pressure side
# x = x_circle[int((N2-1)/2):int(3*(N2-1)/4)+1]
# z = z_circle[int((N2-1)/2):int(3*(N2-1)/4)+1]
# ax.plot(x, z, 'g-')
# x, z = splev(np.linspace((m2+1)/(N1-1), 1, N1), s.raw_data.tck_lower)
# ax.plot(x, z, 'g-')
