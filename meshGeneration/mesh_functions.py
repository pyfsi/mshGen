'''
airfoil_functions.py
author: Thomas Haas
date: 26.07.2023
description: processing of MegAWES airfoil data
'''

# Imports
import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
from matplotlib import pyplot as plt

# -------------------- Class definitions -------------------- #

class vertex:
    def __init__(self, idx, y, pos, line, tck):
        if line in ['s','c','p']:
            x, z = splev(pos, tck)
            coord = [x,y,z]
        else:
            line='m'
            coord=[]
        self.pos = pos
        self.line = line
        self.coord = coord
        self.index = idx

class block:
    def __init__(self, index, vertices, layer, vertices_per_layer):
        offset = int(layer*vertices_per_layer)
        vertices = [i+offset for i in vertices] + [i+offset+vertices_per_layer for i in vertices]
        self.vertices = [vertices[i] for i in [0,1,5,4,3,2,6,7]]
        self.faces = define_block_faces(self.vertices)
        self.resolution = [1,1,1]
        self.grading = [1,1,1]
        self.index = index

class edge:
    def __init__(self, index, vertices, tck, n):
        if tck:
            v0, v1 = vertices
            x, z = splev(np.linspace(v0.pos, v1.pos, num=n), tck)
            y = v0.coord[1]
            coord = [x,y*np.ones(n),z]
            for i in range(3):
                coord[i][0] = v0.coord[i]
                coord[i][-1] = v1.coord[i]
            self.coord = coord
        else:
            self.coord = []
        self.vertices = vertices
        self.index = index
        self.type = 'polyLine'

# -------------------- Helper functions -------------------- #
def get_vertex_coord(vertex_list):
    vertex_coord = np.empty((0, 3))
    for v in vertex_list:
        vertex_coord = np.append(vertex_coord, [v.coord], axis=0)
    return vertex_coord

def get_tck(line, tck_list):
    tck_p, tck_c, tck_s = tck_list
    if line == 's':
        tck = tck_s
    elif line == 'c':
        tck = tck_c
    elif line == 'p':
        tck = tck_p
    return tck

def populate_vertices(slice_vertex_list, spline_vertices, line, tck):
    for v in [slice_vertex_list[i] for i in spline_vertices]:
        u_spline = np.linspace(0., 1., 1001)
        x_spline, z_spline = splev(u_spline, tck)
        x, y, z = v.coord
        index = np.abs(x_spline - x).argmin()
        v.pos = u_spline[index]
        v.line = line
    return slice_vertex_list

def define_block_faces(index_list):
    face_list = []
    face_list += [[index_list[i] for i in [0, 4, 7, 3]]] # 0: Left face
    face_list += [[index_list[i] for i in [0, 3, 2, 1]]] # 1: Bottom face
    face_list += [[index_list[i] for i in [1, 2, 6, 5]]] # 2: Right face
    face_list += [[index_list[i] for i in [4, 5, 6, 7]]] # 3: Top face
    face_list += [[index_list[i] for i in [0, 1, 5, 4]]] # 4: Front face (at origin of local frame)
    face_list += [[index_list[i] for i in [3, 7, 6, 2]]] # 5: Back face (in positive y-direction)
    return face_list

# Retrieve vertices of all blocks
def get_block_vertices(block_list):
    block_vertices = np.empty((0, 8)).astype(np.int32)
    for b in block_list:
        # print(b.vertices)
        block_vertices = np.append(block_vertices, [b.vertices], axis=0)
    return block_vertices

def get_edge_vertices(list_of_consecutive_vertices):
    n_edges = len(list_of_consecutive_vertices)-1
    edge_list = []
    for k in range(n_edges):
        edge_list.append([list_of_consecutive_vertices[k],list_of_consecutive_vertices[k+1]])
    return edge_list

# -------------------- Plot functions -------------------- #
def plot_airfoil(coord, camber):

    # Create figure
    fig = plt.figure(figsize=(16, 8))
    ax = []
    ax.append(fig.add_axes([0.1, 0.1, 0.39, 0.85]))
    ax.append(fig.add_axes([0.6, 0.1, 0.39, 0.85]))

    # Plot data
    for ax0 in ax:
        ax0.plot(coord[:,0], coord[:,1], 'kx', markersize=3, markevery=5)
        ax0.plot(camber[:,0], camber[:,1], 'kx', markersize=3, markevery=5)

    # Layout
    for ax0 in ax:
        ax0.set_xlabel('x [m]')
        ax0.set_ylabel('z [m]')
        ax0.set_aspect('equal')
        ax0.grid()

    # Layout: close-up view
    ax[0].set_xlim([-2.5, 3.5])
    ax[0].set_ylim([-3.0, 3.0])
    ax[0].set_xticks(np.linspace(-2.5, 3.5, 13))
    ax[0].set_yticks(np.linspace(-3.0, 3.0, 13))
    ax[0].set_title("Close-up view")

    # Layout: wide view
    ax[1].set_xlim([-25.0, 50.0])
    ax[1].set_ylim([-25.0, 25.0])
    ax[1].set_xticks(np.linspace(-25.0, 50.0, 16))
    ax[1].set_yticks(np.linspace(-25.0, 25.0, 11))
    ax[1].set_title("Wide view")

    return

def plot_spline(ax, tck, color):
    x, y = splev(np.linspace(0, 1, num=101), tck)
    for ax0 in ax:
        ax0.plot(x, y, '-', color=color)
    return

def plot_vertices(ax, vertex_list):

    # Retrieve coordinates of all vertices
    vertex_coord = get_vertex_coord(vertex_list)
    n = len(vertex_list)

    # Plot vertices
    N_cp = 14
    N_hp = N_cp + 8
    for ax0 in ax:
        for n, v in enumerate(vertex_list):
            if n < N_cp:
                color='tab:red'
            elif N_cp <= n < N_hp:
                color='tab:green'
            elif n >= N_hp:
                color='tab:purple'
            x, y, z = v.coord
            ax0.plot(vertex_coord[n, 0], vertex_coord[n, 2], 'o', color=color)
            ax0.annotate(str(v.index), xy=(x, z), xytext=(5, 10), color=color, textcoords='offset points', ha='left', va='top')

    return

def plot_blocks(ax, vertex_list, layer_block_list):
    # Plot blocks
    vertex_coord = get_vertex_coord(vertex_list)
    block_vertices = get_block_vertices(layer_block_list)
    for i, (vert_list, blck) in enumerate(zip(block_vertices,layer_block_list)):
        if i < 13:
            color1="lightblue"
            color2 = "tab:blue"
        if i >= 13:
            color1="lightgray"
            color2 = "k"
        vlist = [vert_list[k] for k in [0,1,5,4,0]]
        # print(i, vert_list, vlist)
        x = [vertex_coord[k,0] for k in vlist]
        y = [vertex_coord[k,2] for k in vlist]
        x0 = np.mean([vertex_list[k].coord[0] for k in vlist[:4]])
        y0 = np.mean([vertex_list[k].coord[2] for k in vlist[:4]])
        for ax0 in ax:
            ax0.fill(x,y, color=color1, edgecolor='k')
            ax0.annotate(str(blck.index), xy=(x0, y0), ha='center', va='center', color=color2)
    return


# -------------------- Write functions -------------------- #
def write_vertices_to_file(filename, vertex_list, slice, s=1.):
    with open(filename, "w") as file:
        print('// Vertices of slice ' + str(slice), file=file)
        for v in vertex_list:
            x, y, z = v.coord
            values = ['{:>24.12f}'.format(s*val) for val in [x,y,z]]
            print('(' + ' '.join(values) + ') // '+str(v.index), file=file)
    return

def write_blocks_to_file(filename, layer_block_list, layer, list_of_commented_blocks=None):
    with open(filename, "w") as file:
        print('// Blocks of layer ' + str(layer), file=file)
        for blck in layer_block_list:
            if blck.index in list_of_commented_blocks:
                prefix = '// '
            else:
                prefix = ''
            vert = ['{:>3d}'.format(val) for val in blck.vertices]
            resl = ['{:>3d}'.format(val) for val in blck.resolution]
            grad = ['{:>18.12f}'.format(val) for val in blck.grading]
            print(prefix+'hex (' + ' '.join(vert) + ') (' + ' '.join(resl) + ') edgeGrading (' + ' '.join(grad) + ') // '+str(blck.index), file=file)

def write_edges_to_file(filename, slice_edge_list, slice, list_of_commented_edges=None, s=1.):
    with open(filename, "w") as file:
        print('// Edges of slice ' + str(slice), file=file)
        for k, (e) in enumerate(slice_edge_list):
            if k in list_of_commented_edges:
                prefix = '/*\n'
                suffix = '\n*/'
            else:
                prefix = ''
                suffix = ''

            if e.type == "polyLine":
                print(prefix + e.type + ' '.join(['{:>4d}'.format(val.index) for val in e.vertices])+ '\n(', file=file)
                for i, (x, y, z) in enumerate(zip(e.coord[0], e.coord[1], e.coord[2])):
                    values = ['{:>18.12f}'.format(s*val) for val in [x, y, z]]
                    print('(' + ' '.join(values) + ') // '+str(i), file=file)
                print(')'+suffix, file=file)
            elif e.type == "arc":
                vertices = ' '.join(['{:>4d}'.format(val.index) for val in e.vertices])
                values = ['{:>18.12f}'.format(s*val) for val in e.coord]
                coord = ' (' + ' '.join(values) + ') // '+str(e.index)
                print(prefix + e.type + vertices + coord + suffix, file=file)

def write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index):
    with open(filename, "w") as file:
        print(name+'\n{\n'+'type '+type+';\nfaces\n(', file=file)
        for blck, face_index in zip(list_of_blocks, list_of_face_index):
            print('(' + ' '.join(['{:>3d}'.format(f) for f in blck.faces[face_index]]) + ')', file=file)
        print(');\n}', file=file)

def write_lists_to_file(A, B, header, filename):
    with open(filename, 'w') as file:
        file.write(','.join(header) + '\n')  # Writing the header
        for i in range(len(A)):
            file.write(str(A[i]) + ',' + str(B[i]) + '\n')  # Writing the elements of A and B

def read_more_lists_from_file(filename, STR_DATA=False):
    A = []
    B = []
    C = []
    with open(filename, 'r') as file:
        next(file)  # Skipping the header
        for line in file:
            line = line.rstrip('\n')
            values = line.split(',')
            values = [v.replace(" ", "") for v in values]
            A.append(float(values[0]))
            B.append(float(values[1]))
            C.append(float(values[2]))
    return A, B, C

def read_lists_from_file(filename, STR_DATA=False):
    A = []
    B = []
    with open(filename, 'r') as file:
        next(file)  # Skipping the header
        for line in file:
            line = line.rstrip('\n')
            values = line.split(',')
            values = [v.replace(" ", "") for v in values]
            A.append(float(values[0]))
            if STR_DATA:
                B.append(values[1])
            else:
                B.append(float(values[1]))
    return A, B

def Ry(theta):
    '''
    Rotation about y-axis
    '''
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                      [ 0           , 1, 0           ],
                      [-np.sin(theta), 0, np.cos(theta)]])

def wake_zone_refinement(A, B, C, angle2):
    angle1 = np.arctan((B[-1]-A[-1])/(B[0]-A[0]))*(180.0/np.pi)
    D = [C[0] + (B[-1]-A[-1])/np.tan(angle2 * np.pi / 180.), B[-1]]
    dx_wake_refinement = C[0]-A[0]
    gamma = (D[0] - B[0]) / dx_wake_refinement
    return gamma, angle1, angle2
