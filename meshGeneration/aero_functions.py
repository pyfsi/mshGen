'''
airfoil_functions.py
author: Thomas Haas
date: 26.07.2023
description: processing of MegAWES airfoil data
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, CubicSpline
import os
from scipy.interpolate import CubicSpline

def read_airfoil_file(name, filename):
    '''
    Load airfoil coordinates
    '''

    # Check if folder exists
    if not os.path.exists(filename):
        print('Error: ' + filename + ' does not exist')
        return 1

    # Open file to read number of lines
    with open(filename) as f:
        n = len(f.readlines())

    # Re-open file
    data = []
    with open(filename) as f:
        for line, k in zip(f, range(n)):
            data.append([float(val) for val in line.rstrip('\n').split()])

    # Type specific treatment
    if (name == "NACA0012" or name == "Mrev-v2_aileron"):
        reverse_list = data[:int(n/2)]
        reverse_list.reverse()
        for i in range(int(n/2)):
            data[i] = reverse_list[i]

    # Arrange data in x and y coordinates
    x = [val[0] for val in data]
    y = [val[1] for val in data]
    data = np.vstack((x,y)).transpose()

    return data


def read_wing_file(filename):
    '''
    Load wing properties
    '''

    # Check if folder exists
    if not os.path.exists(filename):
        print('Error: ' + filename + ' does not exist')
        return 1

    # Open file to read number of lines
    with open(filename) as f:
        n = len(f.readlines())

    # Re-open file
    with open(filename) as f:

        # Initialize array
        data = np.zeros((n,5))

        # Read lines
        for line, k in zip(f, range(n)):
            data[k,:] = [float(val) for val in line.rstrip('\n').split()]

    return data

def interpolate_wing_shape(filename, y):

    # Read wing file
    if filename == None:
        b = 1000.
        c = 1.
        dx = 0.
        beta = 0.
    else:
        if os.path.exists(filename):
            data = read_wing_file(filename)

            # Span
            b = round(2 * data[:, 1].max(), 2)

            # Wingspan location
            if (abs(y) <= b / 2.):
                y_interp = abs(y)
            elif (abs(y) > b / 2.):
                y_interp = b / 2.

            # Chord:
            c = np.interp(y_interp, data[:, 1], data[:, 2])

            # Shift:
            dx = np.interp(y_interp, data[:, 1], data[:, 4])

            # Beta:
            beta = np.interp(y_interp, data[:, 1], data[:, 3])

        else:
            print("Error - No such wing file: "+filename)
            b = 1.
            c = 1.
            dx = 0.
            beta = 0.

    return b, c, dx, beta

#--------------------------------------------------------#

# Compute the tangent vector at each point
def compute_tangent_vectors(x, y):

    # Initialize
    tangent_vectors = np.zeros((len(x), 2))

    # First order approximation
    for i in range(1, len(x) - 1):
        dx = x[i+1] - x[i-1]
        dy = y[i+1] - y[i-1]
        tangent_vectors[i] = [dx, dy]

    # Handle the special cases for the first and last points
    tangent_vectors[0] = [x[1] - x[0], y[1] - y[0]]  # Tangent vector at the first point
    tangent_vectors[-1] = [x[-1] - x[-2], y[-1] - y[-2]]  # Tangent vector at the last point

    # Normalize the tangent vectors
    normalized_tangent_vectors = np.zeros((len(x), 2))

    for i in range(len(x)):
        magnitude = np.sqrt(tangent_vectors[i, 0]**2 + tangent_vectors[i, 1]**2)
        normalized_tangent_vectors[i] = tangent_vectors[i] / magnitude

    return normalized_tangent_vectors

def rotate_vector(x, y, direction="positive"):
    if direction=="positive":
        new_x = -y
        new_y = x
    elif direction=="negative":
        new_x = y
        new_y = -x
    return [new_x, new_y]

# Compute the normal vector at each point
def compute_vectors(x, y, direction="positive"):

    tangent_vectors = compute_tangent_vectors(x, y)
    normal_vectors = []
    for vector in tangent_vectors:
        normal_vectors.append(rotate_vector(vector[0], vector[1], direction))
    normal_vectors = np.array(normal_vectors)

    return tangent_vectors, normal_vectors

def rotate_airfoil(coordinates, angle):
    shape = coordinates.shape
    data = np.empty(shape)
    data[:, 0] = np.cos(angle * np.pi / 180.0) * coordinates[:, 0] - np.sin(angle * np.pi / 180.0) * coordinates[:, 1]
    data[:, 1] = np.sin(angle * np.pi / 180.0) * coordinates[:, 0] + np.cos(angle * np.pi / 180.0) * coordinates[:, 1]
    return data

def scale_airfoil(coordinates, scale):
    data = scale * coordinates
    return data

def shift_airfoil(coordinates, shift):
    data = coordinates
    data[:, 0] += shift[0]
    data[:, 1] += shift[1]
    return data

def scale_coordinates(x, y, angle, scale, shift):
    coordinates = np.vstack((x, y)).transpose()
    coordinates = rotate_airfoil(coordinates, angle=angle)
    coordinates = scale_airfoil(coordinates, scale=scale)
    coordinates = shift_airfoil(coordinates, shift=shift)
    return coordinates

import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def nearest_point_to_intersection(A, B):
    min_distance = float('inf')
    nearest_point = None

    for x1, y1 in A:
        for x2, y2 in B:
            d = distance(x1, y1, x2, y2)
            if d < min_distance:
                min_distance = d
                nearest_point = (x1, y1)

    return nearest_point, min_distance

def get_point_intersection(AS, normalized_position, target, spline):

    # Initialize
    distance = []
    point = []
    lines = []

    # Loop positions
    for pos in normalized_position:

        # Compute index of point
        idx = int(pos*(AS.scaled_data.N-1))

        # Retrieve position and vectors
        if spline=="front":
            x, y = AS.scaled_data.coordinates_camber[idx]
            vec = [-1,0]
        if spline=="upper":
            x, y = AS.scaled_data.coordinates_upper[idx]
            t = AS.scaled_data.vectors_tangent_upper[idx]
            n = AS.scaled_data.vectors_normal_upper[idx]
            vec = n
        elif spline=="lower":
            x, y = AS.scaled_data.coordinates_lower[idx]
            t = AS.scaled_data.vectors_tangent_lower[idx]
            n = AS.scaled_data.vectors_normal_lower[idx]
            vec = n
        elif spline=="hybrid0":
            x, y = AS.scaled_data.coordinates_camber[idx]
            tc = AS.scaled_data.vectors_tangent_camber[idx]
            nl = AS.scaled_data.vectors_normal_lower[idx]
            vec = (nl - tc)/2
        elif spline == "hybrid1":
            x, y = AS.scaled_data.coordinates_lower[idx]
            nl = AS.scaled_data.vectors_normal_lower[idx]
            nu = AS.scaled_data.vectors_normal_upper[idx]
            vec = [nu[0], (nl[1]-nu[1]/2)]

        # Create intersection list
        line = [[x+vec[0]*val, y+vec[1]*val] for val in np.linspace(0, 1.2 * np.min([np.linalg.norm(element) for element in target]), 501)]
        nearest_point, min_distance = nearest_point_to_intersection(target, line)

        # Append output
        lines.append(line)
        distance.append(min_distance)
        point.append(nearest_point)

    # Find point
    spline_position = normalized_position[np.argmin(distance)]
    target_coordinates = [val for val in point[np.argmin(distance)]]
    target_distance = min(distance)
    target_line = lines[np.argmin(distance)]
    target_angle = np.arctan2(target_coordinates[1], target_coordinates[0])

    # Return
    return spline_position, target_coordinates, target_angle, target_distance, target_line

def write_lists_to_file(A, B, C, header, filename, STR_DATA=False):
    with open(filename, 'w') as file:
        file.write(','.join(header) + '\n')  # Writing the header
        for i in range(len(A)):
            if STR_DATA==True:
                write_statement = ["{:24.12f}".format(A[i]), "{:>4s}".format(B[i]), "{:>4s}".format(C[i])]
            else:
                write_statement = ["{:24.12f}".format(A[i]), "{:24.12f}".format(B[i]), "{:>4s}".format(C[i])]
            file.write(",".join(write_statement)+"\n")  # Writing the elements of A and B and C

def write_more_lists_to_file(A, B, C, D, header, filename, STR_DATA=False):
    with open(filename, 'w') as file:
        file.write(','.join(header) + '\n')  # Writing the header
        for i in range(len(A)):
            write_statement = ["{:24.12f}".format(A[i]), "{:24.12f}".format(B[i]), "{:24.12f}".format(C[i]), "{:>4s}".format(D[i])]
            file.write(",".join(write_statement)+"\n")  # Writing the elements of A and B and C

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

# ------------------- Class AirfoilSection ------------------- #

class RawData:

    def __init__(self, name, filename):

        # Initialization
        self.N = 1001

        # ------------------- Raw coordinates ------------------- #

        # Get raw coordinates
        coordinates = read_airfoil_file(name, filename)

        # Remove 'corrupted' point
        if name=="Mrev-v2":
            self.N_corrupted_point = 38
            self.corrupted_point = coordinates[self.N_corrupted_point]
            coordinates = np.delete(coordinates, self.N_corrupted_point, axis=0)
            self.coordinates = coordinates
        elif name == "Mrev-v2_aileron":
            self.N_corrupted_point = int(len(coordinates) / 2)
            self.corrupted_point = coordinates[self.N_corrupted_point]
            coordinates = np.delete(coordinates, self.N_corrupted_point, axis=0)
            self.coordinates = coordinates
        elif name == "NACA0012":
            self.N_corrupted_point = int(len(coordinates)/2)
            self.corrupted_point = coordinates[self.N_corrupted_point]
            coordinates = np.delete(coordinates, self.N_corrupted_point, axis=0)
            self.coordinates = coordinates

        # Trailing and leading edge indices
        m = np.argmax(coordinates[:, 0])
        n = np.argmin(coordinates[:, 0])
        self.m = m
        self.n = n

        # ------------------- Spline coordinates ------------------- #

        # Camber line control points
        if name == "Mrev-v2":
            xc = [coordinates[n, 0], 0.411, 0.880, coordinates[m, 0]]  # 0.880
            yc = [coordinates[n, 1], 0.090, 0.036, coordinates[m, 1]]  # 0.030
        elif name == "FFA_W3_211":
            xc = [coordinates[n, 0], 0.20, 0.80, coordinates[m, 0]]
            yc = [coordinates[n, 1], 0.03, 0.02, coordinates[m, 1]]
        else:
            xc = [coordinates[n, 0], coordinates[m, 0]]
            yc = [coordinates[n, 1], coordinates[m, 1]]

        # Spline properties
        k = min(len(xc)-1, 3)
        self.tck_upper, u_upper = splprep([np.flip(coordinates[m:n + 1, 0]), np.flip(coordinates[m:n + 1, 1])], s=0) #, k=k, s=0)
        self.tck_lower, u_lower = splprep([coordinates[n:, 0], coordinates[n:, 1]], s=0) #, k=k, s=0)
        self.tck_camber, u_camber = splprep([xc, yc], k=k, s=0) #, k=k, s=0) # Gives same results as CubicSpline

    def plot(self, spline=False):
        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

        # Plot raw data
        ax.plot(self.coordinates[:, 0], self.coordinates[:, 1], '>', color='k', label='raw airfoil points')
        ax.plot(self.corrupted_point[0], self.corrupted_point[1], 's', color='red', markersize=6., label='corrupted point')
        ax.plot(self.coordinates[self.m, 0], self.coordinates[self.m, 1], '*', markersize=15., color='orange', label='trailing edge')
        ax.plot(self.coordinates[self.n, 0], self.coordinates[self.n, 1], '*', markersize=15., color='magenta', label='leading edge')
        ax.plot([self.coordinates[self.m, 0], self.coordinates[self.n, 0]], [self.coordinates[self.m, 1], self.coordinates[self.n, 1]], '--', color='black', label='chord line')

        if spline==True:
            # Spline evaluation
            x_spline_upper, y_spline_upper = splev(np.linspace(0, 1, num=self.N), self.tck_upper)
            x_spline_camber, y_spline_camber = splev(np.linspace(0, 1, num=self.N), self.tck_camber)
            x_spline_lower, y_spline_lower = splev(np.linspace(0, 1, num=self.N), self.tck_lower)

            # Plot splines
            ax.plot(x_spline_upper, y_spline_upper, '-', color='tab:blue', label='upper surface')
            ax.plot(x_spline_camber, y_spline_camber, '-', color='tab:red', label='camber line')
            ax.plot(x_spline_lower, y_spline_lower, '-', color='tab:green', label='lower surface')

        # Layout
        # ax.set_xlim([-0.1, 1.1])
        # ax.set_xticks(np.linspace(0., 1., 11))
        # ax.set_ylim([-0.6, 0.6])
        # ax.set_yticks(np.linspace(-0.5, 0.5, 11))
        ax.set_aspect("equal")
        ax.grid()
        if spline == True:
            ax.legend(loc=1, ncol=2, fontsize=14)
        else:
            ax.legend(loc=1, fontsize=14)
        ax.set_title("Raw airfoil data", fontsize=14)

        return fig, ax

class ScaledData:

    def __init__(self, filename, y, COG, raw_data):

        # Initialization
        self.N = 1001

        # Spline evaluation (from raw data)
        x_spline_upper, y_spline_upper = splev(np.linspace(0, 1, num=raw_data.N), raw_data.tck_upper)
        x_spline_camber, y_spline_camber = splev(np.linspace(0, 1, num=raw_data.N), raw_data.tck_camber)
        x_spline_lower, y_spline_lower = splev(np.linspace(0, 1, num=raw_data.N), raw_data.tck_lower)

        # ------------------- Rotate, Scale and Shift ------------------- #

        b, c, dx, beta = interpolate_wing_shape(filename, y)
        coordinates_upper = scale_coordinates(x_spline_upper, y_spline_upper, -beta, c, [-(COG[0]+dx), -COG[2]])
        coordinates_camber = scale_coordinates(x_spline_camber, y_spline_camber, -beta, c, [-(COG[0]+dx), -COG[2]])
        coordinates_lower = scale_coordinates(x_spline_lower, y_spline_lower, -beta, c, [-(COG[0]+dx), -COG[2]])
        self.c = c

        # ------------------- New spline function after scaling------------------- #

        # Spline characteristics
        self.tck_upper, u_upper = splprep([coordinates_upper[:,0], coordinates_upper[:,1]], s=0)
        self.tck_camber, u_camber = splprep([coordinates_camber[:,0], coordinates_camber[:,1]], s=0)
        self.tck_lower, u_lower = splprep([coordinates_lower[:,0], coordinates_lower[:,1]], s=0)

        # ------------------- Re-evaluate splines ------------------- #

        # Spline re-evaluation (from scaled data)
        x_spline_upper, y_spline_upper = splev(np.linspace(0, 1, num=self.N), self.tck_upper)
        x_spline_camber, y_spline_camber = splev(np.linspace(0, 1, num=self.N), self.tck_camber)
        x_spline_lower, y_spline_lower = splev(np.linspace(0, 1, num=self.N), self.tck_lower)

        self.coordinates_upper = np.vstack((x_spline_upper, y_spline_upper)).transpose()
        self.coordinates_camber = np.vstack((x_spline_camber, y_spline_camber)).transpose()
        self.coordinates_lower = np.vstack((x_spline_lower, y_spline_lower)).transpose()

        # ------------------- Tangential and normal vectors ------------------- #

        # Tangential and normal vectors
        self.vectors_tangent_upper, self.vectors_normal_upper = compute_vectors(x_spline_upper, y_spline_upper)
        self.vectors_tangent_camber, self.vectors_normal_camber = compute_vectors(x_spline_camber, y_spline_camber)
        self.vectors_tangent_lower, self.vectors_normal_lower = compute_vectors(x_spline_lower, y_spline_lower, direction="negative")

    def plot(self, fignum=None, chord=True, camber=True, vectors=False):

        if fignum:
            # Retrieve axis
            ax = plt.figure(fignum).axes[0]
        else:
            # Create figure
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

        # Plot scaled data
        if chord:
            ax.plot([self.coordinates_camber[0,0], self.coordinates_camber[-1,0]], [self.coordinates_camber[0,1], self.coordinates_camber[-1,1]], 'k--', label='chord line')
        ax.plot(self.coordinates_upper[:,0], self.coordinates_upper[:,1], color="tab:blue", label='upper surface')
        if camber:
            ax.plot(self.coordinates_camber[:,0], self.coordinates_camber[:,1], color="tab:red", label='camber line')
        ax.plot(self.coordinates_lower[:,0], self.coordinates_lower[:,1], color="tab:green", label='lower surface')
        # Plot vectors
        if vectors:
            x_spline_upper, y_spline_upper = splev(np.linspace(0, 1, num=self.N), self.tck_upper)
            x_spline_camber, y_spline_camber = splev(np.linspace(0, 1, num=self.N), self.tck_camber)
            x_spline_lower, y_spline_lower = splev(np.linspace(0, 1, num=self.N), self.tck_lower)
            for k in range(self.N):
                ax.quiver(x_spline_upper[k],y_spline_upper[k], self.vectors_normal_upper[k][0], self.vectors_normal_upper[k][1], color='blue')
                ax.quiver(x_spline_camber[k],y_spline_camber[k], self.vectors_normal_camber[k][0], self.vectors_normal_camber[k][1], color='red')
                ax.quiver(x_spline_lower[k],y_spline_lower[k], self.vectors_normal_lower[k][0], self.vectors_normal_lower[k][1], color='green')
                ax.quiver(x_spline_upper[k],y_spline_upper[k], self.vectors_tangent_upper[k][0], self.vectors_tangent_upper[k][1], color='blue')
                ax.quiver(x_spline_camber[k],y_spline_camber[k], self.vectors_tangent_camber[k][0], self.vectors_tangent_camber[k][1], color='red')
                ax.quiver(x_spline_lower[k],y_spline_lower[k], self.vectors_tangent_lower[k][0], self.vectors_tangent_lower[k][1], color='green')

        # Layout
        # ax.set_xlim([-2.5, 3.5])
        # ax.set_xticks(np.linspace(-2, 3, 11))
        # ax.set_ylim([-3, 3])
        # ax.set_yticks(np.linspace(-2.5, 2.5, 11))
        ax.set_aspect("equal")
        if not fignum:
            ax.grid()
            ax.legend(loc=1, fontsize=14)
        ax.set_title("Scaled airfoil data", fontsize=14)

        return

class AirfoilSection:

    def __init__(self, name, filename1, filename2, COG, y):

        # Initialize
        self.name = name
        self.filename1 = filename1
        self.filename2 = filename2
        self.y = y
        self.COG = COG

        # Process raw airfoil data
        self.raw_data = RawData(self.name, self.filename1)

        # Scale airfoil data
        self.scaled_data = ScaledData(self.filename2, self.y, self.COG, self.raw_data)
