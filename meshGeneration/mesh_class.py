'''
generate_section_3Dmesh.py
author: Thomas Haas
date: 22.09.2023
description: Generate files for blockMesh utility
'''

# Imports
from matplotlib import pyplot as plt
import numpy as np

from math import ceil, floor
import os
from scipy.interpolate import splprep, splev
import mshGen.meshGeneration.default_files.cmesh_default_parameters as cparam
import mshGen.meshGeneration.aero_functions as af
import mshGen.meshGeneration.mesh_functions as mf
from mshGen.meshGeneration.mesh_functions import vertex, block, edge

# from meshGeneration.generate_default_airfoil_files import AirfoilSection, interpolate_wing_shape, interpolate_lifting_surface_shape
# from meshGeneration.generate_default_files import interpolate_lifting_surface_shape
from importlib import reload
reload(af)
reload(mf)

# ---------------------- Class 'Mesh' ---------------------- #
class Mesh:
    '''
    Object creation: Retrieve user parameters
    '''
    def __init__(self, names, files, params, flags):

        # Fill in user options...

        # Case name:
        casename = names[0]
        airfoil_name = names[1]
        wing_name = names[2]
        self.casename = casename
        self.airfoil_name = airfoil_name
        self.wing_name = wing_name

        # Files:
        self.airfoil_file = files[0]
        self.wing_file = files[1]
        self.defaultfolder = files[2]
        self.outputfolder = files[3]

        # Mesh parameters:
        self.dim = params[0]
        self.COG = params[1]
        self.slice_locations = params[2]
        self.layer_resolution = params[3]
        self.layer_grading = params[4]
        self.layer_aileron = params[5]
        self.mesh_scale = params[6]
        self.mesh_level = params[7]
        self.grading_relaxation = params[8]

        # Flags:
        self.WRITE = flags[0]
        self.PRINT = flags[1]
        self.PLOT = flags[2]
        self.DEFAULT = flags[3]

        # Print message
        if self.PRINT == True:
            print("")
            print("==============================================================================")
            print("Mesh generation for case: "+self.casename)
            print("==============================================================================")
            print("")

    # ---------------------- Initialize mesh structure ---------------------- #
    '''
    Initialize mesh structure: Compute number of layers
    '''
    def initialize_mesh(self):

        # Compute wing/airfoil parameters at root
        b, c0, dx0, beta0 = af.interpolate_wing_shape(self.wing_file, y=0.0)
        self.b = b
        self.c0 = c0
        self.dx0 = dx0
        self.beta0 = beta0

        # Compute mesh structure
        self.N_slices = len(self.slice_locations)
        self.N_layers = self.N_slices - 1
        self.N_flow_layers = len([element for element in self.slice_locations if element > self.b / 2.])
        self.N_wing_layers = self.N_layers - self.N_flow_layers

        # Define aileron
        if self.layer_aileron != None:
            self.AILERON = True
            if (self.layer_aileron==(self.N_wing_layers-1) and self.N_wing_layers>1):
                self.AILERON=False
                self.layer_aileron=None
        else:
            self.AILERON=False

        # Print message
        if self.PRINT == True:
            print("Wing/Airfoil dimensions:")
            print("Wing span: b = "+"{:.3f}".format(self.b))
            print("Root chord: c0 = "+"{:.3f}".format(self.c0))
            print("Root offset: dx0 = " + "{:.3f}".format(self.dx0))
            print("Root twist: beta0 = "+"{:.3f}".format(self.beta0))
            print("")
            print("Mesh structure ("+self.dim+" case)")
            print("Nbr slices:  "+str(self.N_slices))
            print("Nbr layers:  "+str(self.N_layers))
            print("Wing layers: "+str(self.N_wing_layers))
            print("Flow layers: "+str(self.N_flow_layers))
            if self.AILERON==True:
                print("Aileron: include aileron at layer "+str(self.layer_aileron))
            else:
                print("Aileron: do not include aileron...")
            print("")

    # ---------------------- Generate vertices at y-location ---------------------- #
    '''
    Generate vertices for different y-locations based on scaled dimensions of airfoil sections
    '''
    def generate_default(self):

        # Only generate default files if FLAG is turned on.
        if self.DEFAULT:

            # --------------------- Root airfoil -------------------- #
            # AirfoilSection(name, filename1, filename2, COG, y)
            AS = af.AirfoilSection(self.airfoil_name, self.airfoil_file, self.wing_file, self.COG, y=0.0)

            # Plots
            if self.PLOT==True:
                # AS.raw_data.plot(spline=True)
                AS.scaled_data.plot(chord=False)
                fig_num = plt.get_fignums()
                fig = plt.figure(fig_num[-1])
                ax = plt.figure(fig_num[-1]).axes[0]
                fig.set_size_inches(9, 6)

            # --------------------- Outer limits of C-mesh -------------------- #
            # Outer domain dimensions
            c0 = AS.scaled_data.c
            d = self.mesh_scale * c0

            # Compute limits of outer domain (required for interpolation and for plotting)
            N = 100
            outer_domain = [[2 * d, d * value] for value in np.linspace(0, 1, N+1)]
            outer_domain += [[2 * d * value, d ] for value in np.linspace(1, 0, 5*N+1)]
            outer_domain += [[d * np.cos(value), d * np.sin(value)] for value in np.linspace(np.pi / 2, 2 * np.pi / 2, N+1)]
            outer_domain += [[d * np.cos(value), d * np.sin(value)] for value in np.linspace(2 * np.pi / 2, 3 * np.pi / 2, N+1)]
            outer_domain += [[2 * d * value, -d ] for value in np.linspace(0, 1, 5*N+1)]
            outer_domain += [[2 * d , d * value] for value in np.linspace(1, 0, N+1)]

            # Add outer domain to airfoil plot
            if self.PLOT==True:
                outer_domain_line = ax.plot([value[0] for value in outer_domain], [value[1] for value in outer_domain], color='tab:purple', label='outer domain')
                legend = ax.legend(loc=2)
                legend.get_lines().append(outer_domain_line)

            # --------------------- Default positions of interior vertices  -------------------- #
            # INPUT: Points along airfoil camber and surfaces

            # Define position (relative to spline) of the 16 interior control points
            if self.airfoil_name == "NACA0012":
                interior_positions_list = [1.0] + [0.9, 0.89, 0.9] + [0.7, 0.69, 0.7] \
                                        + [0.3, 0.29, 0.3] + [0.03, 0.03] + [0.06, 0., 0.1, 0.11]
            elif self.airfoil_name == "Mrev-v2":
                interior_positions_list = [1.0] + [0.9, 0.89, 0.9] + [0.7, 0.69, 0.7] \
                                          + [0.3, 0.27, 0.3] + [0.1, 0.1] + [0.06, 0., 0.1, 0.11]
            else:
                interior_positions_list = [1.0] + [0.9, 0.88, 0.9] + [0.7, 0.67, 0.7] \
                                        + 3*[0.3] + [0.1, 0.1] + [0.06, 0., 0.1, 0.11]

            # Associate spline to points
            spline_list = ['c'] + ['s', 'c', 'p'] + ['s', 'c', 'p'] \
                        + ['s', 'c', 'p'] + ['s', 'p'] + 4 * ['c']

            # Associate index to control points
            interior_comment_list = [str(i) for i in list(range(len(interior_positions_list)))]

            # --------------------- Position of lines normal to semi-circle extents  -------------------- #

            # Manual: Correct (overwrite) position of control points 7 and 9 to be normal to semi-circle extents.
            if self.airfoil_name=="NACA0012":
                upper_position0, target, angle, distance, upper_line = af.get_point_intersection(AS, np.linspace(0.2, 0.5, 31), [[0, +d]], "upper")
                lower_position0, target, angle, distance, lower_line = af.get_point_intersection(AS, np.linspace(0.2, 0.5, 31), [[0, -d]], "lower")
                interior_positions_list[7] = upper_position0
                interior_positions_list[9] = lower_position0
                if self.PLOT==True:
                    ax.plot([value[0] for value in upper_line], [value[1] for value in upper_line], '--', color='tab:blue')
                    ax.plot([value[0] for value in lower_line], [value[1] for value in lower_line], '--', color='tab:green')

            # --------------------- Default positions of exterior vertices  -------------------- #
            # Find block vertices on outer domain (exterior_positions_list):
            #   - Vertices of wake region are specified manually
            #   - Vertices on C-shape are the intersections between airfoil normals and outer domain limits

            # Upper half of wake domain extents (including points from TE to outlet)
            # Note: Position of point 4 is manually changed later on
            x_TE = c0 - self.COG[0]
            z_TE = 0. - self.COG[2]
            dx_wake_refinement = 0.5 * c0   # INPUT
            exterior_positions_list = [[x_TE+dx_wake_refinement, z_TE], [2 * d, 0], [2 * d, d], [x_TE+dx_wake_refinement, d]]

            # Extension from upper surface (without LE)
            for position in [interior_positions_list[i] for i in [0,1,4,7,10]]:
                upper_position, target, angle, distance, upper_line = af.get_point_intersection(AS, [position], outer_domain, "upper")
                exterior_positions_list.append(target)
                if self.PLOT==True:
                    ax.plot([value[0] for value in upper_line], [value[1] for value in upper_line], '--', color='tab:blue')

            # Extension from leading edge
            if self.airfoil_name=="NACA0012":
                interp_method = "front"
            elif self.airfoil_name == "Mrev-v2":
                interp_method = "front"
                # interp_method = "hybrid0"
            else:
                interp_method = "front"
            for position in [0.0]:
                camber_position, target, angle, distance, camber_line = af.get_point_intersection(AS, [position], outer_domain, interp_method)
                exterior_positions_list.append(target)
                if self.PLOT==True:
                    ax.plot([value[0] for value in camber_line], [value[1] for value in camber_line], '--', color='tab:red')

            # Extension from lower surface
            for position in [interior_positions_list[i] for i in [11, 9, 6, 3, 0]]:
                lower_position, target, angle, distance, lower_line = af.get_point_intersection(AS, [position], outer_domain,"lower")
                exterior_positions_list.append(target)
                if self.PLOT == True:
                    ax.plot([value[0] for value in lower_line], [value[1] for value in lower_line], '--', color='tab:green')

            # Lower half of wake domain extent
            exterior_positions_list += [[x_TE+dx_wake_refinement, -d]]
            exterior_positions_list += [[2*d, -d]]

            # Manual edit of position of specific vertices
            if self.airfoil_name == "NACA0012":

                # Manual: vertices 25 (exterior point 3) and 37 (exterior point 15)
                gamma, angle1, angle2 = mf.wake_zone_refinement([x_TE, z_TE], exterior_positions_list[4], [x_TE + dx_wake_refinement, z_TE], angle2 = 61.5)
                if self.mesh_scale <= 1.5:
                    gamma, angle1, angle2 = mf.wake_zone_refinement([x_TE, z_TE], exterior_positions_list[4], [x_TE + dx_wake_refinement, z_TE], angle2 = angle1)
                exterior_positions_list[3]  = [exterior_positions_list[4][0]  + gamma*dx_wake_refinement, +d]
                exterior_positions_list[15] = [exterior_positions_list[14][0] + gamma*dx_wake_refinement, -d]
                if self.PLOT == True:
                    ax.plot([exterior_positions_list[i][0] for i in [3, 0, 15]], [exterior_positions_list[i][1] for i in [3, 0, 15]], '--', color='tab:orange')

            elif (self.airfoil_name == "Mrev-v2"):

                # Manual: vertices 29 and 33 (C-region - respectively exterior points 7 and 11)
                exterior_positions_list[7]  = [0, +d]
                exterior_positions_list[11] = [0, -d]

                # Manual: vertices 30 and 32 (C-region - respectively exterior points 8 and 10)
                exterior_positions_list[8]  = [-d*np.cos(np.pi/4), +d*np.sin(np.pi/4)]
                exterior_positions_list[10] = [-d*np.cos(np.pi/4), -d*np.sin(np.pi/4)]

                # Manual: vertices 34, 35, 36 (TE and wake - respectively exterior points 12, 13, 14)
                exterior_positions_list[12][0] = exterior_positions_list[6][0]
                exterior_positions_list[13][0] = exterior_positions_list[5][0]
                exterior_positions_list[14][0] = exterior_positions_list[4][0]

                # Manual: vertices 25 (exterior point 3) and 37 (exterior point 15)
                gamma, angle1, angle2 = mf.wake_zone_refinement([x_TE, z_TE], exterior_positions_list[4], [x_TE + dx_wake_refinement, z_TE], angle2 = 55.0)
                exterior_positions_list[3]  = [exterior_positions_list[4][0]  + gamma*dx_wake_refinement, +d]
                exterior_positions_list[15] = [exterior_positions_list[14][0] + gamma*dx_wake_refinement, -d]

                if self.PLOT == True:
                    ax.plot([exterior_positions_list[i][0] for i in [3, 0, 15]], [exterior_positions_list[i][1] for i in [3, 0, 15]], '--', color='tab:orange')

            elif (self.airfoil_name == "FFA_W3_211"):

                # Manual: vertices 29 and 33 (C-region - respectively exterior points 7 and 11)
                exterior_positions_list[7]  = [0, +d]
                exterior_positions_list[11] = [0, -d]

                # Manual: vertices 30 and 32 (C-region - respectively exterior points 8 and 10)
                exterior_positions_list[8]  = [-d*np.cos(np.pi/4), +d*np.sin(np.pi/4)]
                exterior_positions_list[10] = [-d*np.cos(np.pi/4), -d*np.sin(np.pi/4)]

                # Manual: vertices 26, 27, 28 (TE and wake - respectively exterior points 4, 5, 6)
                exterior_positions_list[4] = [0.75 + d/np.tan(70.*np.pi/180.), d] #exterior_positions_list[6][0]
                exterior_positions_list[5] = [0.65 + d/np.tan(75.*np.pi/180.), d] #exterior_positions_list[5][0]
                exterior_positions_list[6] = [0.45 + d/np.tan(80.*np.pi/180.), d] #exterior_positions_list[4][0]

                # Manual: vertices 34, 35, 36 (TE and wake - respectively exterior points 12, 13, 14)
                exterior_positions_list[12][0] = exterior_positions_list[6][0]
                exterior_positions_list[13][0] = exterior_positions_list[5][0]
                exterior_positions_list[14][0] = exterior_positions_list[4][0]

                # Manual: vertices 25 (exterior point 3) and 37 (exterior point 15)
                gamma, angle1, angle2 = mf.wake_zone_refinement([x_TE, z_TE], exterior_positions_list[4], [x_TE + dx_wake_refinement, z_TE], angle2 = 55.0)
                exterior_positions_list[3]  = [exterior_positions_list[4][0]  + gamma*dx_wake_refinement, +d]
                exterior_positions_list[15] = [exterior_positions_list[14][0] + gamma*dx_wake_refinement, -d]

                if self.PLOT == True:
                    ax.plot([exterior_positions_list[i][0] for i in [3, 0, 15]], [exterior_positions_list[i][1] for i in [3, 0, 15]], '--', color='tab:orange')


            # Compute index of exterior points for commenting in default file
            n_helper_points = 6 # Additional interior points that are computed on 1/4 thickness line
            n_int_pts = len(interior_positions_list) + n_helper_points  # 6: nbr of additional interior interpolated points
            exterior_comment_list = [str(i) for i in list(range(n_int_pts, n_int_pts + len(exterior_positions_list)))]
            if self.PLOT == True:
                ax.plot([x[0] for x in exterior_positions_list], [x[1] for x in exterior_positions_list], '*', color='tab:orange')

            # --------------------- Default edge resolution and grading  -------------------- #
            edge_params = cparam.get_default_edge_parameters(self.airfoil_name, self.mesh_scale, self.mesh_level)
            edge_resolution = edge_params[0]
            edge_grading1 = edge_params[1]
            edge_grading2 = edge_params[2]
            edge_comment_list = ["e" + "{:02d}".format(i) for i in list(range(len(edge_resolution)))]

            # --------------------- Default positions of exterior vertices  -------------------- #
            if self.WRITE == True:
                filename1 = self.defaultfolder + "/default_interior_points_"+self.casename+".csv"
                af.write_lists_to_file(interior_positions_list, spline_list, interior_comment_list,['position', 'spline', 'vertex'], filename1, STR_DATA=True)

                filename2 = self.defaultfolder + "/default_exterior_points_"+self.casename+".csv"
                af.write_lists_to_file([x[0] for x in exterior_positions_list], [x[1] for x in exterior_positions_list], exterior_comment_list, ['x-coord', 'y-coord', 'vertex'], filename2, STR_DATA=False)

                filename3 = self.defaultfolder + "/default_resolution_grading_"+self.casename+".csv"
                af.write_more_lists_to_file(edge_resolution, edge_grading1, edge_grading2, edge_comment_list,['resolution', 'grad1', 'grad2', 'edge'], filename3, STR_DATA=False)

                # print message
                if self.PRINT == True:
                    print("Default files written...")
                    print(filename1)
                    print(filename2)
                    print(filename3)
                    print("")

            # # Manual calculations
            # def calculate_line_length(x, y):
            #     total_length = 0
            #     for i in range(len(x) - 1):
            #         x1, y1 = x[i], y[i]
            #         x2, y2 = x[i + 1], y[i + 1]
            #         distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            #         total_length += distance
            #     return total_length
            #
            # index = [0, 1, 4, 7, 10, 13]
            # values = [interior_positions_list[i] for i in index]
            # tck = AS.scaled_data.tck_upper
            # for v0, v1 in zip(values[:-1], values[1:]):
            #     x, y = splev(np.linspace(v0, v1, num=100), tck)
            #     d = calculate_line_length(x, y)
            #     print(v0, v1, "{:.3f}".format(d))
            #
            # index = [0, 3, 6, 9, 11, 13]
            # values = [interior_positions_list[i] for i in index]
            # tck = AS.scaled_data.tck_lower
            # for v0, v1 in zip(values[:-1], values[1:]):
            #     x, y = splev(np.linspace(v0, v1, num=100), tck)
            #     d = calculate_line_length(x, y)
            #     print(v0, v1, "{:.3f}".format(d))

    # ---------------------- Generate vertices at y-location ---------------------- #
    '''
    Generate vertices for different y-locations based on scaled dimensions of airfoil sections
    '''
    def generate_vertices(self):

        # Create vertices
        vertex_list = []

        # Loop specific spanwise locations
        for slice, y in enumerate(self.slice_locations):

            # -------------------------------------------- #
            '''
            Compute airfoil section (scaled,  at root)
            '''

            # Compute airfoil section (scaled,  at root)
            s = af.AirfoilSection(self.airfoil_name, self.airfoil_file, self.wing_file, self.COG, y=y)
            coord = np.flip(s.scaled_data.coordinates_upper, axis=0)
            coord = np.concatenate((coord, s.scaled_data.coordinates_lower[1:-1,:]))
            camber = s.scaled_data.coordinates_camber

            # Plot airfoil and camber points
            if self.PLOT:
                mf.plot_airfoil(coord, camber)

            # -------------------------------------------- #
            '''
            Compute B-splines
            '''

            # Define splines functions at pressure, suction, and camber lines
            tck_p = s.scaled_data.tck_lower #splprep([coord[mLE:,0], coord[mLE:,1]], s=0)
            tck_s = s.scaled_data.tck_upper #splprep([np.flip(coord[mTE:mLE+1,0]), np.flip(coord[mTE:mLE+1,1])], s=0)
            tck_c = s.scaled_data.tck_camber #splprep([camber[:,0], camber[:,1]], s=0)
            if slice == 0:
                self.tck = [tck_p, tck_c, tck_s]

            # Plot the B-Splines
            if self.PLOT:
                ax = plt.figure(slice+1).axes
                mf.plot_spline(ax, tck_p, 'tab:blue')
                mf.plot_spline(ax, tck_c, 'tab:blue')
                mf.plot_spline(ax, tck_s, 'tab:blue')

            # -------------------------------------------- #
            '''
            Compute and plot local vertices
            '''

            # Create vertices
            N = len(vertex_list)
            slice_vertex_list = []

            # INPUT: Main control points on surface and camber (including 2 helper points on camber)
            filename= self.defaultfolder + "/default_interior_points_"+self.casename+".csv"
            if os.path.exists(filename):
                pos_list,line_list = mf.read_lists_from_file(filename, STR_DATA=True)
            else:
                print("Input file is missing...")

            # MANUAL: Modify position of vertices 4, 5, and 6 for aileron cut-out
            if (self.AILERON==True and (slice==self.layer_aileron or slice==self.layer_aileron+1)):

                # Position of cut-off (on upper surface) in percent of chord
                x_cut = 0.7

                # Rotation of outward-pointing normal vector (positive angle: tilt to right)
                angle = 20.*(np.pi/180.)

                # Find location on upper spline (unscaled)
                n_p = 1001
                x, z = splev(np.linspace(0, 1, n_p), s.raw_data.tck_upper)
                m4 = np.argmin(np.abs(x - x_cut))
                p4 = m4/(n_p-1)

                # Create normal to upper spline
                out = splev(p4, s.raw_data.tck_upper, der=1)
                dzdx = out[1]/out[0]
                vec_tan = np.array([1, 0, dzdx])/np.linalg.norm(np.array([1, 0, dzdx]))
                vec_nor = np.cross(vec_tan, np.array([0,1,0]))
                # Note: normal vector points outwards!

                # Optional: tilt normal vector by angle
                Ry = mf.Ry(angle)
                vec_nor = (Ry @ vec_nor)
                normal_line = [[x[m4] + vec_nor[0]*i, z[m4] + vec_nor[2]*i] for i in np.linspace(0,-1,n_p)]

                # Find intersection with camber spline
                x, z = splev(np.linspace(0, 1, n_p), s.raw_data.tck_camber)
                spline = [[x[i], z[i]] for i in range(len(x))]
                x_intersec, d = af.nearest_point_to_intersection(spline, normal_line) # Point on camber spline
                m5 = np.argmin(np.abs(x - x_intersec[0]))
                p5 = m5/(n_p-1)

                # Find intersection with lower spline
                x, z = splev(np.linspace(0, 1, n_p), s.raw_data.tck_lower)
                spline = [[x[i], z[i]] for i in range(len(x))]
                x_intersec, d = af.nearest_point_to_intersection(spline, normal_line) # Point on camber spline
                m6 = np.argmin(np.abs(x - x_intersec[0]))
                p6 = m6/(n_p-1)

                # Overwrite positions
                pos_list[4] = p4
                pos_list[5] = p5
                pos_list[6] = p6

            for k, [pos, line] in enumerate(zip(pos_list, line_list)):
                tck = mf.get_tck(line, [tck_p, tck_c, tck_s])
                slice_vertex_list.append(vertex(k+N, y, pos, line, tck))

            # # MANUAL: Additional control points at "1/4-thickness"
            pair_list = [[4,5], [5,6], [7,8], [8,9], [10,14], [14,11]]
            for k, pair in enumerate(pair_list, start=len(slice_vertex_list)):
                slice_vertex_list.append(vertex(k+N, y, [], [], []))
                slice_vertex_list[k].coord = np.mean([slice_vertex_list[v].coord for v in pair], axis=0)

            # INPUT: C-Mesh control points
            filename= self.defaultfolder + "/default_exterior_points_"+self.casename+".csv"
            if os.path.exists(filename):
                xp_list, zp_list = mf.read_lists_from_file(filename)
            else:
                print("Input file is missing...")

            # Scale inputs
            xp_array = np.array(xp_list)
            zp_array = np.array(zp_list)
            # xp_array = self.c0*np.array(xp_list)
            # zp_array = self.c0*np.array(zp_list)
            for k, [xp, zp] in enumerate(zip(xp_array, zp_array), start=len(slice_vertex_list)):
                slice_vertex_list.append(vertex(k+N, y, [], [], []))
                slice_vertex_list[k].coord = [xp, y, zp]

            # Plot vertices
            if self.PLOT:
                ax = plt.figure(slice+1).axes
                mf.plot_vertices(ax, slice_vertex_list)

            # -------------------------------------------- #
            '''
            Save local vertices and add them to main vertex list
            '''
            # Add local vertices to main vertex list
            vertex_list += slice_vertex_list

            # Print vertices' coordinates in OpenFOAM format
            if self.WRITE==True:
                filename=self.outputfolder+"/list_vertices_slice"+str(slice)
                mf.write_vertices_to_file(filename, slice_vertex_list, slice)

                # print message
                if self.PRINT==True:
                    if slice==0: print("Generate mesh vertices:")
                    print("Slice "+"{:2d}".format(slice+1)+"/"+"{:2d}".format(self.N_slices)+": List of vertices written...")
                    if slice==(self.N_slices-1): print("")

        # -------------------------------------------- #
        '''
        Add vertices to mesh object
        '''
        self.vertices = vertex_list
        self.N_vertices = len(vertex_list)

    # ---------------------- Generate blocks ---------------------- #
    '''
    Generate blocks
    '''
    def generate_blocks(self):

        # fig_nums = plt.get_fignums()
        N_slices = self.N_slices
        N_vertices = len(self.vertices)

        # Create blocks from vertices
        vertex_list = self.vertices
        block_list = []

        # Loop through (N-1) slices to generate blocks
        for layer in range(self.N_layers):

            # -------------------------------------------- #
            '''
            Compute blocks
            '''

            # Layer blocks
            layer_block_list = []

            # MANUAL: Specify vertices of interior blocks
            block_pts2d = [[0, 1, 2, 3], [1, 4, 5, 2], [2, 5, 6, 3]]
            block_pts2d +=[[4, 7, 18, 16], [16, 18, 8, 5], [5, 8, 19, 17], [17, 19, 9, 6]]
            block_pts2d +=[[7, 10, 20, 18], [18, 20, 12, 8], [8, 12, 21, 19], [19, 21, 11, 9]]
            block_pts2d +=[[10, 13, 12, 20], [13, 11, 21, 12]]
            N_interior = len(block_pts2d)

            # MANUAL: Specify vertices of exterior blocks
            block_pts2d += [[22, 23, 24, 25], [0, 22, 25, 26]]
            block_pts2d += [[1, 0, 26, 27], [4, 1, 27, 28], [7, 4, 28, 29]]
            block_pts2d += [[10, 7, 29, 30], [13, 10, 30, 31], [11, 13, 31, 32], [9, 11, 32, 33]]
            block_pts2d += [[6, 9, 33, 34], [3, 6, 34, 35], [0, 3, 35, 36]]
            block_pts2d += [[22, 0, 36, 37], [23, 22, 37, 38]]
            N_exterior = len(block_pts2d) - N_interior

            # Add blocks to list
            N_pts = int(N_vertices/N_slices)
            N_blck = len(block_pts2d)
            for k, pts2d in enumerate(block_pts2d):
                layer_block_list.append(block(k+int(layer*N_blck), pts2d, layer, N_pts))

            # Plot blocks
            if self.PLOT:
                ax = plt.figure(layer + 1).axes
                mf.plot_blocks(ax, vertex_list, layer_block_list)

            # ---------------------- Block resolution ---------------------- #
            '''
            Specify block resolution
            '''

            # INPUT: block resolution
            filename = self.defaultfolder+"/default_resolution_grading_"+self.casename+".csv"
            if os.path.exists(filename):
                res_list, grd_list = mf.read_lists_from_file(filename)
            else:
                print("Input file is missing...")

            # Retrieve resolution
            n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = [int(n) for n in res_list]
            ny = self.layer_resolution[layer]

            # Resolution of interior blocks
            resolution_list = [[n3, ny, n3]] + [[n4, ny, n3]]*2
            resolution_list +=[[n5, ny, n8], [n5, ny, n7], [n5, ny, n7], [n5, ny, n8]]
            resolution_list +=[[n6, ny, n8], [n6, ny, n7], [n6, ny, n7], [n6, ny, n8]]
            resolution_list +=[[n7, ny, n8]]*2

            # Resolution of exterior blocks
            resolution_list += [[ni, ny, n0] for ni in [n1,n2,n3,n4,n5,n6,n7]]
            resolution_list += [[ni, ny, n0] for ni in [n7,n6,n5,n4,n3,n2,n1]]

            # Specify resolution of blocks
            for k, [blck, res] in enumerate(zip(layer_block_list, resolution_list)):
                blck.resolution = res

            # ---------------------- Block grading ---------------------- #
            '''
            Specify block grading
            '''

            # INPUT: Block grading
            filename = self.defaultfolder+"/default_resolution_grading_"+self.casename+".csv"
            if os.path.exists(filename):
                res_list, grd1_list, grd2_list = mf.read_more_lists_from_file(filename)
            else:
                print("Input file is missing...")

            # Retrieve grading
            g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10 = grd1_list
            h0, h1, h2, h3, h4, h5, h6, h7, h8, g9, g10 = grd2_list
            gy = self.layer_grading[layer]

            # grading of interior blocks
            grading_list = [[i for i in [g3, gy, g3] for _ in range(4)]]      # Block 0
            grading_list +=[[i for i in [g4, gy, g3] for _ in range(4)]]      # Block 1
            grading_list +=[[i for i in [g4, gy, 1./g3] for _ in range(4)]]   # Block 2
            grading_list +=[[i for i in [g5, gy, g8] for _ in range(4)]]      # Block 3 to 6
            grading_list +=[[i for i in [g5, gy, g7] for _ in range(4)]]
            grading_list +=[[i for i in [g5, gy, 1./g7] for _ in range(4)]]
            grading_list +=[[i for i in [g5, gy, 1./g8] for _ in range(4)]]
            grading_list +=[[i for i in [g6, gy, g8] for _ in range(4)]]      # Block 7 to 10
            grading_list +=[[i for i in [g6, gy, g7] for _ in range(4)]]
            grading_list +=[[i for i in [g6, gy, 1./g7] for _ in range(4)]]
            grading_list +=[[i for i in [g6, gy, 1./g8] for _ in range(4)]]
            grading_list +=[[i for i in [g7, gy, g8] for _ in range(4)]]      # Block 11
            grading_list +=[[i for i in [1./g7, gy, g8] for _ in range(4)]]   # Block 12

            # grading of exterior blocks
            grading_list +=[[i for i in [g1, gy, g0] for _ in range(4)]]      # Block 13
            grading_list +=[[i for i in [g2, gy, g0] for _ in range(4)]]      # Block 14
            grading_list +=[[i for i in [1./g3, gy, g0] for _ in range(4)]]      # Block 15 to 19
            grading_list +=[[i for i in [1./g4, gy, g0] for _ in range(4)]]
            grading_list +=[[i for i in [1./g5, gy, g0] for _ in range(4)]]
            grading_list +=[[i for i in [1./g6, gy, g0] for _ in range(4)]]
            grading_list +=[[i for i in [1./g7, gy, g0] for _ in range(4)]]
            grading_list +=[[i for i in [g7, gy, g0] for _ in range(4)]]      # Block 20 to 24
            grading_list +=[[i for i in [g6, gy, g0] for _ in range(4)]]
            grading_list +=[[i for i in [g5, gy, g0] for _ in range(4)]]
            grading_list +=[[i for i in [g4, gy, g0] for _ in range(4)]]
            grading_list +=[[i for i in [g3, gy, g0] for _ in range(4)]]
            grading_list +=[[i for i in [1./g2, gy, g0] for _ in range(4)]]      # Block 25
            grading_list +=[[i for i in [1./g1, gy, g0] for _ in range(4)]]      # Block 26

            # Manual changes:
            # Vertical grading (Change relative to h0, g0)
            grading_list[13][8:] = [g10, g9, g9, g10]
            grading_list[14][8:] = [g0, g10, g10, g0]
            grading_list[25][8:] = [g10, g0, g0, g10]
            grading_list[26][8:] = [g9, g10, g10, g9]

            # Horizontal/Circular grading
            grading_list[13][:4] = [g1, g1, h1, h1]
            grading_list[14][:4] = [g2, g2, h2, h2]
            grading_list[15][:4] = [1/g3, 1/g3, 1/h3, 1/h3]
            grading_list[16][:4] = [1/g4, 1/g4, 1/h4, 1/h4]
            grading_list[17][:4] = [1/g5, 1/g5, 1/h5, 1/h5]
            grading_list[18][:4] = [1/g6, 1/g6, 1/h6, 1/h6]
            grading_list[19][:4] = [1/g7, 1/g7, 1/h7, 1/h7]

            # Manual overwrite for blocks 20-26
            grading_list[20][:4] = [g7, g7, h7, h7]
            grading_list[21][:4] = [g6, g6, h6, h6]
            grading_list[22][:4] = [g5, g5, h5, h5]
            grading_list[23][:4] = [g4, g4, h4, h4]
            grading_list[24][:4] = [g3, g3, h3, h3]
            grading_list[25][:4] = [1/g2, 1/g2, 1/h2, 1/h2]
            grading_list[26][:4] = [1/g1, 1/g1, 1/h1, 1/h1]

            # vertical grading relaxation
            if ((self.N_flow_layers > 0) and (layer >= self.N_wing_layers-1)):
                ys = self.slice_locations[self.N_wing_layers-1]
                ye = self.slice_locations[-1]
                y0 = self.slice_locations[layer]
                y1 = self.slice_locations[layer+1]
                relaxation0 = 1 + (y0 - ys) / (ye - ys) * (self.grading_relaxation - 1)
                relaxation1 = 1 + (y1 - ys) / (ye - ys) * (self.grading_relaxation - 1)
                for n_block in range(N_interior, N_blck):
                    grading_list[n_block][8:10] = [grading/relaxation0 for grading in grading_list[n_block][8:10]]
                    grading_list[n_block][10:12] =[grading/relaxation1 for grading in grading_list[n_block][10:12]]

            # Specify grading of blocks
            for k, [blck, grd] in enumerate(zip(layer_block_list, grading_list)):
                blck.grading = grd #[elem for elem in grd for _ in range(4)]

            # -------------------------------------------- #
            '''
            Add blocks to main block list and save block layer
            '''

            # Add local blocks to main block list
            block_list += layer_block_list

            # Print blocks data in OpenFOAM format
            if self.WRITE==True:
                if layer in range(self.N_wing_layers):
                    if (self.AILERON==True and layer==self.layer_aileron):
                        list_of_commented_blocks = [layer*N_blck + i for i in range(3,N_interior)]
                    else:
                        list_of_commented_blocks = [layer * N_blck + i for i in range(N_interior)]
                else:
                    list_of_commented_blocks = []
                filename = self.outputfolder + "/list_blocks_layer" + str(layer)
                mf.write_blocks_to_file(filename, layer_block_list, layer, list_of_commented_blocks)

                # print message
                if self.PRINT==True:
                    if layer==0: print("Generate mesh layers:")
                    print("Layer "+"{:2d}".format(layer+1)+"/"+"{:2d}".format(self.N_layers)+": List of blocks written...")
                    if layer==(self.N_layers-1): print("")

        # -------------------------------------------- #
        '''
        Add blocks to mesh object
        '''
        self.blocks = block_list
        self.N_blocks = len(block_list)
        self.N_interior = N_interior
        self.N_exterior = N_exterior

    # ---------------------- Generate edges ---------------------- #
    '''
    Generate edges for each slice of points
    '''
    def generate_edges(self):

        # Retrieve mesh structure
        N_slices = self.N_slices
        N_vertices = self.N_vertices
        N_pts = int(N_vertices/N_slices)

        # Retrieve vertices and blocks
        vertex_list = self.vertices

        # Create edges
        edge_list = []
        n_pts_along_edge = 51 #51

        # Loop through slices
        for slice in range(N_slices):

            # Local list of vertices and edges
            slice_vertex_list = [vertex_list[i] for i in range(slice*N_pts, (slice+1)*N_pts)]
            slice_edge_list = []

            # -------------------------------------------- #
            '''
            Re-Compute existing splines along suction, camber a pressure lines
            '''

            # Define splines functions at pressure, suction, and camber lines at airfoil section
            s = af.AirfoilSection(self.airfoil_name, self.airfoil_file, self.wing_file, self.COG, y=self.slice_locations[slice])

            # Define splines functions at pressure, suction, and camber lines
            tck_p = s.scaled_data.tck_lower #splprep([coord[mLE:,0], coord[mLE:,1]], s=0)
            tck_s = s.scaled_data.tck_upper #splprep([np.flip(coord[mTE:mLE+1,0]), np.flip(coord[mTE:mLE+1,1])], s=0)
            tck_c = s.scaled_data.tck_camber #splprep([camber[:,0], camber[:,1]], s=0)
            # tck_p, _ = splprep([s.coord[s.mLE:,0], s.coord[s.mLE:,1]], s=0)
            # tck_s, _ = splprep([np.flip(s.coord[s.mTE:s.mLE+1,0]), np.flip(s.coord[s.mTE:s.mLE+1,1])], s=0)
            # tck_c, _ = splprep([s.camber[:,0], s.camber[:,1]], s=0)

            # -------------------------------------------- #
            '''
            Create edges of suction, camber, and pressure lines
            '''

            # MANUAL: Create edges of suction line
            spline_vertices_s = [0,1,4,7,10,13]
            edge_vertices = mf.get_edge_vertices(spline_vertices_s)
            edge_tck = [tck_s]*len(edge_vertices)
            edge_points = [n_pts_along_edge]*len(edge_vertices)
            k0 = len(edge_list) + len(slice_edge_list)
            for k, [vert, tck, pts] in enumerate(zip(edge_vertices, edge_tck, edge_points), start=k0):
                slice_edge_list.append(edge(k, [slice_vertex_list[i] for i in vert], tck, pts))

            # MANUAL: Create edges of camber line
            spline_vertices_s = [2,5,8,12,13]
            edge_vertices = mf.get_edge_vertices(spline_vertices_s)
            edge_tck = [tck_c]*len(edge_vertices)
            edge_points = [n_pts_along_edge]*len(edge_vertices)
            k0 = len(edge_list) + len(slice_edge_list)
            for k, [vert, tck, pts] in enumerate(zip(edge_vertices, edge_tck, edge_points), start=k0):
                slice_edge_list.append(edge(k, [slice_vertex_list[i] for i in vert], tck, pts))

            # MANUAL: Create edges of pressure line
            spline_vertices_s = [0,3,6,9,11,13]
            edge_vertices = mf.get_edge_vertices(spline_vertices_s)
            edge_tck = [tck_p]*len(edge_vertices)
            edge_points = [n_pts_along_edge]*len(edge_vertices)
            k0 = len(edge_list) + len(slice_edge_list)
            for k, [vert, tck, pts] in enumerate(zip(edge_vertices, edge_tck, edge_points), start=k0):
                slice_edge_list.append(edge(k, [slice_vertex_list[i] for i in vert], tck, pts))

            # -------------------------------------------- #
            '''
            Create new splines for top and bottom blocks
            '''

            # MANUAL: Create new splines for top, bottom, and left/front blocks
            spline_vertices_t = [13,20,18,16,0]
            spline_vertices_b = [13,21,19,17,0]
            tck_t, _ = splprep([[slice_vertex_list[k].coord[0] for k in spline_vertices_t], [slice_vertex_list[k].coord[2] for k in spline_vertices_t]], s=0)
            tck_b, _ = splprep([[slice_vertex_list[k].coord[0] for k in spline_vertices_b], [slice_vertex_list[k].coord[2] for k in spline_vertices_b]], s=0)

            # Populate "pos" and "line" along new splines
            slice_vertex_list = mf.populate_vertices(slice_vertex_list, spline_vertices_t[1:-1], "t", tck_t)
            slice_vertex_list = mf.populate_vertices(slice_vertex_list, spline_vertices_b[1:-1], "b", tck_b)

            # Plot new splines
            if self.PLOT:
                ax = plt.figure(slice + 1).axes
                mf.plot_spline(ax, tck_t, 'tab:cyan')
                mf.plot_spline(ax, tck_b, 'tab:cyan')

            # -------------------------------------------- #
            '''
            Create edges of airfoil interior (top/bottom blocks)
            '''

            # MANUAL: Top block spline edge
            edge_vertices = mf.get_edge_vertices(spline_vertices_t[1:-1])
            edge_tck = [tck_t]*len(edge_vertices)
            edge_points = [n_pts_along_edge]*len(edge_vertices)
            k0 = len(edge_list) + len(slice_edge_list)
            for k, [vert, tck, pts] in enumerate(zip(edge_vertices, edge_tck, edge_points), start=k0):
                slice_edge_list.append(edge(k, [slice_vertex_list[i] for i in vert], tck, pts))

            # MANUAL: Bottom block spline edge
            edge_vertices = mf.get_edge_vertices(spline_vertices_b[1:-1])
            edge_tck = [tck_b]*len(edge_vertices)
            edge_points = [n_pts_along_edge]*len(edge_vertices)
            k0 = len(edge_list) + len(slice_edge_list)
            for k, [vert, tck, pts] in enumerate(zip(edge_vertices, edge_tck, edge_points), start=k0):
                slice_edge_list.append(edge(k, [slice_vertex_list[i] for i in vert], tck, pts))

            # -------------------------------------------- #
            '''
            Create new spline and edges for mid-plane of exterior blocks
            '''

            # MANUAL: Create new spline for mid-plane of exterior blocks
            spline_vertices_m = [13,8,5,0,22,23] #[37, 16, ... exclude first  control points
            tck_m, _ = splprep([[slice_vertex_list[k].coord[0] for k in spline_vertices_m], [slice_vertex_list[k].coord[2] for k in spline_vertices_m]], s=0)

            # Plot new spline
            if self.PLOT:
                ax = plt.figure(slice + 1).axes
                mf.plot_spline(ax, tck_m, 'tab:purple')

            # MANUAL: Populate "pos" and "line" along new splines
            slice_vertex_list = mf.populate_vertices(slice_vertex_list, [13,0,22,23], "m", tck_m) #[37, 16, ... exclude first control point

            # MANUAL: Create edges
            edge_vertices = [[0,22],[22,23]] #[[31, 13], ... exclude first edge
            edge_tck = [tck_m]*len(edge_vertices)
            edge_points = [n_pts_along_edge]*len(edge_vertices)
            k0 = len(edge_list) + len(slice_edge_list)
            for k, [vert, tck, pts] in enumerate(zip(edge_vertices, edge_tck, edge_points), start=k0):
                slice_edge_list.append(edge(k, [slice_vertex_list[i] for i in vert], tck, pts))

            # -------------------------------------------- #
            '''
            Create edges of front arcs
            '''

            # MANUAL: Manually add arc edges
            for arc_pair in [[29,30],[30,31],[31,32],[32,33]]:
                theta = []
                for i in arc_pair:
                    x,y,z = slice_vertex_list[i].coord
                    value = np.arctan2(z,x)
                    if value < 0:
                        value = 2*np.pi + value
                    theta.append(value)
                mean_theta = sum(theta)/len(theta)
                slice_edge_list.append(edge(0, [slice_vertex_list[i] for i in arc_pair], None, None))
                slice_edge_list[-1].type = "arc"
                slice_edge_list[-1].coord = [self.mesh_scale*self.c0*np.cos(mean_theta), y, self.mesh_scale*self.c0*np.sin(mean_theta)]

            # -------------------------------------------- #
            '''
            Create edges for face matching
            '''

            # MANUAL: Manually add edges for face matching
            # Top blocks
            k0 = len(edge_list) + len(slice_edge_list)
            slice_edge_list.append(edge(k0+1, [slice_vertex_list[i] for i in [4,5]], None, None))
            coordinates = [v.coord for v in [slice_vertex_list[i] for i in [4, 16, 5]]]
            slice_edge_list[-1].coord = [[sublist[i] for sublist in coordinates] for i in range(len(coordinates[0]))]
            slice_edge_list.append(edge(k0+2, [slice_vertex_list[i] for i in [4,16]], None, None))
            coordinates = [v.coord for v in [slice_vertex_list[i] for i in [4, 16]]]
            slice_edge_list[-1].coord = [[sublist[i] for sublist in coordinates] for i in range(len(coordinates[0]))]
            slice_edge_list.append(edge(k0+3, [slice_vertex_list[i] for i in [16,5]], None, None))
            coordinates = [v.coord for v in [slice_vertex_list[i] for i in [16, 5]]]
            slice_edge_list[-1].coord = [[sublist[i] for sublist in coordinates] for i in range(len(coordinates[0]))]

            # Bottom blocks
            k0 = len(edge_list) + len(slice_edge_list)
            slice_edge_list.append(edge(k0+1, [slice_vertex_list[i] for i in [5,6]], None, None))
            coordinates = [v.coord for v in [slice_vertex_list[i] for i in [5,17,6]]]
            slice_edge_list[-1].coord = [[sublist[i] for sublist in coordinates] for i in range(len(coordinates[0]))]
            slice_edge_list.append(edge(k0+2, [slice_vertex_list[i] for i in [5,17]], None, None))
            coordinates = [v.coord for v in [slice_vertex_list[i] for i in [5,17]]]
            slice_edge_list[-1].coord = [[sublist[i] for sublist in coordinates] for i in range(len(coordinates[0]))]
            slice_edge_list.append(edge(k0+3, [slice_vertex_list[i] for i in [17,6]], None, None))
            coordinates = [v.coord for v in [slice_vertex_list[i] for i in [17,6]]]
            slice_edge_list[-1].coord = [[sublist[i] for sublist in coordinates] for i in range(len(coordinates[0]))]

            # -------------------------------------------- #
            '''
            Add edges to main edge list and save edges of slice
            '''

            # Add local edges to main edge list
            edge_list += slice_edge_list

            # Write to OpenFOAM format
            if self.WRITE==True:
                if slice in range(self.N_wing_layers + 1): # Wing section only (2D or 3D)
                    list_of_commented_edges = list(range(5, 9))         # Edges of camber line
                    list_of_commented_edges += list(range(14, 18))      # Edges of top/bottom lines
                    list_of_commented_edges += list(range(18, 20))      # Edges of TE tail
                    list_of_commented_edges += list(range(24, 30))      # Edges of interior interface
                else:
                    list_of_commented_edges = list(range(18, 20))      # Edges of TE tail
                filename = self.outputfolder + "/list_edges_slice" + str(slice)
                mf.write_edges_to_file(filename, slice_edge_list, slice, list_of_commented_edges)

                # print message
                if self.PRINT==True:
                    if slice==0: print("Generate mesh edges:")
                    print("Slice "+"{:2d}".format(slice+1)+"/"+"{:2d}".format(self.N_slices)+": List of edges written...")
                    if slice==(self.N_slices-1): print("")

        # -------------------------------------------- #
        '''
        Add edges to mesh object
        '''
        self.edges = edge_list
        self.N_edges = len(edge_list)

    # ---------------------- Generate interfaces for patch pairs merging ---------------------- #
    '''
    Generate interfaces for patch pairs merging
    '''
    def generate_interfaces(self):

        # Retrieve mesh structure
        block_list = self.blocks
        N_blck = self.N_interior + self.N_exterior

        # Only for flow layers
        if self.N_flow_layers != 0:

            # Get left interface faces
            left_list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in range(self.N_wing_layers, self.N_layers) for element in [3, 4, 5, 6]]]
            left_list_of_face_index = [0]*len(left_list_of_blocks)

            # Get right interface faces
            right_list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in range(self.N_wing_layers, self.N_layers) for element in [1, 2]]]
            right_list_of_face_index = [2]*len(right_list_of_blocks)

            if self.WRITE==True:

                # Write left interface to OpenFOAM format
                filename1 = self.outputfolder + "/list_boundaries_leftInterface"
                name, type = ["leftInterface", "patch"]
                mf.write_boundaries_to_file(filename1, name, type, left_list_of_blocks, left_list_of_face_index)

                # Write right interface to OpenFOAM format
                filename2 = self.outputfolder + "/list_boundaries_rightInterface"
                name, type = ["rightInterface", "patch"]
                mf.write_boundaries_to_file(filename2, name, type, right_list_of_blocks, right_list_of_face_index)

                # print message
                if self.PRINT == True:
                    print("Generate flow interfaces:")
                    print("Left flow interface: List of faces written...")
                    print("Right flow interface: List of faces written...")
                    print("")

    # ---------------------- Define boundary conditions ---------------------- #
    '''
    Define boundary conditions
    '''
    def generate_boundaries(self):

        # Retrieve mesh structure
        N_layers = self.N_layers
        N_wing_layers = self.N_wing_layers
        block_list = self.blocks
        N_blocks = self.N_blocks
        N_slices = self.N_slices
        N_blck = self.N_interior + self.N_exterior
        N_interior = self.N_interior
        N_exterior = self.N_exterior

        if self.WRITE==True:

            # Write inlet to OpenFOAM format
            list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in range(self.N_layers) for element in list(range(self.N_interior, N_blck))]]
            list_of_face_index = [3]*len(list_of_blocks)
            filename = self.outputfolder + "/list_boundaries_inlet"
            name, type = ["inlet", "patch"]
            mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
            if self.PRINT:
                print("Generate boundary conditions:")
                print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")

            # Write outlet to OpenFOAM format
            list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in range(self.N_layers) for element in [self.N_interior, N_blck-1]]]
            list_of_face_index = [2,0]*self.N_layers
            filename = self.outputfolder + "/list_boundaries_outlet"
            name, type = ["outlet", "patch"]
            mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
            if self.PRINT:
                print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")

            # Wing BCs and sides BCs without aileron
            if self.AILERON==False:

                # Write wing surface to OpenFOAM format
                if self.N_flow_layers == 0:
                    list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in range(self.N_wing_layers) for element in list(range(self.N_interior+2, N_blck-2))]]
                    list_of_face_index = [1]*len(list_of_blocks)
                elif self.N_flow_layers != 0:
                    list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in range(self.N_wing_layers-1) for element in list(range(self.N_interior+2, N_blck-2))]]
                    list_of_face_index = [1]*len(list_of_blocks)
                filename = self.outputfolder + "/list_boundaries_wing"
                name, type = ["wing", "wall"]
                mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                if self.PRINT:
                    print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")

                # Write wing tip to OpenFOAM format
                if self.N_flow_layers != 0:
                    # Tip wing surface
                    list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in [self.N_wing_layers-1] for element in list(range(self.N_interior+2, N_blck-2))]]
                    list_of_face_index = [1]*len(list_of_blocks)
                    # Tip side surface
                    list_of_blocks +=[block_list[i] for i in [element + N_blck * layer for layer in [self.N_wing_layers-1] for element in list(range(self.N_interior))]]
                    list_of_face_index += [5]*self.N_interior
                    filename = self.outputfolder + "/list_boundaries_wingTip"
                    name, type = ["wingTip", "wall"]
                    mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                    if self.PRINT:
                        print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")

                # Write sides (symmetries/frontAndBack) to OpenFOAM format
                if self.dim == "3D":
                    # Root symmetry
                    list_of_blocks = [block_list[i] for i in [element for element in list(range(self.N_interior, N_blck))]]
                    list_of_face_index = [4]*len(list_of_blocks)
                    filename = self.outputfolder + "/list_boundaries_symRoot"
                    name, type = ["symRoot", "symmetry"]
                    mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                    if self.PRINT:
                        print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")

                    # Tip symmetry
                    if self.N_flow_layers==0: # (Wing section only)
                        list_of_blocks = [block_list[i] for i in [element for element in list(range(self.N_interior, N_blck))]]
                        list_of_face_index = [5]*len(list_of_blocks)
                    elif self.N_flow_layers != 0: # (incl. flow domain)
                        list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in [self.N_layers-1] for element in list(range(N_blck))]]
                        list_of_face_index = [5]*N_exterior + [5]*N_interior
                    filename = self.outputfolder + "/list_boundaries_symTip"
                    name, type = ["symTip", "symmetry"]
                    mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                    if self.PRINT:
                        print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")
                        print("")

                # Write front and back faces to OpenFOAM format
                elif self.dim == "2D":
                    list_of_blocks = [block_list[i] for i in [element for element in list(range(self.N_interior, N_blck))]]
                    list_of_face_index = [4]*len(list_of_blocks)
                    list_of_blocks +=[block_list[i] for i in [element for element in list(range(self.N_interior, N_blck))]]
                    list_of_face_index +=[5]*len(list_of_blocks)
                    filename = self.outputfolder + "/list_boundaries_frontAndBack"
                    name, type = ["frontAndBack", "empty"]
                    mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                    if self.PRINT:
                        print("Boundary condition " + "{:>12s}".format(name) + ": List of faces written...")
                        print("")

            # Wing BCs and sides BCs WITH aileron
            elif self.AILERON==True:
                # Write wing surface to OpenFOAM format
                if self.N_flow_layers == 0:
                    list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in range(self.N_wing_layers) for element in list(range(self.N_interior+4, N_blck-4))]]
                    list_of_face_index = [1]*len(list_of_blocks)
                elif self.N_flow_layers != 0:
                    list_blocks_full_surface = list(range(self.N_interior + 2, N_blck - 2))
                    list_blocks_cut_surface = list(range(self.N_interior + 4, N_blck - 4))
                    list_layers_full_surface = list(range(self.N_wing_layers - 1))
                    list_layers_full_surface.remove(self.layer_aileron)
                    list_layers_cut_surface = [self.layer_aileron]
                    list_blocks_surface = [element + N_blck * layer for layer in list_layers_full_surface for element in list_blocks_full_surface]
                    list_blocks_surface+= [element + N_blck * layer for layer in list_layers_cut_surface for element in list_blocks_cut_surface]
                    list_of_blocks = [block_list[i] for i in list_blocks_surface]
                    list_of_face_index = [1]*len(list_of_blocks)
                filename = self.outputfolder + "/list_boundaries_wing"
                name, type = ["wing", "wall"]
                mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                if self.PRINT:
                    print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")

                # Write wing tip to OpenFOAM format
                if self.N_flow_layers != 0:
                    # Tip wing surface
                    list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in [self.N_wing_layers-1] for element in list(range(self.N_interior+2, N_blck-2))]]
                    list_of_face_index = [1]*len(list_of_blocks)
                    # Tip side surface
                    list_of_blocks +=[block_list[i] for i in [element + N_blck * layer for layer in [self.N_wing_layers-1] for element in list(range(self.N_interior))]]
                    list_of_face_index += [5]*self.N_interior
                    filename = self.outputfolder + "/list_boundaries_wingTip"
                    name, type = ["wingTip", "wall"]
                    mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                    if self.PRINT:
                        print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")

                # Write aileron cut-out to OpenFOAM format
                if self.N_flow_layers == 0:
                    list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in [self.layer_aileron] for element in [1,2]]]
                    list_of_face_index = [2]*len(list_of_blocks)
                elif self.N_flow_layers != 0:
                    list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in [self.layer_aileron] for element in [1,2]]]
                    list_of_face_index = [2]*2
                    list_of_blocks+= [block_list[i] for i in [element + N_blck * layer for layer in [self.layer_aileron] for element in [0,1,2]]]
                    list_of_face_index+= [5]*3
                    if self.layer_aileron != 0:
                        list_of_blocks+= [block_list[i] for i in [element + N_blck * layer for layer in [self.layer_aileron] for element in [0,1,2]]]
                        list_of_face_index+= [4]*3
                filename = self.outputfolder + "/list_boundaries_wingCut"
                name, type = ["wingCut", "wall"]
                mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                if self.PRINT:
                    print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")


                # Write sides (symmetries/frontAndBack) to OpenFOAM format
                if self.dim == "3D":
                    # Root symmetry
                    list_of_blocks = [block_list[i] for i in [element for element in list(range(self.N_interior, N_blck))]]
                    list_of_face_index = [4]*len(list_of_blocks)
                    if self.layer_aileron==0:
                        list_of_blocks+= [block_list[i] for i in [element + N_blck * layer for layer in [self.layer_aileron] for element in [0,1,2]]]
                        list_of_face_index+= [4]*3
                    filename = self.outputfolder + "/list_boundaries_symRoot"
                    name, type = ["symRoot", "symmetry"]
                    mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                    if self.PRINT:
                        print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")

                    # Tip symmetry
                    if self.N_flow_layers==0: # (Wing section only)
                        list_of_blocks = [block_list[i] for i in [element for element in list(range(self.N_interior, N_blck))]]
                        list_of_face_index = [5]*len(list_of_blocks)
                        list_of_blocks+= [block_list[i] for i in [element + N_blck * layer for layer in [self.layer_aileron] for element in [0,1,2]]]
                        list_of_face_index+= [5]*3
                    elif self.N_flow_layers != 0: # (incl. flow domain)
                        list_of_blocks = [block_list[i] for i in [element + N_blck * layer for layer in [self.N_layers-1] for element in list(range(N_blck))]]
                        list_of_face_index = [5]*N_exterior + [5]*N_interior
                    filename = self.outputfolder + "/list_boundaries_symTip"
                    name, type = ["symTip", "symmetry"]
                    mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                    if self.PRINT:
                        print("Boundary condition "+"{:>7s}".format(name)+": List of faces written...")
                        print("")

                # Write front and back faces to OpenFOAM format
                elif self.dim == "2D":
                    list_of_blocks = [block_list[i] for i in [element for element in list(range(self.N_interior, N_blck))]]
                    list_of_face_index = [4]*len(list_of_blocks)
                    list_of_blocks +=[block_list[i] for i in [element for element in list(range(self.N_interior, N_blck))]]
                    list_of_face_index +=[5]*len(list_of_blocks)
                    filename = self.outputfolder + "/list_boundaries_frontAndBack"
                    name, type = ["frontAndBack", "empty"]
                    mf.write_boundaries_to_file(filename, name, type, list_of_blocks, list_of_face_index)
                    if self.PRINT:
                        print("Boundary condition " + "{:>12s}".format(name) + ": List of faces written...")
                        print("")




# ---------------------- END ---------------------- #

# Wing surface in 2D ?
# if (self.dim=="2D"):
#     list_of_blocks = [block_list[i] for i in range(self.N_interior+2, N_blck-2)]
#     list_of_face_index = [1]*len(list_of_blocks)
# elif (self.dim=="3D"):
