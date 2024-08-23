'''
Default values for different C-meshes
'''
import numpy as np

def get_default_edge_parameters(airfoil_name, mesh_scale, mesh_level):

    #------------------------------------------------#
    # Find closest default values
    if (airfoil_name == "NACA0012"):
        default_scale = np.array([20, 10, 5, 3, 1.5, 1])
        idx = np.argmin(np.abs(default_scale - (mesh_scale+1e-6)))
        mesh_scale = default_scale[idx]
    elif (airfoil_name == "Mrev-v2"):
        default_scale = np.array([10, 5, 3])
        idx = np.argmin(np.abs(default_scale - (mesh_scale+1e-6)))
        mesh_scale = default_scale[idx]
    elif (airfoil_name == "FFA_W3_211"):
        default_scale = np.array([10])
        idx = np.argmin(np.abs(default_scale - (mesh_scale+1e-6)))
        mesh_scale = default_scale[idx]
    else:
        print("Unknown airfoil: Implementation necessary...")

    #------------------------------------------------#
    # Header message
    msg = "Mesh with default values from "
    msg += "C-mesh of " + airfoil_name + " airfoil with "
    msg += "R/c=" + "{:06.2f}".format(mesh_scale) + " and "
    msg += mesh_level + " refinement level."
    print(msg)

    #------------------------------------------------#
    # NACA0012 airfoil
    if (airfoil_name == "NACA0012"):

        # C-mesh R=20c
        if (mesh_scale == 20):

            # Fine level
            if (mesh_level == "FINE"):

                # C-mesh with 020c (gamma_c0 = 7.5, A = 3000, B = 41)
                n0 = 512
                g0 = 4.1e4
                g9 = g0/3e3
                g10= g0/41

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [128, 256]  + [96, 32, 40, 64, 64]       + [48]  + [n0] + [n0]
                edge_grading1 =   [g0] + [200.,55.] + [30., 2.5, 1.1, 0.125, 0.1] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [12., 5.]  + [1., 2., 1., 6., 4.]        + [1.] + [g9] + [g10]

            # Medium level
            elif (mesh_level == "MEDIUM"):

                # C-mesh with 020c
                n0 = 160
                g0 = 2000
                g9 = g0/200.
                g10= g0/2.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [64, 64]   + [24, 26, 32, 56, 24]   + [12]  + [n0] + [n0]
                edge_grading1 =   [g0] + [250., 3.] + [1., 2.6, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [10.,  1.] + [0.25, 1., 4., 3., 10.] + [1.] + [g9] + [g10]

            # Coarse level
            elif (mesh_level == "COARSE"):

                # C-mesh with 020c
                n0 = 64
                g0 = 360
                g9 = g0/160.
                g10= g0/2.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [48, 20] + [8, 12, 16, 28, 12]  + [4]  + [n0] + [n0]
                edge_grading1 =   [g0] + [90., 3.5] + [1., 1.5, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [2.2, 1.] + [0.3, 1., 3, 5., 5.]  + [1.] + [g9] + [g10]

        # C-mesh R=10c
        elif (mesh_scale == 10):

            # Fine level
            if (mesh_level == "FINE"):

                # C-mesh with 010c (gamma_c0 = 4.0, A = 2850, B = 65)
                n0 = 500
                g0 = 2.0e4
                g9 = g0/2850.
                g10= g0/65.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [128, 256]  + [96, 32, 40, 64, 64]       + [48]  + [n0] + [n0]
                edge_grading1 =   [g0] + [75., 70.] + [30., 2.5, 1.1, 0.125, 0.1] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [12., 5.]  + [1., 4., 1., 6., 4.]        + [1.] + [g9] + [g10]

            # Medium level
            elif (mesh_level == "MEDIUM"):

                # C-mesh with 010c
                n0 = 140
                g0 = 1000
                g9 = g0/150.
                g10= g0/2.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [48, 64]   + [24, 26, 32, 56, 24]   + [12]  + [n0] + [n0]
                edge_grading1 =   [g0] + [150., 3.] + [1., 2.6, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [10.,  1.] + [0.25, 1., 4., 2., 15.] + [1.] + [g9] + [g10]

            # Coarse level
            elif (mesh_level == "COARSE"):

                # C-mesh with 010c
                n0 = 56
                g0 = 190
                g9 = g0/10.
                g10= g0/2.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [32, 20] + [8, 12, 16, 28, 12]  + [4]  + [n0] + [n0]
                edge_grading1 =   [g0] + [50., 3.5] + [1., 1.5, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [3., 1.] + [0.3, 1., 2, 5., 5.]  + [1.] + [g9] + [g10]

        # C-mesh R=5c
        elif (mesh_scale == 5):

            # Fine level
            if (mesh_level == "FINE"):

                # C-mesh with 005c (gamma_c0 = 2.3, A = 1000, B = 50)
                n0 = 465
                g0 = 9.8e3
                g9 = g0/1000.
                g10= g0/50.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [64, 256]  + [96, 32, 40, 64, 64]        + [48] + [n0] + [n0]
                edge_grading1 =   [g0] + [70., 70.] + [30., 2.5, 1.1, 0.125, 0.1] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [15., 3.]  + [1., 4., 1., 5., 4.]        + [1.] + [g9] + [g10]

            # Medium level
            elif (mesh_level == "MEDIUM"):

                # C-mesh with 005c (gamma_c0 = 2.3, A = 70, B = 5)
                n0 = 128
                g0 = 475
                g9 = g0/70.
                g10= g0/5.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [32, 64]   + [24, 26, 32, 56, 24]   + [12] + [n0] + [n0]
                edge_grading1 =   [g0] + [100., 3.] + [1., 2.6, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [10.,  2.] + [0.5, 1., 4., 1., 10.] + [1.] + [g9] + [g10]

            # Coarse level
            elif (mesh_level == "COARSE"):

                # C-mesh with 005c (gamma_c0 = 2.3, A = 1000, B = 50)
                n0 = 48
                g0 = 90
                g9 = g0/15.
                g10= g0/1.5

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [32, 20] + [8, 12, 16, 28, 12]  + [4]  + [n0] + [n0]
                edge_grading1 =   [g0] + [22., 3.5] + [1., 1.5, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [2.1, 1.] + [0.3, 1., 2.5, 3., 4.]  + [1.] + [g9] + [g10]

        # C-mesh R=3c
        elif (mesh_scale == 3):

            # Fine level
            if (mesh_level == "FINE"):

                # C-mesh with 003c
                n0 = 440
                g0 = 6e3
                g9 = g0/5000.
                g10= g0/80.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [32, 256]  + [96, 32, 40, 64, 64]        + [48]  + [n0] + [n0]
                edge_grading1 =   [g0] + [76., 70.] + [30., 2.5, 1.1, 0.125, 0.1] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [25. , 3.]  + [1., 4., 1., 2.5, 2.]        + [1.] + [g9] + [g10]

            # Medium level
            elif (mesh_level == "MEDIUM"):

                # C-mesh with 003c
                n0 = 116
                g0 = 300
                g9 = g0/40.
                g10= g0/2.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [24, 64]   + [24, 26, 32, 56, 24]   + [12]  + [n0] + [n0]
                edge_grading1 =   [g0] + [60., 3.] + [1., 2.6, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [10.,  1.] + [0.5, 1., 3., 1., 10.] + [1.] + [g9] + [g10]

            # Coarse level
            elif (mesh_level == "COARSE"):

                # C-mesh with 003c
                n0 = 44
                g0 = 56
                g9 = g0/2.5
                g10= g0/1.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [16, 20] + [8, 12, 16, 28, 12]  + [4]  + [n0] + [n0]
                edge_grading1 =   [g0] + [22., 3.5] + [1., 1.5, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [5., 1.] + [0.3, 1., 3., 1., 10.]  + [1.] + [g9] + [g10]

        # C-mesh R=3c
        elif (mesh_scale == 1.5):

            # Fine level
            if (mesh_level == "FINE"):

                # C-mesh with 001.5c
                n0 = 400
                g0 = 3000
                g9 = g0/2000.
                g10= g0/86.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [24, 256]  + [96, 32, 40, 64, 64]        + [48]  + [n0] + [n0]
                edge_grading1 =   [g0] + [32., 70.] + [30., 2.5, 1.1, 0.125, 0.1] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [90. , 1.]  + [1., 5., 1., 1., 2.]        + [1.] + [g9] + [g10]

            # Medium level
            elif (mesh_level == "MEDIUM"):

                # C-mesh with 001.5c
                n0 = 102
                g0 = 150
                g9 = g0/15.
                g10= g0/2.5

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [16, 64]   + [24, 26, 32, 56, 24]   + [12]  + [n0] + [n0]
                edge_grading1 =   [g0] + [25., 3.] + [1., 2.6, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [50.,  1.] + [1., 1., 2., 0.7, 12.] + [1.] + [g9] + [g10]

            # Coarse level
            elif (mesh_level == "COARSE"):

                # C-mesh with 001.5c
                n0 = 36
                g0 = 30
                g9 = g0/3.
                g10= g0/1.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [16, 20] + [8, 12, 16, 28, 12]  + [4]  + [n0] + [n0]
                edge_grading1 =   [g0] + [4.5, 3.5] + [1., 1.5, 1., 0.1, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [9., 1.] + [1., 1., 3., 0.5, 15.]  + [1.] + [g9] + [g10]

        # C-mesh R=1c
        elif (mesh_scale == 1):

            # Coarse level
            if (mesh_level == "COARSE"):

                # C-mesh with 001c
                n0 = 36 #50
                g0 = 200 #250
                g9 = g0/50.
                g10= g0/10.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [12, 24] + [16, 12, 12, 16, 8]  + [8]  + [n0] + [n0]
                edge_grading1 =   [g0] + [1, 8.] + [1., 4., 1., 0.15, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [2., 2.] + [1., 4., 1., 2., 2.]  + [1.] + [g9] + [g10]

    # ------------------------------------------------#
    # TODO: MegAWES Mrev-v2 airfoil
    elif (airfoil_name == "Mrev-v2"):

        # C-mesh R=5c
        if (mesh_scale == 10):

            # Fine level
            if (mesh_level == "FINE"):

                # C-mesh with 010c
                n0 = 500
                g0 = 2.0e4
                g9 = g0/1000.
                g10= g0/10. #65.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [64, 256]  + [96, 32, 40, 32, 64]        + [48] + [n0] + [n0]
                edge_grading1 =   [g0] + [175., 70.] + [30., 2.5, 1.1, 0.33, 0.1] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [130., 1.]  + [2., 5., 1.5, 2., 0.5]        + [1.] + [g9] + [g10]

            # Medium level
            elif (mesh_level == "MEDIUM"):

                # C-mesh with 010c
                n0 = 140
                g0 = 1000
                g9 = g0/150.
                g10= g0/2.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [48, 64]   + [24, 26, 32, 32, 24]   + [12]  + [n0] + [n0]
                edge_grading1 =   [g0] + [150., 3.] + [1., 2.6, 1., 0.33, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [32., 1.] + [2., 1., 4., 1., 2.] + [1.] + [g9] + [g10]

            # Coarse level
            elif (mesh_level == "COARSE"):

                # C-mesh with 010c
                n0 = 56
                g0 = 190
                g9 = g0/10.
                g10= g0/2.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [32, 20] + [8, 10, 16, 14, 12]  + [4]  + [n0] + [n0]
                edge_grading1 =   [g0] + [55., 3] + [1., 2., 1., 0.3, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [11., 1.] + [1., 1., 3., 1.5, 1.]  + [1.] + [g9] + [g10]

        # C-mesh R=5c
        elif (mesh_scale == 5):

            # Fine level
            if (mesh_level == "FINE"):

                # C-mesh with 005c (gamma_c0 = 2.3, A = 1000, B = 50)
                n0 = 465
                g0 = 9.8e3
                g9 = g0/1000.
                g10= g0/50.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [64, 256]  + [96, 32, 40, 32, 64]        + [48] + [n0] + [n0]
                edge_grading1 =   [g0] + [70., 70.] + [30., 2.5, 1.1, 0.33, 0.1] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [80., 1.]  + [2., 5., 1.5, 2., 0.5]        + [1.] + [g9] + [g10]

            # Medium level
            elif (mesh_level == "MEDIUM"):

                # C-mesh with 005c (gamma_c0 = 2.3, A = 70, B = 5)
                n0 = 128
                g0 = 475
                g9 = g0/70.
                g10= g0/5.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [32, 64]   + [24, 26, 32, 32, 24]   + [12]  + [n0] + [n0]
                edge_grading1 =   [g0] + [100., 3.] + [1., 2.6, 1., 0.33, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [33., 1.] + [2., 1., 4., 1., 2.] + [1.] + [g9] + [g10]

            # Coarse level
            elif (mesh_level == "COARSE"):

                # C-mesh with 005c (gamma_c0 = 2.3, A = 1000, B = 50)
                n0 = 48
                g0 = 90
                g9 = g0/15.
                g10= g0/1.5

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [32, 18] + [8, 10, 16, 14, 12]  + [4]  + [n0] + [n0]
                edge_grading1 =   [g0] + [22., 2.5] + [1., 2., 1., 0.3, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [5., 1.] + [1., 1., 2., 2., 1.]  + [1.] + [g9] + [g10]

        # C-mesh R=3c
        elif (mesh_scale == 3):

            # Fine level
            if (mesh_level == "FINE"):

                # C-mesh with 003c
                n0 = 440
                g0 = 6e3
                g9 = g0/5000.
                g10= g0/80.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [48, 256]  + [96, 32, 40, 32, 64]        + [48] + [n0] + [n0]
                edge_grading1 =   [g0] + [45., 70.] + [30., 2.5, 1.1, 0.33, 0.1] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [65., 1.]  + [2., 5., 1.5, 2., 0.5]        + [1.] + [g9] + [g10]


            # Medium level
            elif (mesh_level == "MEDIUM"):

                # C-mesh with 003c
                n0 = 116
                g0 = 300
                g9 = g0/50.
                g10= g0/2.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [32, 64]   + [24, 26, 32, 32, 24]   + [12]  + [n0] + [n0]
                edge_grading1 =   [g0] + [42., 3.] + [1., 2.6, 1., 0.33, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [14., 1.5] + [2., 1., 3., 1., 2.] + [1.] + [g9] + [g10]

            # Coarse level
            elif (mesh_level == "COARSE"):

                # C-mesh with 003c
                n0 = 44
                g0 = 56
                g9 = g0/2.5
                g10= g0/1.

                # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
                edge_resolution = [n0] + [24, 18] + [8, 10, 16, 14, 12]  + [4]  + [n0] + [n0]
                edge_grading1 =   [g0] + [12., 4] + [1., 2., 1., 0.3, 1.] + [1.] + [g9] + [g10]
                edge_grading2 =   [g0] + [4., 1.] + [1., 1., 3., 1., 1.25]  + [1.] + [g9] + [g10]

    # ------------------------------------------------#
    # FFA W3 211 airfoil and other (only R=10c)
    else:

        # Fine level
        if (mesh_level == "FINE"):

            # C-mesh with 010c
            n0 = 128
            g0 = 2e5  # 25000
            g9 = g0 / 4000.
            g10 = g0 / 32.  # 65.

            edge_resolution = [n0] + [48, 64] + [72, 26, 32, 32, 24] + [12] + [n0] + [n0]
            edge_grading1 = [g0] + [50., 200.] + [24., 2.6, 1., 0.33, 1.] + [1.] + [g9] + [g10]
            edge_grading2 = [g0] + [3., 10.] + [1., 5., 1., 6., 0.4] + [1.] + [g9] + [g10]

        # Medium level
        elif (mesh_level == "MEDIUM"):

            # C-mesh with 010c
            n0 = 128
            g0 = 120
            g9 = 30.    # Domain end
            g10 = 60.   # Refinement zone

            edge_resolution = [n0] + [48, 64] + [36, 26, 32, 32, 24] + [12] + [n0] + [n0]
            edge_grading1 = [g0] + [100., 10.] + [2., 2.6, 1., 0.33, 1.] + [1.] + [g9] + [g10]
            edge_grading2 = [g0] + [5., 4.] + [1., 2., 1., 6., 0.4] + [1.] + [g9] + [g10]

            # C-mesh with 010c
            n0 = 108
            g0 = 66
            g9 = 20.    # Domain end
            g10 = 40.   # Refinement zone

            edge_resolution = [n0] + [32, 48] + [24, 18, 24, 24, 18] + [12] + [n0] + [n0]
            edge_grading1 = [g0] + [100., 10.] + [2., 2.6, 1., 0.33, 1.] + [1.] + [g9] + [g10]
            edge_grading2 = [g0] + [5., 4.] + [1., 2., 1., 6., 0.4] + [1.] + [g9] + [g10]

            # C-mesh with 010c
            n0 = 98
            g0 = 40
            g9 = 10.    # Domain end
            g10 = 20.   # Refinement zone
            edge_resolution = [n0] + [32, 24] + [12, 10, 16, 18, 12] + [12] + [n0] + [n0]
            edge_grading1 = [g0] + [50., 8.] + [2., 2., 1., 0.33, 1.] + [1.] + [g9] + [g10]
            edge_grading2 = [g0] + [3., 4.] + [1., 1., 1., 6., 0.4] + [1.] + [g9] + [g10]
        # Coarse level
        elif (mesh_level == "COARSE"):

            # C-mesh with 010c
            n0 = 56
            g0 = 190
            g9 = g0/10.
            g10= g0/2.

            # n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10
            edge_resolution = [n0] + [32, 20] + [8, 10, 16, 14, 12]  + [4]  + [n0] + [n0]
            edge_grading1 =   [g0] + [55., 3] + [1., 2., 1., 0.3, 1.] + [1.] + [g9] + [g10]
            edge_grading2 =   [g0] + [2., 2.] + [1., 1., 2., 3., 1.]  + [1.] + [g9] + [g10]

    return edge_resolution, edge_grading1, edge_grading2
