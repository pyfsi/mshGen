'''
GUI/core: logic of the mshGen GUI, without any GUI dependency.

Used by the Panel app (GUI/app.py), the command line (GUI/mshgen_case.py) and the tests (GUI/tests).
'''
# mshGen modules import matplotlib.pyplot; the GUI and the command line never plot interactively.
import matplotlib
matplotlib.use("Agg")
