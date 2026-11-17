'''
paths.py: locations inside the mshGen repository, derived from this file (<mshGen>/GUI/core/paths.py)
'''
import os
import sys

GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSHGEN_DIR = os.path.dirname(GUI_DIR)
MESHGEN_DIR = os.path.join(MSHGEN_DIR, "meshGeneration")
CASES_DIR = os.path.join(MSHGEN_DIR, "cases")
EXAMPLES_DIR = os.path.join(MSHGEN_DIR, "examples")
AIRFOILS_DIR = os.path.join(MESHGEN_DIR, "airfoils")
WINGS_DIR = os.path.join(MESHGEN_DIR, "wings")
TEMPLATES_DIR = os.path.join(GUI_DIR, "templates")


def enable_mshgen_import():
    '''Make "import mshGen.meshGeneration..." work (the package name is the repository folder name).'''
    if os.path.basename(MSHGEN_DIR) != "mshGen":
        raise RuntimeError("The mshGen repository folder must be named 'mshGen' (package name), found: " + MSHGEN_DIR)
    parent = os.path.dirname(MSHGEN_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def list_cases(cases_dir=CASES_DIR):
    '''Case folders (containing case.json) in cases_dir, sorted by name.'''
    if not os.path.isdir(cases_dir):
        return []
    return sorted(d for d in os.listdir(cases_dir) if os.path.isfile(os.path.join(cases_dir, d, "case.json")))


def list_examples():
    '''Example scripts in mshGen/examples.'''
    return sorted(f for f in os.listdir(EXAMPLES_DIR) if f.startswith("generate_mesh_") and f.endswith(".py"))
