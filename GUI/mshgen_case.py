'''
mshgen_case.py: command line for mshGen cases (the same functions the GUI uses)

Usage (Anaconda module loaded):
    python GUI/mshgen_case.py list
    python GUI/mshgen_case.py import <example.py> <case_name> [--overwrite]
    python GUI/mshgen_case.py validate <case>
    python GUI/mshgen_case.py write <case>
    python GUI/mshgen_case.py new <case_name> <airfoil>    (new wing: library airfoil, default planform)
    python GUI/mshgen_case.py blocks <case>                (block check: inverted/folded blocks)
    python GUI/mshgen_case.py build <case> [--images] [--force]   (write + block check + Allrun.pre)
    python GUI/mshgen_case.py check <case>                 (checkMesh summary)
<case> is a case name in mshGen/cases or a path to a case folder.
'''
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import paths  # noqa: E402
from core.case import Case  # noqa: E402


def case_dir(arg):
    folder = arg if os.path.isdir(arg) else os.path.join(paths.CASES_DIR, arg)
    if not os.path.isfile(os.path.join(folder, "case.json")):
        sys.exit("No case.json in " + folder)
    return os.path.abspath(folder)


def block_problems(folder):
    from core import blockcheck
    case = Case.load(folder)
    return blockcheck.check(case, os.path.join(folder, case.airfoil.file))


def print_summary(s):
    if s is None:
        print("No log.checkMesh (build the case first)")
        return
    print("cells {cells}  (hexahedra {hexahedra}, polyhedra {polyhedra})".format(**s))
    print("max aspect ratio {max_aspect_ratio}, max non-orthogonality {max_non_orthogonality}, "
          "max skewness {max_skewness}".format(**s))
    print("Mesh OK" if s["mesh_ok"] else "Failed checks: " + str(s["failed_checks"]))
    for w in s["warnings"]:
        print("  " + w)
    print("patches: " + ", ".join(k + " " + str(v) for k, v in s["patches"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("import")
    p.add_argument("example")
    p.add_argument("name")
    p.add_argument("--overwrite", action="store_true")
    for name in ("validate", "write", "check", "blocks"):
        sub.add_parser(name).add_argument("case")
    p = sub.add_parser("new")
    p.add_argument("name")
    p.add_argument("airfoil")
    p = sub.add_parser("build")
    p.add_argument("case")
    p.add_argument("--images", action="store_true")
    p.add_argument("--force", action="store_true", help="build even if the block check fails")
    a = ap.parse_args()

    if a.cmd == "list":
        for c in paths.list_cases():
            print(c)
    elif a.cmd == "import":
        from core.importer import case_from_example
        example = a.example if os.path.isfile(a.example) else os.path.join(paths.EXAMPLES_DIR, a.example)
        case = case_from_example(example, a.name, overwrite=a.overwrite)
        print("Created " + os.path.join(paths.CASES_DIR, case.name, "case.json"))
    elif a.cmd == "validate":
        problems = Case.load(case_dir(a.case)).validate()
        print("\n".join(problems) if problems else "OK")
        sys.exit(1 if problems else 0)
    elif a.cmd == "write":
        from core.writer import write_case
        folder = case_dir(a.case)
        for f in write_case(folder):
            print("Written " + os.path.relpath(f, folder))
    elif a.cmd == "new":
        from core.geometry import new_wing_case
        case = new_wing_case(a.name, a.airfoil)
        print("Created " + os.path.join(paths.CASES_DIR, case.name, "case.json"))
    elif a.cmd == "blocks":
        problems = block_problems(case_dir(a.case))
        print("\n".join(problems) if problems else "All blocks valid")
        sys.exit(1 if problems else 0)
    elif a.cmd == "build":
        from core.writer import write_case
        from core.runner import run_allrun, checkmesh_summary
        folder = case_dir(a.case)
        write_case(folder)
        problems = block_problems(folder)
        if problems and not a.force:
            sys.exit("Invalid blocks (blockMesh would fail; --force to build anyway):\n" + "\n".join(problems))
        code = run_allrun(folder, images=a.images)
        print_summary(checkmesh_summary(folder))
        sys.exit(code)
    elif a.cmd == "check":
        from core.runner import checkmesh_summary
        print_summary(checkmesh_summary(case_dir(a.case)))


if __name__ == "__main__":
    main()
