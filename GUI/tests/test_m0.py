'''
test_m0.py: M0 regression test of the case pipeline (no OpenFOAM needed)

For every example script in mshGen/examples:
  1. run the original example (in a temporary folder, on a temporary copy of default_files, so
     nothing in the repository is written) -> reference blockMesh lists
  2. import it as a case (case.json), write the case and run its generate_mesh.py -> case lists
  3. the lists must be identical, and the cell estimate must equal the sum over the block lists

Run (Anaconda module loaded):  python GUI/tests/test_m0.py [example ...]    or    pytest GUI/tests
'''
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from core import paths, estimate  # noqa: E402
from core.case import Case  # noqa: E402
from core.importer import case_from_example  # noqa: E402
from core.writer import write_case  # noqa: E402


def run_original(example, workdir):
    '''Run the example with foldername1 pointing to a copy of default_files; returns its output folder.'''
    defaults = os.path.join(workdir, "default_files")
    shutil.copytree(os.path.join(paths.MESHGEN_DIR, "default_files"), defaults)
    src = open(os.path.join(paths.EXAMPLES_DIR, example)).read()
    src, n = re.subn(r"^foldername1\s*=.*$", "foldername1 = " + repr(defaults), src, flags=re.M)
    assert n == 1, "foldername1 not found in " + example
    script = os.path.join(paths.EXAMPLES_DIR, "_test_copy_" + example)   # same folder: same path logic
    run = os.path.join(workdir, "original")
    os.makedirs(run)
    try:
        with open(script, "w") as f:
            f.write(src)
        subprocess.run([sys.executable, script], cwd=run, check=True, capture_output=True, text=True,
                       env=dict(os.environ, MPLBACKEND="Agg"))
    finally:
        os.remove(script)
    return os.path.join(run, "output_files")


def run_case(example, workdir):
    cases = os.path.join(workdir, "cases")
    case_from_example(os.path.join(paths.EXAMPLES_DIR, example), "test_case", cases_dir=cases)
    folder = os.path.join(cases, "test_case")
    write_case(folder)
    subprocess.run([sys.executable, "generate_mesh.py"], cwd=folder, check=True, capture_output=True, text=True,
                   env=dict(os.environ, MPLBACKEND="Agg"))
    return folder


def block_cells(output_files):
    '''Cells of all active (not commented) blocks in the list_blocks_layer* files.'''
    total = 0
    for f in os.listdir(output_files):
        if f.startswith("list_blocks_layer"):
            for line in open(os.path.join(output_files, f)):
                m = re.match(r"\s*hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", line)
                if m:
                    total += int(m.group(1)) * int(m.group(2)) * int(m.group(3))
    return total


def check_example(example):
    work = tempfile.mkdtemp(prefix="mshgen_test_")
    try:
        ref = run_original(example, work)
        folder = run_case(example, work)
        out = os.path.join(folder, "output_files")
        cmp = filecmp.dircmp(ref, out)
        assert not cmp.left_only and not cmp.right_only, (cmp.left_only, cmp.right_only)
        _, mismatch, errors = filecmp.cmpfiles(ref, out, sorted(os.listdir(ref)), shallow=False)
        assert not mismatch and not errors, "different lists: " + str(mismatch + errors)
        assert os.path.isfile(os.path.join(folder, "system", "blockMeshDict"))
        est, real = estimate.cell_count(Case.load(folder)), block_cells(out)
        assert est == real, "cell estimate {} != {} from the block lists".format(est, real)
        return len(os.listdir(ref)), real
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_all_examples():
    for example in paths.list_examples():
        check_example(example)


if __name__ == "__main__":
    examples = sys.argv[1:] or paths.list_examples()
    failed = 0
    for ex in examples:
        try:
            n, cells = check_example(ex)
            print("OK    {:55s} {:3d} lists identical, {:>10,} cells".format(ex, n, cells))
        except Exception as e:  # noqa: BLE001
            failed += 1
            msg = e.stderr[-500:] if isinstance(e, subprocess.CalledProcessError) else repr(e)[:500]
            print("FAIL  {:55s} {}".format(ex, msg))
    sys.exit(1 if failed else 0)
