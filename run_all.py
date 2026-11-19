"""Repo-root entry point: `python run_all.py --seed 0` runs the main experiment (src/run_all.py).
Resumable: every generation batch is cached in results/gen/. Add --budget-seconds N to run in slices."""
import pathlib, runpy, sys
SRC = pathlib.Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
sys.argv[0] = str(SRC / "run_all.py")
runpy.run_path(str(SRC / "run_all.py"), run_name="__main__")
