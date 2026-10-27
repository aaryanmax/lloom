"""E4 — Generalization Gap / Uniform Convergence driver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.experiments.e4 import main

if __name__ == "__main__":
    main()
