"""E5: No-Free-Lunch (NFL) Theorem Empirical Demonstration driver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.experiments.e5 import main

if __name__ == "__main__":
    main()
