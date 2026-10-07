"""Thin CLI wrapper around lloom_lab.data.build (kept for discoverability).

    python scripts/build_dataset.py [--seed 42] [--variants 10] [--out DIR]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.data.build import main  # noqa: E402

if __name__ == "__main__":
    main()
