import runpy
from pathlib import Path

target = Path(__file__).parent / "src" / "lloom_lab" / "dashboard" / "app.py"
runpy.run_path(str(target), run_name="__main__")
