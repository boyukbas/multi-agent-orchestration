# Makes `import orchestrator` work in tests without `pip install -e .`.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
