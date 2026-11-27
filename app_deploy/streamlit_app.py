"""Entry point for the hosted demo on Streamlit Community Cloud (Person 5).

Community Cloud uses the full requirements.txt at the repository root.
This wrapper runs the same app.py used locally.

Locally, keep using:  streamlit run app.py
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
