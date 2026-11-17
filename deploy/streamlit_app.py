"""Entry point for the hosted demo on Streamlit Community Cloud (Person 5).

Community Cloud installs the packages listed next to the entry point
(deploy/requirements.txt) instead of the full project requirements. The screen
with sample findings only needs those few packages. When the app is connected
to the real rules and RAG pipeline, add their packages to deploy/requirements.txt.

Locally, keep using:  streamlit run app.py
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
