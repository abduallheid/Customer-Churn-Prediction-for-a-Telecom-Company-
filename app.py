"""Root entry point for Streamlit Community Cloud and local launch."""

import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import and execute the production UI app
from telecom_churn.ui.app import *
