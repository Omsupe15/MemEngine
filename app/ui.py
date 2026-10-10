"""Streamlit UI module forwarder."""

import sys
from pathlib import Path

# Add root directory to sys.path if not present
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit_app  # noqa: F401
