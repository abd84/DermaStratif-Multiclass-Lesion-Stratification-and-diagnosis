import sys
import os

# Ensure repo root is on the path so Application package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Application.app import app

__all__ = ["app"]
