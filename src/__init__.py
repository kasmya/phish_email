"""Phishing email detection package."""

import os
from pathlib import Path

# Keep plotting caches inside the project so matplotlib/fontconfig stay writable.
_BASE_DIR = Path(__file__).resolve().parent.parent
_CACHE_DIR = _BASE_DIR / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
