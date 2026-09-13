"""Shared asset location; independent of the developer's home directory."""
import os
from pathlib import Path

ASSET_ROOT = Path(os.environ.get('MICRODUCK_ROOT', Path(__file__).resolve().parents[3])).expanduser()
