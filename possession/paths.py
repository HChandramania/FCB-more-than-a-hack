"""Locations of the inputs, the caches and the trained models. An environment variable overrides each default."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The match bundle unpacked as data/tactical/ and data/broadcast/ (see docs/dataset.md).
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data"))
# External matches: external/dfl (Sportec Open DFL dataset) and external/metrica (Metrica sample data).
EXTERNAL_DIR = Path(os.environ.get("MTAH_EXTERNAL_DIR", ROOT / "external"))
# Parsed matches and training tables; everything in it can be deleted and is rebuilt on demand.
CACHE_DIR = Path(os.environ.get("MTAH_CACHE_DIR", ROOT / "cache"))
# Trained classifiers of the submission.
MODEL_DIR = Path(os.environ.get("MTAH_MODEL_DIR", ROOT / "models"))
