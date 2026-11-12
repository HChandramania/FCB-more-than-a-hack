"""Locations of the inputs, the caches and the trained models. An environment variable overrides each default."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The organisers' repository (github.com/metrica-sports/more-than-a-hack-2027) with the match data in data/.
COMPETITION_REPO = Path(os.environ.get("MTAH_COMPETITION_REPO", ROOT.parent / "more-than-a-hack-2027"))
# External matches: external/dfl (Sportec Open DFL dataset) and external/metrica (Metrica sample data).
EXTERNAL_DIR = Path(os.environ.get("MTAH_EXTERNAL_DIR", ROOT / "external"))
# Parsed matches and training tables; everything in it can be deleted and is rebuilt on demand.
CACHE_DIR = Path(os.environ.get("MTAH_CACHE_DIR", ROOT / "cache"))
# Trained classifiers of the submission.
MODEL_DIR = Path(os.environ.get("MTAH_MODEL_DIR", ROOT / "models"))


def competition_modules():
    """Put the organisers' scorer and game table on the import path, once."""
    for folder in (COMPETITION_REPO, COMPETITION_REPO / "evaluation"):
        if not (folder / ("games.py" if folder == COMPETITION_REPO else "possession_metrics.py")).is_file():
            raise FileNotFoundError("{} is not the organisers' repository; set MTAH_COMPETITION_REPO".format(COMPETITION_REPO))
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
