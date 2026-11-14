import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from possession import paths  # noqa: E402

needs_match_data = pytest.mark.skipif(not (paths.DATA_DIR / "tactical").is_dir(),
                                      reason="the competition match data is not in data/tactical/ (or MTAH_DATA_DIR)")
