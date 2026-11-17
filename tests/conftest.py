import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from possession import paths  # noqa: E402

needs_match_data = pytest.mark.skipif(not (paths.COMPETITION_REPO / "data" / "tactical").is_dir(),
                                      reason="the competition match data is not present")
