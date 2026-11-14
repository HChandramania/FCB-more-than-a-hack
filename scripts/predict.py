"""Write the submission file for the competition match from the trained classifiers.

    python scripts/predict.py submission.csv

Frames outside the two graded halves are written as X; they are never scored.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from possession import data, pipeline  # noqa: E402


def main(path):
    match = data.load_competition_match()
    labels = np.where(data.graded_frames(len(match.labels)), pipeline.predict(match), data.X)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, data.LABELS[code]) for frame, code in enumerate(labels))
    print("wrote", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="path of the CSV to write")
    main(parser.parse_args().output)
