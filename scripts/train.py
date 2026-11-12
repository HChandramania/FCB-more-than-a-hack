"""Train the classifiers of the submission on the external matches and save them to the model folder.

    python scripts/train.py                 # every kind; the TCNs use the GPU if one is present
    python scripts/train.py --kinds tcn_dead team_ball_advance

LightGBM trains on the CPU. Each TCN takes about 15 s on an RTX 5080 and some minutes on a CPU.
GPU training is deterministic (tcn.py), so the same hardware and versions give the same weights.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from possession import gbm, paths, tcn  # noqa: E402

ALL_KINDS = gbm.KINDS + tcn.KINDS


def main(kinds):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for kind in kinds:
        start = time.time()
        if kind in gbm.KINDS:
            gbm.save(gbm.fit(kind), kind)
        else:
            tcn.save(tcn.train(kind, device=device, log=lambda line: None), kind)
        print("trained {:<24} {:>5.0f} s  -> {}".format(kind, time.time() - start, paths.MODEL_DIR), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kinds", nargs="+", choices=ALL_KINDS, default=list(ALL_KINDS))
    main(parser.parse_args().kinds)
