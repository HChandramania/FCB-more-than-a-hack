"""Checks on the competition match with the shipped models: team symmetry and the exact submission file."""
import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from conftest import needs_match_data
from possession import data, pipeline

ROOT = Path(__file__).resolve().parent.parent
# SHA-256 of the submitted file; scripts/predict.py must write exactly this file (RULES.md rule 3).
SUBMISSION_SHA256 = "2467a1396f08f2e0f4f65bb8ae261da82f44cae9678b8547dcbe4d2b17ac6f8e"


def exchanged(match):
    """The match with the roles of the teams exchanged."""
    slot_team = np.where(match.slot_team == data.H, data.A, data.H).astype(match.slot_team.dtype)
    return dataclasses.replace(match, slot_team=slot_team)


@needs_match_data
def test_probabilities_are_team_symmetric():
    match = data.only_halves(data.load_competition_match())
    other = exchanged(match)
    assert np.allclose(pipeline.home_probability(other), 1.0 - pipeline.home_probability(match), atol=1e-5)
    assert np.allclose(pipeline.dead_probability(other), pipeline.dead_probability(match), atol=1e-5)


@needs_match_data
def test_predict_writes_the_submitted_file(tmp_path):
    output = tmp_path / "submission.csv"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "predict.py"), str(output)], check=True)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == SUBMISSION_SHA256
