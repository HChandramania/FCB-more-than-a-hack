"""Synthetic-only verification of evaluation infrastructure; no real matches or models."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_external as evaluation
from possession import data, pipeline


def synthetic_comparison():
    reference = np.r_[data.H, np.full(20, data.X), data.A].astype(np.int8)
    dead = np.zeros(22)
    home = np.r_[0.6, np.full(20, 0.5), 0.001]
    return dead, home, reference, ((0, 1), (21, 22))


def test_paired_paths_use_identical_probability_memory_and_penalties(monkeypatch):
    dead, home, reference, halves = synthetic_comparison()
    original = evaluation.decoding.possession_states
    calls = []

    def inspect(d, h, team, stop):
        assert np.shares_memory(d, dead) and np.shares_memory(h, home)
        assert not d.flags.writeable and not h.flags.writeable
        assert team == pipeline.TEAM_SWITCH_COST and stop == pipeline.DEAD_SWITCH_COST
        calls.append((d.copy(), h.copy()))
        return original(d, h, team, stop)

    monkeypatch.setattr(evaluation.decoding, "possession_states", inspect)
    before_dead, before_home = dead.copy(), home.copy()
    result = evaluation.compare_paths(dead, home, reference, halves)
    assert len(calls) == 3
    for index, (start, stop) in enumerate(halves, 1):
        np.testing.assert_array_equal(calls[index][0], calls[0][0][start:stop])
        np.testing.assert_array_equal(calls[index][1], calls[0][1][start:stop])
    np.testing.assert_array_equal(dead, before_dead)
    np.testing.assert_array_equal(home, before_home)
    assert result["decoded_labels_differ"]
    assert result["changed_half_frames"] == result["changed_scored_frames"] == 1


def test_official_scorer_results_and_x_gaps(monkeypatch):
    dead, home, reference, halves = synthetic_comparison()
    official = data.metrics.evaluate
    calls = []

    def inspect(ref, pred, bootstrap_n):
        assert ref is reference and bootstrap_n == 0
        assert np.all(pred[1:21] == data.X)
        actual = official(ref, pred, bootstrap_n=bootstrap_n)
        calls.append(actual)
        return actual

    monkeypatch.setattr(data.metrics, "evaluate", inspect)
    result = evaluation.compare_paths(dead, home, reference, halves)
    for index, arm in enumerate(("original", "patched")):
        assert result[arm] == {"spellF1": calls[index]["spell_f1_mean"], "macroF1": calls[index]["macro_f1"]}
    assert result["patched"]["spellF1"] == 1.0
    for metric in evaluation.METRICS:
        assert result["delta"][metric] == result["patched"][metric] - result["original"][metric]


def record(name, dataset, value, mode="DIAGNOSTIC"):
    return {**evaluation.mode_metadata(mode, name if mode == "OUT-OF-FOLD" else None),
            "match": name, "dataset": dataset,
            "original": {"spellF1": value, "macroF1": value},
            "patched": {"spellF1": value + 0.1, "macroF1": value + 0.1},
            "delta": {"spellF1": 0.1, "macroF1": 0.1}}


def test_aggregation_does_not_weight_datasets_by_match_count():
    result = evaluation.aggregate([record("a", "dfl", 0.2), record("b", "dfl", 0.4),
                                   record("c", "metrica", 0.8)])
    assert result["per_dataset_match_means"]["dfl"]["original"]["spellF1"] == pytest.approx(0.3)
    assert result["equal_dataset_means"]["original"]["spellF1"] == pytest.approx(0.55)
    assert result["equal_dataset_means"]["delta"]["macroF1"] == pytest.approx(0.1)


@pytest.mark.parametrize("mode,held_out", [("DIAGNOSTIC", "a"), ("OUT-OF-FOLD", None), ("heldout", "a")])
def test_modes_cannot_silently_fall_back(mode, held_out):
    with pytest.raises(ValueError):
        evaluation.mode_metadata(mode, held_out)


def test_diagnostic_scores_cannot_be_relabeled_as_held_out():
    diagnostic = record("a", "dfl", 0.2)
    diagnostic["mode"] = "OUT-OF-FOLD"
    with pytest.raises(ValueError):
        evaluation.aggregate([diagnostic])
    diagnostic["held_out"] = "a"
    with pytest.raises(ValueError):
        evaluation.aggregate([diagnostic])


def test_aggregation_rejects_mixed_modes_and_duplicate_matches():
    with pytest.raises(ValueError):
        evaluation.aggregate([record("a", "dfl", 0.2), record("b", "dfl", 0.3, "OUT-OF-FOLD")])
    with pytest.raises(ValueError):
        evaluation.aggregate([record("a", "dfl", 0.2), record("a", "dfl", 0.2)])


def cache_metadata():
    return {"mode": "DIAGNOSTIC", "held_out": None, "n_frames": 2,
            "model_hashes": {"synthetic": "abc"}, "config": {"cost": 2.5},
            "scorer_sha256": "scorer", "reference_hash": "reference", "external_hashes": {"fixture": "data"}}


def test_probability_cache_reuses_exact_metadata_only(tmp_path):
    calls = []
    def generate():
        calls.append(True)
        return np.array([0.1, 0.2]), np.array([0.8, 0.7])
    metadata = cache_metadata()
    a = evaluation.probability_cache(tmp_path, metadata, generate)
    b = evaluation.probability_cache(tmp_path, metadata, generate)
    assert len(calls) == 1 and not a[2] and b[2]
    np.testing.assert_array_equal(a[0], b[0])
    assert len(list(tmp_path.glob("*.npz"))) == 1


@pytest.mark.parametrize("field,value", [("mode", "OUT-OF-FOLD"), ("held_out", "heldout"),
    ("model_hashes", {"synthetic": "different"}), ("config", {"cost": 20}),
    ("scorer_sha256", "other"), ("reference_hash", "other"), ("external_hashes", {"fixture": "other"})])
def test_cache_key_separates_models_splits_configuration_and_data(field, value):
    before = cache_metadata()
    after = copy.deepcopy(before)
    after[field] = value
    assert evaluation.cache_key(before) != evaluation.cache_key(after)


def test_cache_rejects_corrupt_array_bytes(tmp_path):
    metadata = cache_metadata()
    evaluation.probability_cache(tmp_path, metadata, lambda: (np.zeros(2), np.ones(2)))
    target = next(tmp_path.glob("*.npz"))
    with np.load(target, allow_pickle=False) as saved:
        contents = {key: saved[key] for key in saved.files}
    contents["home"] = np.zeros(2)
    np.savez(target, **contents)
    with pytest.raises(ValueError, match="integrity"):
        evaluation.probability_cache(tmp_path, metadata, lambda: pytest.fail("must not regenerate"))


def test_cache_rejects_invalid_generated_probabilities(tmp_path):
    with pytest.raises(ValueError):
        evaluation.probability_cache(tmp_path, cache_metadata(), lambda: (np.zeros(1), np.ones(1)))
    assert not list(tmp_path.glob("*.npz"))


def test_missing_permission_stops_before_dataset_access(monkeypatch):
    monkeypatch.setattr(evaluation, "verify_external_files", lambda _: pytest.fail("dataset accessed"))
    with pytest.raises(PermissionError):
        evaluation.run(SimpleNamespace(metrica_permission_confirmed=False))


def test_out_of_fold_requires_explicit_training_permission(tmp_path):
    with pytest.raises(ValueError, match="allow-training"):
        evaluation.fold_models(tmp_path, {}, "heldout", False, "cpu")
    assert not list(tmp_path.iterdir())


def test_out_of_fold_excludes_match_from_all_twelve_fits(tmp_path, monkeypatch):
    calls = []
    metadata = {**evaluation.mode_metadata("OUT-OF-FOLD", "heldout"),
                "training_matches": ["training"], "config": {"dilations": (1, 2)}}
    monkeypatch.setattr(evaluation.gbm, "fit", lambda kind, excluded: calls.append((kind, excluded)) or kind)
    monkeypatch.setattr(evaluation.tcn, "train",
                        lambda kind, excluded, device, log: calls.append((kind, excluded)) or kind)
    def save(value, kind):
        suffix = ".txt" if kind in evaluation.gbm.KINDS else ".pt"
        (evaluation.paths.MODEL_DIR / (kind + suffix)).write_text("synthetic " + value)
    monkeypatch.setattr(evaluation.gbm, "save", save)
    monkeypatch.setattr(evaluation.tcn, "save", save)
    previous = evaluation.paths.MODEL_DIR
    folder, hashes = evaluation.fold_models(tmp_path, metadata, "heldout", True, "cpu")
    assert len(calls) == 12 and all(excluded == ("heldout",) for _, excluded in calls)
    assert evaluation.paths.MODEL_DIR == previous
    assert len(hashes) == 12 and folder.is_relative_to(tmp_path)
    # JSON round-trip must not invalidate provenance containing tuples.
    assert evaluation.fold_models(tmp_path, metadata, "heldout", True, "cpu") == (folder, hashes)
    assert len(calls) == 12


def test_fold_training_cannot_use_diagnostic_provenance(tmp_path):
    metadata = {**evaluation.mode_metadata("DIAGNOSTIC"), "training_matches": ["heldout"]}
    with pytest.raises(ValueError):
        evaluation.fold_models(tmp_path, metadata, "heldout", True, "cpu")


def test_repositories_and_ancestor_paths_rejected():
    for path in (evaluation.ROOT, evaluation.ROOT / "cache", evaluation.ROOT.parent):
        with pytest.raises(ValueError):
            evaluation.outside_repositories(path)


def test_reference_outside_halves_is_not_silently_scored():
    dead, home, reference, halves = synthetic_comparison()
    reference[10] = data.H
    with pytest.raises(ValueError, match="outside"):
        evaluation.compare_paths(dead, home, reference, halves)


def test_synthetic_end_to_end_run_keeps_model_versions_and_summaries_separate(tmp_path, monkeypatch):
    dead, home, reference, halves = synthetic_comparison()
    match = data.Match("synthetic-external", np.zeros((22, 2, 2)), np.zeros((22, 2)),
                       np.array([data.H, data.A]), reference, (105.0, 68.0), halves)
    monkeypatch.setattr(evaluation, "verify_external_files", lambda _: {"synthetic": "hash"})
    monkeypatch.setattr(evaluation, "source_metadata", lambda: {"synthetic_source": "hash"})
    monkeypatch.setattr(data, "external_matches", lambda: [("synthetic_dataset", "synthetic-external", lambda: match)])
    model_version = ["one"]
    monkeypatch.setattr(evaluation, "model_hashes", lambda _: {"synthetic_model": model_version[0]})
    calls = []
    monkeypatch.setattr(pipeline, "dead_probability", lambda _: calls.append("dead") or dead)
    monkeypatch.setattr(pipeline, "home_probability", lambda _: calls.append("home") or home)
    args = SimpleNamespace(mode="DIAGNOSTIC", external_dir=tmp_path / "external",
                           workspace=tmp_path / "workspace", matches=None, device="cpu",
                           allow_training=False, metrica_permission_confirmed=True)
    reports = []
    previous = evaluation.paths.EXTERNAL_DIR, evaluation.paths.CACHE_DIR, evaluation.paths.MODEL_DIR
    evaluation.run(args, report=reports.append)
    evaluation.run(args, report=reports.append)
    assert reports[0]["interpretation"] == evaluation.MODES["DIAGNOSTIC"]
    assert reports[2]["probability_cache_reused"]
    assert calls == ["dead", "home"]
    model_version[0] = "two"
    evaluation.run(args, report=reports.append)
    assert calls == ["dead", "home", "dead", "home"]
    assert len(list((args.workspace / "summaries").glob("*.json"))) == 2
    assert len(list((args.workspace / "probabilities").glob("*.npz"))) == 2
    assert (evaluation.paths.EXTERNAL_DIR, evaluation.paths.CACHE_DIR, evaluation.paths.MODEL_DIR) == previous
    assert all("reference" not in json.loads(p.read_text()) for p in (args.workspace / "results").glob("*.json"))


def test_dfl_diagnostic_never_loads_metrica_or_requires_permission(tmp_path, monkeypatch):
    dead, home, reference, halves = synthetic_comparison()
    match = data.Match("DFL-J03WMX", np.zeros((22, 2, 2)), np.zeros((22, 2)),
                       np.array([data.H, data.A]), reference, (105.0, 68.0), halves)
    def forbidden():
        pytest.fail("Metrica loader accessed")
    monkeypatch.setattr(data, "external_matches", lambda: [
        ("metrica", "Metrica-1", forbidden), ("dfl", "DFL-J03WMX", lambda: match)])
    checked = []
    def verify(directory, dataset, match_names):
        checked.append((dataset, match_names))
        return {"dfl/synthetic": "hash"}
    monkeypatch.setattr(evaluation, "verify_external_files", verify)
    monkeypatch.setattr(evaluation, "source_metadata", lambda: {"synthetic": "hash"})
    monkeypatch.setattr(evaluation, "model_hashes", lambda _: {"synthetic": "hash"})
    monkeypatch.setattr(pipeline, "dead_probability", lambda _: dead)
    monkeypatch.setattr(pipeline, "home_probability", lambda _: home)
    args = SimpleNamespace(mode="DIAGNOSTIC", dataset="dfl", external_dir=tmp_path / "data",
                           workspace=tmp_path / "work", matches=["DFL-J03WMX"], device="cpu",
                           allow_training=False, metrica_permission_confirmed=False)
    reports = []
    evaluation.run(args, report=reports.append)
    assert checked == [("dfl", {"DFL-J03WMX"})]
    assert reports[0]["dataset"] == "dfl" and reports[0]["evaluation_dataset_scope"] == "dfl"
    assert reports[0]["probability_cache_integrity"] == "PASS"
    stored = json.loads(next((args.workspace / "results").glob("*.json")).read_text())
    assert not stored["metadata"]["metrica_permission_confirmed"]
    assert stored["metadata"]["training_corpus"] == ["Metrica-1", "DFL-J03WMX"]


def test_dfl_only_oof_rejected_before_any_data_or_training():
    with pytest.raises(ValueError, match="DIAGNOSTIC"):
        evaluation.run(SimpleNamespace(dataset="dfl", mode="OUT-OF-FOLD"))


def test_dfl_verifier_checks_only_selected_pair(tmp_path, monkeypatch):
    entries = [line.split() for line in (evaluation.ROOT / "scripts/external_sha256.txt").read_text().splitlines()]
    expected = {relative: digest for digest, relative in entries
                if relative.startswith("dfl/") and relative.endswith("DFL-MAT-J03WMX.xml")}
    for relative in expected:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
    def checked_hash(path):
        relative = Path(path).relative_to(tmp_path).as_posix()
        assert relative in expected
        return expected[relative]
    monkeypatch.setattr(evaluation, "file_hash", checked_hash)
    assert evaluation.verify_external_files(tmp_path, dataset="dfl", match_names={"DFL-J03WMX"}) == expected


def test_dfl_downloader_manifest_selection_never_includes_metrica():
    from scripts import download_dfl
    selected = download_dfl.selected_files(["J03WMX"])
    assert len(selected) == 2 and all(name.startswith("dfl/") for name in selected)
    assert len(download_dfl.selected_files(download_dfl.MATCHES)) == 14
