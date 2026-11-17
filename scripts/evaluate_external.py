"""External-only paired decoding evaluation. No downloads or competition-match entry points.

DIAGNOSTIC uses shipped models and reports in-sample diagnostics only. OUT-OF-FOLD explicitly
requires --allow-training and trains all twelve models with the scored match excluded.
All-dataset modes require documented Metrica permission. DFL-only diagnostics need no Metrica data.
"""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from possession import augment, data, decoding, gbm, paths, pipeline, tcn

MODES = {"DIAGNOSTIC": "in_sample_diagnostic_not_generalization",
         "OUT-OF-FOLD": "out_of_fold_fixed_configuration_not_nested_selection"}
METRICS = ("spellF1", "macroF1")
KINDS = gbm.KINDS + tcn.KINDS


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def cache_key(metadata):
    return hashlib.sha256(canonical(metadata).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode()
                          + array.tobytes()).hexdigest()


def mode_metadata(mode, held_out=None):
    if mode not in MODES:
        raise ValueError("explicit evaluation mode required")
    if (mode == "OUT-OF-FOLD") != (held_out is not None):
        raise ValueError("held-out identifier must be present only in OUT-OF-FOLD mode")
    return {"mode": mode, "interpretation": MODES[mode], "held_out": held_out}


def compare_paths(p_dead, p_home, reference, halves):
    """Same immutable probabilities, same references, same official scorer for both paths."""
    reference = np.asarray(reference)
    n_frames = len(reference)
    halves = pipeline._validated_halves(halves, n_frames)
    inside = np.zeros(n_frames, dtype=bool)
    for start, stop in halves:
        inside[start:stop] = True
    if reference.ndim != 1 or not np.isin(reference, [data.H, data.A, data.D, data.X]).all():
        raise ValueError("invalid reference codes")
    if np.any(reference[~inside] != data.X):
        raise ValueError("reference labels outside declared halves must be X")
    if not np.any(reference != data.X):
        raise ValueError("no scored external reference frames")
    p_dead, p_home = np.asarray(p_dead), np.asarray(p_home)
    for probability in (p_dead, p_home):
        if probability.shape != (n_frames,) or not np.isfinite(probability).all():
            raise ValueError("invalid probability shape or nonfinite values")
        if np.any((probability < 0) | (probability > 1)):
            raise ValueError("probabilities must be in [0, 1]")
    p_dead, p_home = p_dead.view(), p_home.view()
    p_dead.setflags(write=False)
    p_home.setflags(write=False)
    states = decoding.possession_states(p_dead, p_home, pipeline.TEAM_SWITCH_COST,
                                       pipeline.DEAD_SWITCH_COST)
    original = data.apply_control_rule(np.array([data.H, data.A, data.D], dtype=np.int8)[states])
    original[~inside] = data.X  # Mask after global decoding/control, as the original writer did.
    patched = pipeline._decode_halves(p_dead, p_home, halves, n_frames)
    scores = {}
    for name, prediction in (("original", original), ("patched", patched)):
        result = data.metrics.evaluate(reference, prediction, bootstrap_n=0)
        scores[name] = {"spellF1": float(result["spell_f1_mean"]),
                        "macroF1": float(result["macro_f1"])}
    scores["delta"] = {metric: scores["patched"][metric] - scores["original"][metric]
                       for metric in METRICS}
    different = (original != patched) & inside
    scores.update(decoded_labels_differ=bool(different.any()),
                  changed_half_frames=int(different.sum()),
                  changed_scored_frames=int((different & (reference != data.X)).sum()))
    return scores


def aggregate(records):
    """Unweighted match means within dataset; equal dataset means as a separate summary."""
    if not records:
        raise ValueError("no evaluation records")
    mode = records[0]["mode"]
    mode_metadata(mode, records[0]["held_out"])
    for record in records:
        expected = mode_metadata(record["mode"], record["held_out"])
        if record["mode"] != mode or record["interpretation"] != expected["interpretation"]:
            raise ValueError("mixed or incorrectly labeled evaluation modes")
    if len({r["match"] for r in records}) != len(records):
        raise ValueError("duplicate match records")
    datasets = {}
    for dataset in sorted({r["dataset"] for r in records}):
        selected = [r for r in records if r["dataset"] == dataset]
        datasets[dataset] = {"matches": len(selected), **{
            arm: {metric: float(np.mean([r[arm][metric] for r in selected]))
                  for metric in METRICS} for arm in ("original", "patched", "delta")}}
    return {"mode": mode, "interpretation": MODES[mode], "per_dataset_match_means": datasets,
            "equal_dataset_means": {
                arm: {metric: float(np.mean([v[arm][metric] for v in datasets.values()]))
                      for metric in METRICS} for arm in ("original", "patched", "delta")}}


def probability_cache(directory, metadata, generate):
    """Cache identity includes split/mode, model bytes, code/config, scorer, data and reference hashes."""
    target = Path(directory) / (cache_key(metadata) + ".npz")
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if str(saved["metadata"].item()) != canonical(metadata):
                raise ValueError("probability cache metadata mismatch")
            dead, home = saved["dead"], saved["home"]
            if str(saved["dead_hash"].item()) != array_hash(dead) or str(saved["home_hash"].item()) != array_hash(home):
                raise ValueError("probability cache integrity failure")
        validate_cached_probabilities(dead, home, metadata["n_frames"])
        return dead, home, True
    dead, home = generate()
    dead, home = np.asarray(dead), np.asarray(home)
    validate_cached_probabilities(dead, home, metadata["n_frames"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, dead=dead, home=home, metadata=canonical(metadata),
                 dead_hash=array_hash(dead), home_hash=array_hash(home))
    temporary.replace(target)
    return dead, home, False


def validate_cached_probabilities(dead, home, n_frames):
    for values in (dead, home):
        if values.shape != (n_frames,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("invalid cached probabilities")


def outside_repositories(path):
    path = Path(path).resolve()
    for repository in (ROOT, paths.COMPETITION_REPO.resolve()):
        if path == repository or path.is_relative_to(repository) or repository.is_relative_to(path):
            raise ValueError("external data and outputs must be in dedicated directories outside both repositories")
    return path


def verify_external_files(directory, dataset="all", match_names=None):
    """Only read exact external-file names from the pinned external manifest."""
    manifest = ROOT / "scripts/external_sha256.txt"
    hashes = {}
    for line in manifest.read_text().splitlines():
        expected, relative = line.split()
        if dataset == "dfl":
            if not relative.startswith("dfl/"):
                continue
            if match_names and not any(relative.endswith("DFL-MAT-" + name.removeprefix("DFL-") + ".xml")
                                       for name in match_names):
                continue
        target = (directory / relative).resolve()
        if not target.is_relative_to(directory) or not target.is_file():
            raise FileNotFoundError("required external dataset file absent")
        actual = file_hash(target)
        if actual != expected:
            raise ValueError("external dataset checksum mismatch")
        hashes[relative] = actual
    return hashes


def model_hashes(directory):
    return {kind: file_hash(Path(directory) / (kind + (".txt" if kind in gbm.KINDS else ".pt")))
            for kind in KINDS}


@contextmanager
def model_directory(directory):
    previous = paths.MODEL_DIR
    paths.MODEL_DIR = Path(directory)
    try:
        yield
    finally:
        paths.MODEL_DIR = previous


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fold_models(workspace, metadata, held_out, allow_training, device):
    """Never substitute shipped models for an incomplete or missing OUT-OF-FOLD fold."""
    if not allow_training:
        raise ValueError("OUT-OF-FOLD requires explicit --allow-training")
    if metadata["mode"] != "OUT-OF-FOLD" or metadata["held_out"] != held_out:
        raise ValueError("fold training requires matching OUT-OF-FOLD provenance")
    if held_out in metadata["training_matches"]:
        raise ValueError("held-out match must not enter training")
    folder = workspace / "fold-models" / cache_key(metadata)
    manifest = folder / "provenance.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if canonical(saved["metadata"]) != canonical(metadata) or saved["model_hashes"] != model_hashes(folder):
            raise ValueError("fold model provenance mismatch")
        return folder, saved["model_hashes"]
    if folder.exists():
        raise ValueError("incomplete fold model directory; use a fresh workspace")
    folder.mkdir(parents=True)
    with model_directory(folder):
        for kind in gbm.KINDS:
            gbm.save(gbm.fit(kind, excluded=(held_out,)), kind)
        for kind in tcn.KINDS:
            tcn.save(tcn.train(kind, excluded=(held_out,), device=device, log=lambda _: None), kind)
    hashes = model_hashes(folder)
    write_json(manifest, {"metadata": metadata, "model_hashes": hashes})
    return folder, hashes


def source_metadata():
    files = list((ROOT / "possession").glob("*.py")) + [Path(__file__), ROOT / "pyproject.toml", ROOT / "uv.lock",
                                                                    ROOT / "scripts/external_sha256.txt"]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(files)}


def configuration():
    return {"team_switch_cost": pipeline.TEAM_SWITCH_COST, "dead_switch_cost": pipeline.DEAD_SWITCH_COST,
            "control_seconds": data.CONTROL_SECONDS, "control_frames": round(data.CONTROL_SECONDS * data.FPS),
            "player_gap_fill_seconds": pipeline.PLAYER_GAP_FILL_SECONDS, "bootstrap_n": 0,
            "gbm_params": gbm.MODEL_PARAMS, "tcn_seeds": {k: tcn.SEED_OF_KIND.get(k, 0) for k in tcn.KINDS},
            "augmentation_seed_policy": {"gbm_ball": "crc32(match.name)+7",
                                         "tcn_ball": "crc32(source.name)+13",
                                         "player_dropout": "crc32(match.name)+19"},
            "tcn_train_steps": tcn.TRAIN_STEPS, "tcn_channels": tcn.CHANNELS,
            "tcn_dilations": tcn.DILATIONS, "reference_processing": "existing external loader plus 2s control",
            "half_processing": "existing 1000-frame NaN/X separator; probability context unchanged",
            "aggregation": "unweighted per-match means per dataset; separate equal-dataset mean"}


def run(args, report=lambda value: print(canonical(value), flush=True)):
    dataset_scope = getattr(args, "dataset", "all")
    if dataset_scope not in ("all", "dfl"):
        raise ValueError("unsupported evaluation dataset scope")
    if dataset_scope == "dfl" and args.mode != "DIAGNOSTIC":
        raise ValueError("DFL-only scope supports DIAGNOSTIC; existing OOF trainers require the full corpus")
    if dataset_scope == "all" and not args.metrica_permission_confirmed:
        raise PermissionError("Metrica permission unresolved; no dataset access performed")
    if args.mode == "OUT-OF-FOLD" and not args.allow_training:
        raise ValueError("OUT-OF-FOLD requires explicit --allow-training")
    if args.mode == "DIAGNOSTIC" and args.allow_training:
        raise ValueError("DIAGNOSTIC must not train")
    external = outside_repositories(args.external_dir)
    workspace = outside_repositories(args.workspace)
    if external == workspace or external.is_relative_to(workspace) or workspace.is_relative_to(external):
        raise ValueError("dataset and workspace directories must be separate")
    all_matches = data.external_matches()
    training_names = [name for _, name, _ in all_matches]
    matches = [item for item in all_matches if dataset_scope == "all" or item[0] == "dfl"]
    names = [name for _, name, _ in matches]
    if args.matches and not set(args.matches) <= set(names):
        raise ValueError("only configured external match identifiers are allowed")
    selected = set(args.matches or names)
    hashes = (verify_external_files(external, dataset="dfl", match_names=selected)
              if dataset_scope == "dfl" else verify_external_files(external))
    import torch
    context = {"schema_version": 1, "sources": source_metadata(), "external_hashes": hashes,
               "scorer_sha256": file_hash(data.metrics.__file__), "config": configuration(),
               "packages": {name: importlib.metadata.version(name) for name in
                            ("numpy", "pandas", "numba", "lightgbm", "torch")},
               "python": platform.python_version(), "platform": platform.platform(),
               "device": args.device, "torch_threads": torch.get_num_threads(),
               "environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")},
               "metrica_permission_confirmed": bool(args.metrica_permission_confirmed) if dataset_scope == "all" else False,
               "evaluation_dataset_scope": dataset_scope, "training_corpus": training_names,
               "selection_bias_warning": "Historical settings used these datasets; OOF is not independent nested selection"}
    # Source-derived cache namespace prevents stale match/feature/training-row caches.
    namespace = workspace / "prepared" / cache_key(context)
    previous_external, previous_cache = paths.EXTERNAL_DIR, paths.CACHE_DIR
    shipped_models = paths.MODEL_DIR
    records = []
    try:
        paths.EXTERNAL_DIR, paths.CACHE_DIR = external, namespace
        for dataset, name, load in matches:
            if name not in selected:
                continue
            match_started = time.monotonic()
            split = mode_metadata(args.mode, name if args.mode == "OUT-OF-FOLD" else None)
            metadata = {**context, **split, "training_matches": [n for n in training_names if n != split["held_out"]]}
            if args.mode == "OUT-OF-FOLD":
                folder, weights = fold_models(workspace, metadata, name, args.allow_training, args.device)
            else:
                folder, weights = shipped_models, model_hashes(shipped_models)
            match = load()  # Only functions returned by data.external_matches(), never competition loader.
            key = {**metadata, "model_hashes": weights, "match": name, "halves": match.halves,
                   "n_frames": len(match.labels),
                   "reference_hash": array_hash(match.labels), "players_shape": match.players_m.shape,
                   "ball_shape": match.ball_m.shape}

            def generate():
                prepared = data.only_halves(augment.fill_player_gaps(match, pipeline.PLAYER_GAP_FILL_SECONDS))
                with model_directory(folder):
                    return pipeline.dead_probability(prepared), pipeline.home_probability(prepared)

            dead, home, reused = probability_cache(workspace / "probabilities", key, generate)
            def cache_must_exist():
                raise ValueError("probability cache disappeared during verification")
            verified_dead, verified_home, _ = probability_cache(workspace / "probabilities", key, cache_must_exist)
            if not np.array_equal(dead, verified_dead) or not np.array_equal(home, verified_home):
                raise ValueError("probability cache round-trip mismatch")
            record = {**split, "dataset": dataset, "match": name, "probability_cache_key": cache_key(key),
                      "probability_cache_reused": reused,
                      "probability_cache_integrity": "PASS", "evaluation_dataset_scope": dataset_scope,
                      **compare_paths(dead, home, match.labels, match.halves)}
            record["elapsed_seconds"] = round(time.monotonic() - match_started, 2)
            records.append(record)
            write_json(workspace / "results" / (cache_key(key) + ".json"), {"metadata": key, "result": record})
            report(record)  # Aggregate metrics/counts only, never sequences.
    finally:
        paths.EXTERNAL_DIR, paths.CACHE_DIR = previous_external, previous_cache
    summary = aggregate(records)
    summary["evaluation_dataset_scope"] = dataset_scope
    summary_metadata = {"context": context, "mode": args.mode, "matches": sorted(selected),
                        "probability_cache_keys": [record["probability_cache_key"] for record in records]}
    write_json(workspace / "summaries" / (cache_key(summary_metadata) + ".json"),
               {"metadata": summary_metadata, "summary": summary})
    report(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--dataset", choices=("all", "dfl"), default="all")
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--matches", nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--metrica-permission-confirmed", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    cpu_started = time.process_time()
    # Suppress Python AND native third-party logs. Forward only structured aggregate results.
    saved_out, saved_err = os.dup(1), os.dup(2)
    error_category = None
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                run(args, report=lambda value: os.write(saved_out, (canonical(value) + "\n").encode()))
        except Exception as error:
            error_category = type(error).__name__
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
    if error_category:
        print(canonical({"status": "FAIL", "error_category": error_category}), flush=True)
        return 1
    print(canonical({"status": "PASS", "elapsed_seconds": round(time.monotonic() - started, 2),
                     "cpu_seconds": round(time.process_time() - cpu_started, 2),
                     "peak_process_memory_mib": peak_memory_mib()}))
    return 0


def peak_memory_mib():
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak", "working", "peak_paged", "paged",
                                               "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD)
    if not query(current(), ctypes.byref(counters), counters.cb):
        return None
    return round(counters.peak / (1024 * 1024), 2)


if __name__ == "__main__":
    sys.exit(main())
