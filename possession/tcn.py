"""Temporal convolutional classifiers over per-frame signals, trained on the external matches only.

The network is a stack of dilated residual 1-D convolutions over time (the TCN of Lea et al. 2017,
arXiv:1611.05267, in the residual form of MS-TCN, Farha & Gall 2019, arXiv:1903.01945), at 5 steps
per second. It learns its own temporal filters from the base signals, in place of the fixed
multi-scale windows of the LightGBM classifiers.

- tcn_dead: P(dead) from player movement and per-team layout; no ball input (see gbm.py).
- tcn_team_change: P(home | live) from per-team signals with the ball, touch timing and ball
  advance, trained on copies with simulated ball loss. Steps within CHANGE_WINDOW_S of a true team
  change weigh CHANGE_WEIGHT times more. _s1 and _s2 are the same network with other seeds.
- Suffix _pd: also trained on copies with lost player tracks.
Each network is exactly team-symmetric: swapped training crops, pooled normalisation of each
_H/_A column pair, and the mean over the team-exchanged input at inference.
"""
import dataclasses
import os
import zlib

# The submission must be reproducible exactly, so GPU training must be deterministic. cuBLAS needs
# this workspace setting before CUDA starts (https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from . import augment, data, features, gbm, paths  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

STEP_FRAMES = 5  # 25 fps / 5 = 5 steps per second
STEPS_PER_SECOND = data.FPS // STEP_FRAMES
CHANNELS = 64
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)  # receptive field 1 + 2 * 255 = 511 steps, about 102 s
DROPOUT = 0.1
CROP_STEPS = 1024
BATCH = 16
TRAIN_STEPS = 1500
LEARNING_RATE = 1e-3
TEAM_TRAIN_GAPS = ((0.0, "live"), (0.15, "uniform"), (0.30, "uniform"))  # (ball-loss share, placement) copies
CHANGE_WINDOW_S = 2.0
CHANGE_WEIGHT = 3.0
SEED_OF_KIND = {"tcn_team_change_s1": 1, "tcn_team_change_s2": 2, "tcn_team_change_s1_pd": 1, "tcn_team_change_s2_pd": 2}
_TEAM_BASES = ("tcn_team_change", "tcn_team_change_s1", "tcn_team_change_s2")
KINDS = ("tcn_dead", "tcn_dead_pd") + _TEAM_BASES + tuple(kind + gbm.PLAYER_DROPOUT_SUFFIX for kind in _TEAM_BASES)


def is_dead_kind(kind):
    return kind.startswith("tcn_dead")


def signal_set(kind):
    """The input signals of a kind: "dead" or "team"; kinds with the same set share one input table."""
    return "dead" if is_dead_kind(kind) else "team"


def base_signals(match, kind):
    """(names, (frames, signals)) at 25 fps; names end in _H/_A where a signal belongs to one team."""
    names, table = features.player_signals(match)
    if is_dead_kind(kind):
        team_names, team_table = features.team_signals(match, with_ball=False)
        return names + team_names, np.concatenate((table, team_table), axis=1)
    parts = [(names, table), features.team_signals(match, with_ball=True), features.touch_signals(match),
             features.ball_advance_signals(match)]
    return [n for part in parts for n in part[0]], np.concatenate([part[1] for part in parts], axis=1)


def to_steps(values, segments=None):
    """Mean over blocks of STEP_FRAMES frames that skips NaN; NaN where a block has no value."""
    if segments is not None:
        return np.concatenate([to_steps(values[start:stop]) for start, stop in segments])
    steps = len(values) // STEP_FRAMES
    blocks = values[:steps * STEP_FRAMES].reshape(steps, STEP_FRAMES, *values.shape[1:])
    valid = np.isfinite(blocks)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid, blocks, 0.0).sum(axis=1) / valid.sum(axis=1)


def step_labels(labels, segments=None):
    """Label of the centre frame of each block."""
    if segments is not None:
        return np.concatenate([step_labels(labels[start:stop]) for start, stop in segments])
    steps = len(labels) // STEP_FRAMES
    return labels[STEP_FRAMES // 2:steps * STEP_FRAMES:STEP_FRAMES]


def step_inputs(match, kind):
    """(signal names, signals at steps) of one match: the input of `probability`."""
    names, table = base_signals(match, kind)
    return names, to_steps(table, match.halves or None).astype(np.float32)


def _prepared(match, kind, share=0.0, placement="live"):
    """(names, signals at steps, labels at steps) of one training copy of an external match, cached."""
    path = paths.CACHE_DIR / "steps" / "{}-{}-{}-{:.2f}.npz".format(signal_set(kind), match.name, placement, share)
    if path.is_file():
        stored = np.load(path)
        return list(stored["names"]), stored["signals"], stored["labels"]
    gapped = augment.add_live_ball_gaps(match, share, zlib.crc32(match.name.encode()) + 13, placement)
    names, signals = step_inputs(gapped, kind)
    labels = step_labels(match.labels, match.halves or None)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.save_npz_atomic(path, names=np.array(names), signals=signals, labels=labels)
    return names, signals, labels


def _sources(kind, match):
    """[match], plus its copy with lost player tracks for a _pd kind; the copy has its own name and so its own seeds."""
    sources = [match]
    if kind.endswith(gbm.PLAYER_DROPOUT_SUFFIX):
        dropped = gbm.dropout_copy(match)
        sources.append(dataclasses.replace(dropped, name="{}~pd{:.3f}".format(match.name, gbm.TRAIN_PLAYER_DROPOUT)))
    return sources


def training_copies(kind, match):
    """[(names, signals, labels)] of one external match: the dead kinds use each source once, the team kinds its ball-loss copies."""
    if is_dead_kind(kind):
        return [_prepared(source, kind) for source in _sources(kind, match)]
    return [_prepared(source, kind, share, placement) for source in _sources(kind, match) for share, placement in TEAM_TRAIN_GAPS]


class Block(nn.Module):
    def __init__(self, dilation, channels):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return x + self.mix(self.dropout(torch.relu(self.dilated(x))))


class TCN(nn.Module):
    def __init__(self, n_inputs, channels=CHANNELS):
        super().__init__()
        self.inputs = nn.Conv1d(n_inputs, channels, 1)
        self.blocks = nn.Sequential(*[Block(d, channels) for d in DILATIONS])
        self.output = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        return self.output(self.blocks(self.inputs(x)))[:, 0]


def network_input(signals, mean, std):
    """(channels, steps) float32: normalised signals with 0 where absent, then the presence mask."""
    present = np.isfinite(signals)
    normalised = np.where(present, (signals - mean) / std, 0.0)
    return np.concatenate((normalised, present), axis=1).T.astype(np.float32)


def exchanged_input(network_inputs, order):
    """The network input of the team-exchanged match: each block of signal channels in swap order."""
    n_names = len(order)
    return np.concatenate([network_inputs[b * n_names:(b + 1) * n_names][order] for b in range(network_inputs.shape[0] // n_names)])


def _targets(kind, labels):
    """(target, weight) per step: dead against labelled steps, or home against live steps; weight 0 elsewhere."""
    if is_dead_kind(kind):
        return (labels == data.D).astype(np.float32), (labels != data.X).astype(np.float32)
    live = (labels == data.H) | (labels == data.A)
    weight = live.astype(np.float32)
    change = np.flatnonzero(live[1:] & live[:-1] & (labels[1:] != labels[:-1])) + 1
    near = np.zeros(len(labels), dtype=bool)
    reach = int(round(CHANGE_WINDOW_S * STEPS_PER_SECOND))
    for index in change:
        near[max(0, index - reach):index + reach] = True
    weight[near & live] = CHANGE_WEIGHT
    return (labels == data.H).astype(np.float32), weight


def pooled_statistics(stacked, order):
    """(mean, std) per signal column over the training steps.

    The _H and _A column of a pair share one mean and one std; otherwise the normalised input of the
    team-exchanged match is not the swapped normalised input, and the swap average is not symmetric.
    """
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0) + 1e-3
    return 0.5 * (mean + mean[order]), 0.5 * (std + std[order])


def train(kind, excluded=(), device="cpu", log=print):
    """A TCN of one kind trained on every external match except the excluded names; returns the state to save."""
    if kind not in KINDS:
        raise ValueError(kind)
    seed = SEED_OF_KIND.get(kind, 0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sources = []
    for set_name, match_name, load in data.external_matches():
        if match_name in excluded:
            continue
        for names, signals, labels in training_copies(kind, load()):
            sources.append((set_name, names, signals, labels))
    names = sources[0][1]
    order = features.swap_order(names)
    mean, std = pooled_statistics(np.concatenate([s[2] for s in sources]), order)
    inputs, targets, weights, set_of = [], [], [], []
    for set_name, _, signals, labels in sources:
        inputs.append(network_input(signals, mean, std))
        target, weight = _targets(kind, labels)
        targets.append(target)
        weights.append(weight)
        set_of.append(set_name)
    # Each set of matches gets the same share of the crops, as each gets the same weight in the selection.
    set_names = sorted(set(set_of))
    pick = np.array([1.0 / (len(set_names) * set_of.count(s)) for s in set_of])
    network = TCN(2 * len(names), CHANNELS).to(device)
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    for step in range(TRAIN_STEPS):
        batch_x, batch_y, batch_w = [], [], []
        for _ in range(BATCH):
            source = rng.choice(len(inputs), p=pick)
            start = int(rng.integers(0, max(1, inputs[source].shape[1] - CROP_STEPS)))
            x = inputs[source][:, start:start + CROP_STEPS]
            y = targets[source][start:start + CROP_STEPS]
            w = weights[source][start:start + CROP_STEPS]
            if rng.random() < 0.5:
                x = exchanged_input(x, order)
                if not is_dead_kind(kind):
                    y = 1.0 - y
            pad = CROP_STEPS - x.shape[1]
            batch_x.append(np.pad(x, ((0, 0), (0, pad))))
            batch_y.append(np.pad(y, (0, pad)))
            batch_w.append(np.pad(w, (0, pad)))
        x = torch.from_numpy(np.stack(batch_x)).to(device)
        y = torch.from_numpy(np.stack(batch_y)).to(device)
        w = torch.from_numpy(np.stack(batch_w)).to(device)
        loss = (loss_function(network(x), y) * w).sum() / w.sum().clamp(min=1.0)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 500 == 0 or step == TRAIN_STEPS - 1:
            log("  {} step {} loss {:.4f}".format(kind, step, loss.item()))
    return {"state": {k: v.cpu() for k, v in network.state_dict().items()}, "names": names, "mean": mean, "std": std,
            "channels": CHANNELS}


def model_path(kind):
    return paths.MODEL_DIR / "{}.pt".format(kind)


def save(saved, kind):
    path = model_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(saved, temporary)
    temporary.replace(path)


def load(kind):
    path = model_path(kind)
    if not path.is_file():
        raise FileNotFoundError("{} is missing; run scripts/train.py".format(path))
    # The saved dictionary holds numpy arrays (signal names and statistics), which weights_only loading refuses.
    return torch.load(path, weights_only=False)


def probability(saved, names, signals, kind, segments=None):
    """Per-frame team-symmetric probability (P(dead) or P(home | live)) at 25 fps from step signals.

    Runs on the CPU: inference of one match takes seconds.
    """
    if list(names) != list(saved["names"]):
        raise ValueError("signal names differ from the training names")
    network = TCN(2 * len(names), saved["channels"])
    network.load_state_dict(saved["state"])
    network.eval()
    if segments is None:
        chunks = [signals]
    else:
        counts = [(stop - start) // STEP_FRAMES for start, stop in segments]
        chunks, offset = [], 0
        for count in counts:
            chunks.append(signals[offset:offset + count])
            offset += count
    outputs = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        x = network_input(chunk, saved["mean"], saved["std"])
        swapped = exchanged_input(x, features.swap_order(list(names)))
        with torch.no_grad():
            direct = torch.sigmoid(network(torch.from_numpy(x)[None]))[0].numpy()
            mirror = torch.sigmoid(network(torch.from_numpy(swapped)[None]))[0].numpy()
        outputs.append(0.5 * (direct + (mirror if is_dead_kind(kind) else 1.0 - mirror)))
    steps = np.concatenate(outputs) if outputs else np.empty(0)
    return np.repeat(steps, STEP_FRAMES)
