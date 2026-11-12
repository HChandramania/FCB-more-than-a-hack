"""Every match in one form: per-frame positions in metres and per-frame H/A/D/X labels."""
import dataclasses
import io
import os
import xml.etree.ElementTree as ElementTree

import numpy as np
import pandas as pd

from . import external, paths

paths.competition_modules()
import possession_metrics as metrics  # noqa: E402
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

H, A, D, X = metrics.H, metrics.A, metrics.D, metrics.X
FPS = 25
# NaN frames between the halves of an external match. Wider than twice any smoothing window, so
# no window spans two halves, and labelled X, so the scorer treats each half as its own run.
HALF_SEPARATOR_FRAMES = 1000
# CHALLENGE.md: possession changes only on controlled possession, with the proxy "a player keeps
# the ball for a few seconds"; contested or loose phases stay with the last team. 2 s is the
# smallest reading of "a few". It comes from that text, never from the competition labels.
CONTROL_SECONDS = 2.0

DFL_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
METRICA_GAMES = (1, 2, 3)


@dataclasses.dataclass
class Match:
    name: str
    players_m: np.ndarray  # (frames, slots, 2) metres from the pitch corner, NaN where not tracked
    ball_m: np.ndarray  # (frames, 2) metres, NaN where the ball is not tracked
    slot_team: np.ndarray  # (slots,) H or A
    labels: np.ndarray  # (frames,) H/A/D/X codes; all X on the competition match, whose labels are never read
    pitch_m: tuple  # (length, width)
    halves: tuple = ()  # ((start, stop), ...) frame ranges of the halves; the teams change sides between them


def save_npz_atomic(path, **arrays):
    """np.savez through a temporary file and one rename, so that an interrupted write leaves no partial cache file.

    The temporary name ends in .npz because np.savez adds .npz to any other name.
    """
    path = str(path)
    temporary = "{}.{}.partial.npz".format(path[:-len(".npz")], os.getpid())
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _cached(name, build):
    path = paths.CACHE_DIR / "matches" / name
    if path.is_file():
        data = np.load(path)
        return {key: data[key] for key in data.files}
    arrays = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(path, **arrays)
    return arrays


def _join_halves(halves):
    """One timeline from the halves, with a separator so that nothing crosses half time."""
    n_slots = halves[0]["players_m"].shape[1]
    gap_players = np.full((HALF_SEPARATOR_FRAMES, n_slots, 2), np.nan)
    gap_ball = np.full((HALF_SEPARATOR_FRAMES, 2), np.nan)
    gap_codes = np.full(HALF_SEPARATOR_FRAMES, X, dtype=np.int8)
    players, ball, codes = [], [], []
    for index, half in enumerate(halves):
        if index > 0:
            players.append(gap_players)
            ball.append(gap_ball)
            codes.append(gap_codes)
        players.append(half["players_m"])
        ball.append(half["ball_m"])
        codes.append(half["codes"].astype(np.int8))
    return np.concatenate(players), np.concatenate(ball), np.concatenate(codes)


def _separated_halves(players):
    """Half ranges of a timeline from _join_halves: the separator is the only long run without any player."""
    empty = ~np.isfinite(players[:, :, 0]).any(axis=1)
    edges = np.diff(np.concatenate(([0], empty.astype(np.int8), [0])))
    runs = [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if b - a >= HALF_SEPARATOR_FRAMES]
    if len(runs) != 1:
        raise ValueError("expected one half separator, found {}".format(len(runs)))
    (start, stop), = runs
    return ((0, int(start)), (int(stop), len(players)))


def only_halves(match):
    """Copy of the match without positions outside its halves (warm-up, halftime, after the match).

    The external timelines hold only the halves, so the multi-scale windows must not see other frames.
    """
    inside = np.zeros(len(match.labels), dtype=bool)
    for start, stop in match.halves:
        inside[start:stop] = True
    if inside.all():
        return match
    return dataclasses.replace(match, players_m=np.where(inside[:, None, None], match.players_m, np.nan),
                               ball_m=np.where(inside[:, None], match.ball_m, np.nan))


def apply_control_rule(codes, control_seconds=CONTROL_SECONDS):
    """Labels in the competition convention of controlled possession.

    Inside each stretch of live play the first team (the restarting team) owns the ball. A run of
    the other team shorter than control_seconds is a touch or a contest, so its frames stay with
    the owner; a run at least that long transfers possession.
    """
    min_frames = int(round(control_seconds * FPS))
    out = codes.copy()
    alive = (codes == H) | (codes == A)
    edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        stretch = codes[start:stop]
        bounds = np.concatenate(([0], np.flatnonzero(stretch[1:] != stretch[:-1]) + 1, [len(stretch)]))
        owner = stretch[0]
        for low, high in zip(bounds[:-1], bounds[1:]):
            team = stretch[low]
            if team == owner:
                continue
            if high - low >= min_frames:
                owner = team
            else:
                out[start + low:start + high] = owner
    return out


def load_competition_match():
    """The competition match from the provided tactical tracking. No ground truth is read: every label is X.

    Positions are parsed in float64 from the raw ATD text, as starter/baseline.py does, so that the
    nearest-player choice matches the published baselines exactly.
    """
    files = asset_paths()

    def build():
        with open(files["atd_raw"]) as handle:
            text = handle.read().replace(":", ",").replace(";", ",")
        # round_trip parses each number exactly as Python float() does in starter/baseline.py.
        table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64, na_values=["NaN"],
                            float_precision="round_trip").values
        if not np.array_equal(table[:, 0], np.arange(1, len(table) + 1)):
            raise ValueError("ATD frame ids are not 1..n; video_frame = frame_id - 1 would be wrong")
        n_slots = (table.shape[1] - 4) // 3
        root = ElementTree.parse(files["atd_meta"]).getroot()
        team_ids = [team.get("id") for team in root.iter("Team")]
        slot_team = np.array([H if p.get("teamId") == team_ids[0] else A for p in root.iter("Player")],
                             dtype=np.int8)
        players = table[:, 1:1 + 3 * n_slots].reshape(len(table), n_slots, 3)[:, :, :2]
        ball = table[:, 1 + 3 * n_slots:1 + 3 * n_slots + 2]
        return {"players": players, "ball": ball, "slot_team": slot_team}

    arrays = _cached("{}.npz".format(GAME), build)
    pitch = np.array(GAMES[GAME]["pitch"], dtype=np.float64)
    labels = np.full(len(arrays["players"]), X, dtype=np.int8)
    halves = tuple((first, last + 1) for _, first, last in half_video_blocks(GAME))
    return Match(GAME, arrays["players"] * pitch, arrays["ball"] * pitch, arrays["slot_team"],
                 labels, tuple(pitch), halves)


def graded_frames(n_frames):
    """Mask of the frames inside the two halves that the organisers grade (games.half_video_blocks)."""
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    return graded


def _external_match(name, build):
    arrays = _cached(name + ".npz", build)
    return Match(name, arrays["players"], arrays["ball"], arrays["slot_team"], apply_control_rule(arrays["codes"]),
                 tuple(arrays["pitch"]), _separated_halves(arrays["players"]))


def _joined(halves, slot_team, pitch_m):
    players, ball, codes = _join_halves(halves)
    return {"players": players, "ball": ball, "codes": codes, "slot_team": slot_team, "pitch": np.array(pitch_m)}


def load_dfl(match_id):
    folder = paths.EXTERNAL_DIR / "dfl"

    def build():
        positions, = folder.glob("DFL_04_03_positions_raw_observed_*{}.xml".format(match_id))
        info, = folder.glob("DFL_02_01_matchinformation_*{}.xml".format(match_id))
        return _joined(*external.read_dfl(positions, info))

    return _external_match("DFL-" + match_id, build)


def load_metrica(game):
    return _external_match("Metrica-{}".format(game), lambda: _joined(*external.read_metrica(paths.EXTERNAL_DIR / "metrica", game)))


def external_matches():
    """[(set name, match name, loader)] of the labelled external matches, in a fixed order.

    The order fixes the order of the training rows, so it is part of the reproduction of the models.
    """
    return ([("metrica", "Metrica-{}".format(game), lambda game=game: load_metrica(game)) for game in METRICA_GAMES]
            + [("dfl", "DFL-" + match_id, lambda match_id=match_id: load_dfl(match_id)) for match_id in DFL_MATCHES])
