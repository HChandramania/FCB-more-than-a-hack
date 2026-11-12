"""Readers for the external matches with per-frame possession labels: the training and validation data.

- Sportec Open DFL dataset: Bassek et al. 2025, Scientific Data 12:195,
  https://doi.org/10.6084/m9.figshare.28196177, CC BY 4.0. The ball object carries BallStatus
  (1 alive, 0 dead) and BallPossession (1 home, 2 away) on every frame.
- Metrica Sports sample games 1-3: https://github.com/metrica-sports/sample-data. Metrica publishes
  events, not per-frame possession, so derive_metrica_labels() builds the labels from the events.

Every reader returns a list of halves. A half is a dict with codes (H/A/D/X per frame),
players_m (frames, slots, 2) and ball_m (frames, 2) in metres from the pitch corner, with NaN where
an object is absent, and the match-wide slot_team and pitch_m.
"""
import csv
import json
import os
import re
import xml.etree.ElementTree as ElementTree
from array import array

import numpy as np

H, A, D, X = 0, 1, 2, 3  # the codes of evaluation/possession_metrics.py

DFL_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)"')
DFL_FRAME = re.compile(r'<Frame N="(\d+)" [^>]*?X="([-\d.]+)" Y="([-\d.]+)"')
DFL_BALL_FLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')
DFL_PITCH = re.compile(r'<PitchSize X="([\d.]+)" Y="([\d.]+)"')
DFL_SECTIONS = ("firstHalf", "secondHalf")

METRICA_PITCH_M = (105.0, 68.0)  # the sample-data README gives 105 x 68 m for all three games
METRICA_ON_BALL = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT"}


def read_dfl(positions_path, info_path):
    """Halves of one DFL match. Coordinates move from the pitch centre to the corner."""
    with open(info_path) as handle:
        home_id = re.search(r'HomeTeamId="([^"]+)"', handle.read()).group(1)

    sections, pitch_m, current = {}, None, None
    with open(positions_path) as handle:
        for line in handle:
            if pitch_m is None and "<PitchSize" in line:
                found = DFL_PITCH.search(line)
                pitch_m = (float(found.group(1)), float(found.group(2)))
            if "<FrameSet " in line:
                section, team, person = DFL_FRAMESET.search(line).groups()
                store = sections.setdefault(section, {"ball": None, "players": {}})
                if team == "BALL":
                    current = {"n": array("i"), "x": array("f"), "y": array("f"),
                               "possession": array("b"), "status": array("b")}
                    store["ball"] = current
                elif team == "referee":
                    current = None
                else:
                    current = {"team": H if team == home_id else A,
                               "n": array("i"), "x": array("f"), "y": array("f")}
                    store["players"][person] = current
                continue
            if current is None or "<Frame " not in line:
                continue
            found = DFL_FRAME.search(line)
            current["n"].append(int(found.group(1)))
            current["x"].append(float(found.group(2)))
            current["y"].append(float(found.group(3)))
            if "possession" in current:
                possession, status = DFL_BALL_FLAGS.search(line).groups()
                current["possession"].append(int(possession))
                current["status"].append(int(status))

    persons = sorted({p for name in DFL_SECTIONS for p in sections[name]["players"]})
    team_of = {}
    for name in DFL_SECTIONS:
        for person, track in sections[name]["players"].items():
            team_of[person] = track["team"]
    slot_team = np.array([team_of[p] for p in persons], dtype=np.int8)
    offset = np.array([pitch_m[0] / 2.0, pitch_m[1] / 2.0], dtype=np.float64)

    halves = []
    for name in DFL_SECTIONS:
        ball = sections[name]["ball"]
        first = int(np.min(ball["n"]))
        n_frames = int(np.max(ball["n"])) - first + 1
        rows = np.asarray(ball["n"]) - first
        status = np.full(n_frames, -1, dtype=np.int8)
        possession = np.zeros(n_frames, dtype=np.int8)
        status[rows] = np.asarray(ball["status"])
        possession[rows] = np.asarray(ball["possession"])
        codes = np.full(n_frames, X, dtype=np.int8)
        codes[status == 0] = D
        codes[(status == 1) & (possession == 1)] = H
        codes[(status == 1) & (possession == 2)] = A

        ball_m = np.full((n_frames, 2), np.nan)
        ball_m[rows, 0] = np.asarray(ball["x"], dtype=np.float64) + offset[0]
        ball_m[rows, 1] = np.asarray(ball["y"], dtype=np.float64) + offset[1]
        players_m = np.full((n_frames, len(persons), 2), np.nan)
        for column, person in enumerate(persons):
            track = sections[name]["players"].get(person)
            if track is None:
                continue
            track_rows = np.asarray(track["n"]) - first
            keep = (track_rows >= 0) & (track_rows < n_frames)
            players_m[track_rows[keep], column, 0] = np.asarray(track["x"], dtype=np.float64)[keep] + offset[0]
            players_m[track_rows[keep], column, 1] = np.asarray(track["y"], dtype=np.float64)[keep] + offset[1]
        halves.append({"codes": codes, "players_m": players_m, "ball_m": ball_m})
    return halves, slot_team, pitch_m


def _metrica_stop_frame(event):
    """Frame where this event stops play, or None (Metrica events-definitions.pdf)."""
    kind, subtypes = event["type"], event["subtypes"]
    if kind == "BALL OUT":
        return event["end"]
    if kind == "SHOT" and (subtypes & {"GOAL", "OUT"}):
        return event["end"]
    if kind == "FAULT RECEIVED":
        return event["start"]
    if kind == "CHALLENGE" and "FAULT" in subtypes and "ADVANTAGE" not in subtypes:
        return event["start"]
    if kind == "BALL LOST" and (subtypes & {"FORCED", "OFFSIDE", "END HALF"}):
        return event["end"]
    return None


def _metrica_is_restart(event):
    return event["type"] == "SET PIECE" or "GOAL KICK" in event["subtypes"]


def derive_metrica_labels(events, n_frames, first_frame):
    """Per-frame codes from Metrica events, an approximation of the competition convention.

    A team keeps possession until an opponent makes an on-ball event, so contested phases stay
    with the last team. The ball is dead from a stoppage event to the next on-ball event.
    """
    codes = np.full(n_frames, X, dtype=np.int8)

    def fill(start, stop, code):
        if code is not None and stop > start:
            codes[max(0, start - first_frame):max(0, stop - first_frame)] = code

    for period in sorted({event["period"] for event in events}):
        period_events = sorted((e for e in events if e["period"] == period), key=lambda e: e["start"])
        owner, cursor, dead_since = None, period_events[0]["start"], None
        previous_end = cursor
        for index, event in enumerate(period_events):
            on_ball = event["type"] in METRICA_ON_BALL
            # A restart without a recorded stoppage (a saved shot that goes out) means the ball was
            # dead from the end of the previous event.
            if index and _metrica_is_restart(event) and dead_since is None:
                dead_since = max(previous_end, cursor)
            if on_ball and dead_since is not None:
                # Game 3 records a CARRY that starts on the same frame as the BALL OUT; such an
                # event belongs to the play that just ended and is not the restart.
                if event["start"] >= dead_since:
                    fill(cursor, dead_since, owner)
                    fill(dead_since, event["start"], D)
                    cursor, dead_since, owner = event["start"], None, event["team"]
            elif on_ball and event["team"] != owner:
                fill(cursor, event["start"], owner)
                cursor, owner = event["start"], event["team"]
            stop = _metrica_stop_frame(event)
            if stop is not None and dead_since is None:
                dead_since = max(stop, cursor)
            previous_end = max(previous_end, event["end"])
        last = previous_end + 1
        if dead_since is None:
            fill(cursor, last, owner)
        else:
            fill(cursor, dead_since, owner)
            fill(dead_since, last, D)
    return codes


def _metrica_csv_game(folder, game):
    events = []
    with open(os.path.join(folder, "Sample_Game_%d_RawEventsData.csv" % game)) as handle:
        for row in csv.DictReader(handle):
            events.append({"team": H if row["Team"] == "Home" else A, "type": row["Type"],
                           "subtypes": set(filter(None, row["Subtype"].split("-"))),
                           "start": int(row["Start Frame"]), "end": int(row["End Frame"]),
                           "period": int(row["Period"])})
    tables = [np.genfromtxt(os.path.join(folder, "Sample_Game_%d_RawTrackingData_%s_Team.csv" % (game, side)),
                            delimiter=",", skip_header=3, dtype=np.float64) for side in ("Home", "Away")]
    frame = tables[0][:, 1].astype(np.int64)
    period = tables[0][:, 0].astype(np.int64)
    pitch = np.array(METRICA_PITCH_M)
    ball = tables[0][:, -2:] * pitch
    blocks = [t[:, 3:-2].reshape(len(t), -1, 2) * pitch for t in tables]
    players = np.concatenate(blocks, axis=1)
    slot_team = np.concatenate([np.full(b.shape[1], code, dtype=np.int8) for b, code in zip(blocks, (H, A))])
    return events, frame, period, players, slot_team, ball


def _metrica_epts_game(folder):
    """Sample game 3: FIFA EPTS tracking (the competition ATD layout) and JSON events."""
    root = ElementTree.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    team_code = {team.get("id"): H if team.findtext("Name") == "Team A" else A for team in root.iter("Team")}
    player_team = {p.get("id"): team_code[p.get("teamId")] for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    player_ids = sorted(player_team)
    column_of = {player: index for index, player in enumerate(player_ids)}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        columns = [column_of[channel_player[refs[k]]] for k in range(0, len(refs), 2)]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), np.array(columns)))
    with open(os.path.join(folder, "Sample_Game_3_tracking.txt")) as handle:
        lines = handle.read().splitlines()
    frame = np.empty(len(lines), dtype=np.int64)
    players = np.full((len(lines), len(player_ids), 2), np.nan)
    ball = np.full((len(lines), 2), np.nan)
    for row, line in enumerate(lines):
        frame_text, player_text, ball_text = line.split(":")
        frame[row] = int(frame_text)
        columns = next(c for start, stop, c in specs if start <= frame[row] <= stop)
        values = np.array([float(v) for group in player_text.split(";") for v in group.split(",")])
        players[row, columns] = values.reshape(-1, 2)
        ball[row] = [float(v) for v in ball_text.split(",")[:2]]
    pitch = np.array(METRICA_PITCH_M)
    players *= pitch
    ball *= pitch
    slot_team = np.array([player_team[p] for p in player_ids], dtype=np.int8)
    with open(os.path.join(folder, "Sample_Game_3_events.json")) as handle:
        raw = json.load(handle)["data"]
    events = []
    for event in raw:
        subtypes = event["subtypes"] or []
        subtypes = subtypes if isinstance(subtypes, list) else [subtypes]
        events.append({"team": H if (event["team"] or {}).get("name") == "Team A" else A,
                       "type": event["type"]["name"], "subtypes": {s["name"] for s in subtypes},
                       "start": int(event["start"]["frame"]), "end": int(event["end"]["frame"]),
                       "period": int(event["period"])})
    period = np.zeros(len(frame), dtype=np.int64)
    for value in sorted({e["period"] for e in events}):
        inside = [e for e in events if e["period"] == value]
        low, high = min(e["start"] for e in inside), max(e["end"] for e in inside)
        period[(frame >= low) & (frame <= high)] = value
    return events, frame, period, players, slot_team, ball


def read_metrica(folder, game):
    """Halves of Metrica sample game 1, 2 or 3."""
    events, frame, period, players, slot_team, ball = (
        _metrica_epts_game(folder) if game == 3 else _metrica_csv_game(folder, game))
    if not np.array_equal(frame, np.arange(frame[0], frame[0] + len(frame))):
        raise ValueError("tracking frames are not contiguous; the labels index by frame offset")
    codes = derive_metrica_labels(events, len(frame), int(frame[0]))
    halves = [{"codes": codes[period == p], "players_m": players[period == p], "ball_m": ball[period == p]}
              for p in sorted(set(period.tolist()) - {0})]
    return halves, slot_team, METRICA_PITCH_M
