"""BAR-ATH constants and asset paths, taken from the organisers' ``games.py``.

The challenge repository (github.com/metrica-sports/more-than-a-hack-2027, MIT licence,
commit f8f2629ff1e0985f81d7e59d8413678514ee1530) owns these constants. They are copied here
so that prediction needs nothing but this repository and the match data in ``data/``. See
``docs/third-party.md`` for the full MIT notice; the data is separately licensed.

The data root defaults to ``<repo>/data``. Point ``MTAH_DATA_DIR`` somewhere else to use a
different bundle location; nothing below opens a file, so a missing bundle is reported by the
caller with a normal ``FileNotFoundError``.
"""
from pathlib import Path

from . import paths

DATA_DIR = paths.DATA_DIR

GAME = "BAR-ATH"

# The tactical recording is the frame-index authority: row i of a submission is video frame i.
GAMES = {
    "BAR-ATH": {
        "n_video_frames": 199319,
        "fps": 25,
        # id_frame = video_frame + offsets[half], measured frame by frame against the video.
        "offsets": {1: 1702057, 2: 1702058},
        # IdFrame span of each half in the tracking data (the only gap is halftime).
        "halves": {1: (1713242, 1792708), 2: (1812866, 1892255)},
        "pitch": (105.3, 68.0),  # metres
    },
}

SOURCES = {
    "tactical": {
        "stem": "BAR-ATH_tactical",
        "n_video_frames": 199319,
        "assets": ("video", "atd_meta", "atd_raw", "homography", "smart_tagging"),
    },
    "broadcast": {
        "stem": "BAR-ATH_broadcast",
        "n_video_frames": 164488,
        "assets": ("video", "atd_meta", "atd_raw", "homography"),
    },
}
DEFAULT_SOURCE = "tactical"

SUFFIXES = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}


def asset_paths(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Every provided file for one source of a game, keyed by asset, under ``DATA_DIR``.

    The bundle is one folder per recording: ``data/tactical`` and ``data/broadcast``. The
    broadcast recording is optional and is not synchronised to the tactical one.
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    if source not in SOURCES:
        raise KeyError("no such source {!r}; have {}".format(source, sorted(SOURCES)))
    if game not in GAMES:
        raise KeyError("no such game {!r}; have {}".format(game, sorted(GAMES)))
    entry = SOURCES[source]
    folder = root / source
    return {name: _resolve(folder / (entry["stem"] + SUFFIXES[name])) for name in entry["assets"]}


def _resolve(path):
    """The canonical file, or a single download variant such as ``..._tactical-002.mp4``.

    The tracking files always carry the canonical names. A video downloaded from the bundle may
    keep a ``-001``/``-002`` suffix; accepting it here keeps ``asset_paths()["video"]`` usable.
    """
    if path.is_file():
        return path
    variants = sorted(path.parent.glob(path.stem + "-*" + path.suffix))
    return variants[0] if len(variants) == 1 else path


def halves(game=GAME):
    """[(half, first_id_frame, last_id_frame)] in play order."""
    table = GAMES[game]["halves"]
    return [(half, table[half][0], table[half][1]) for half in sorted(table)]


def half_video_blocks(game=GAME, adjust=0):
    """[(half, first_video_frame, last_video_frame)], each half mapped by its own offset.

    For BAR-ATH this is video 11185..90651 and 110808..190197. The two blocks are far apart,
    so a video frame belongs to at most one of them.
    """
    blocks = []
    for half, first_id, last_id in halves(game):
        offset = GAMES[game]["offsets"][half] + adjust
        blocks.append((half, first_id - offset, last_id - offset))
    return blocks
