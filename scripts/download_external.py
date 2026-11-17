"""Download the external training matches and check each file against scripts/external_sha256.txt.

    python scripts/download_external.py

- Sportec Open DFL dataset (Bassek et al. 2025, Scientific Data 12:195, CC BY 4.0), figshare
  article 28196177: match information and raw positions of 7 matches, about 2.6 GB.
- Metrica Sports sample data, games 1-3 (github.com/metrica-sports/sample-data), at a fixed commit.
Files that exist and match their checksum are not downloaded again.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from possession import paths  # noqa: E402

FIGSHARE_FILES = "https://api.figshare.com/v2/articles/28196177/files?page_size=100"
METRICA_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/e706dd506b360d69d9d123d5b8026e7294b13996/data/"
CHECKSUMS = Path(__file__).with_name("external_sha256.txt")
CHUNK_BYTES = 1 << 20


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected):
    if path.is_file() and sha256(path) == expected:
        print("ok      ", path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    print("download", path.name, flush=True)
    with urllib.request.urlopen(url) as response, open(temporary, "wb") as handle:
        for block in iter(lambda: response.read(CHUNK_BYTES), b""):
            handle.write(block)
    if sha256(temporary) != expected:
        temporary.unlink()
        raise RuntimeError("checksum of {} differs from {}".format(path.name, CHECKSUMS.name))
    temporary.replace(path)


def main():
    expected = {}
    for line in CHECKSUMS.read_text().splitlines():
        digest, relative = line.split()
        expected[relative] = digest
    with urllib.request.urlopen(FIGSHARE_FILES) as response:
        figshare_url = {item["name"]: item["download_url"] for item in json.load(response)}
    for relative, digest in expected.items():
        folder, name = relative.split("/")
        if folder == "dfl":
            url = figshare_url[name]
        else:
            url = METRICA_RAW + "Sample_Game_{}/{}".format(name.split("_")[2], name)
        download(url, paths.EXTERNAL_DIR / folder / name, digest)


if __name__ == "__main__":
    main()
