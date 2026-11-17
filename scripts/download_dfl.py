"""Restore only explicit checksum-pinned DFL files, never Metrica or competition assets."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from possession import paths
from scripts.download_external import CHECKSUMS, download, sha256

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


def selected_files(matches):
    expected = {}
    for line in CHECKSUMS.read_text().splitlines():
        digest, relative = line.split()
        if relative.startswith("dfl/") and any(relative.endswith("DFL-MAT-" + match + ".xml") for match in matches):
            expected[relative] = digest
    if len(expected) != 2 * len(set(matches)):
        raise ValueError("expected exactly two pinned files per DFL match")
    return expected


def restore(destination, matches):
    destination = destination.resolve()
    for repository in (ROOT, paths.COMPETITION_REPO.resolve()):
        if destination == repository or destination.is_relative_to(repository) or repository.is_relative_to(destination):
            raise ValueError("destination must be a dedicated directory outside both repositories")
    expected = selected_files(matches)
    with urllib.request.urlopen(ARTICLE, timeout=60) as response:
        article = json.load(response)
    if article["license"]["url"].rstrip("/") != LICENSE.rstrip("/") or article["version"] != 1:
        raise PermissionError("dataset version or CC BY 4.0 license verification failed")
    files = {item["name"]: item for item in article["files"]}
    if not all(Path(relative).name in files for relative in expected):
        raise ValueError("pinned DFL files absent from official article")
    destination.mkdir(parents=True, exist_ok=True)
    attribution = ("# DFL dataset attribution\n\nManuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert. "
                   "An integrated dataset of spatiotemporal and event data in elite soccer. "
                   "Dataset DOI: https://doi.org/10.6084/m9.figshare.28196177.v1 ; "
                   "paper DOI: https://doi.org/10.1038/s41597-025-04505-y .\n\n"
                   "Official DFL/Sportec data; licensed CC BY 4.0: " + LICENSE + " .\n\n"
                   "Downloaded source files are unchanged. Evaluation derives metric coordinates, "
                   "event/status-based labels, a two-second possession rule, and model probabilities. "
                   "No endorsement by the authors or provider is implied.\n")
    (destination / "DFL_ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
    provenance_path = destination / "DFL_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {
        "article": ARTICLE, "doi": article["doi"], "version": 1, "license": article["license"],
        "authors": [a["full_name"] for a in article["authors"]], "files": {}}
    for relative, digest in expected.items():
        info = files[Path(relative).name]
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("invalid destination")
        download(info["download_url"], target, digest)
        provenance["files"][relative] = {"source_url": info["download_url"], "figshare_file_id": info["id"],
            "size": target.stat().st_size, "sha256": sha256(target), "dataset_match": next(m for m in matches if relative.endswith("DFL-MAT-" + m + ".xml"))}
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"restoration": "PASS", "files_verified": len(expected), "provider": "DFL", "license": "CC BY 4.0"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--matches", choices=MATCHES, nargs="+", required=True)
    args = parser.parse_args()
    try:
        restore(args.destination, args.matches)
    except Exception as error:
        print(json.dumps({"restoration": "FAIL", "error_category": type(error).__name__}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
