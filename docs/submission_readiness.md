# Final submission readiness audit

17 November 2026. Documentation/source inspection only in this audit. The validated CSV was not
opened, hashed, changed, regenerated or transmitted. Runtime PASS results below are prior local
sanitized audit evidence, not new match-data checks.

## Official checklist

References are to the inspected organizer checkout; current official README, SUBMISSION,
EVALUATION and RULES documents were also available online.

| Requirement | Status | Evidence / remaining action |
|---|---|---|
| CSV with exact header, 199319 rows, ordered 0..199318, permitted nonblank labels | PASS | Prior official validator and local structural checks; `SUBMISSION.md:18-26` |
| Exact final upload artifact validated | NOT VERIFIED | Working CSV passed previously; packaging is not done; `SUBMISSION.md:99-101` |
| Team-named CSV | FAIL | Registered team name confirmed by user: Visca Barca. Existing filename is `submission.csv`; prepare `Visca Barca.csv` without changing its bytes; `SUBMISSION.md:94-95` |
| Public repository accessible to organizers | NOT VERIFIED | Local origin URL is configured; public fetch could not establish accessibility; `SUBMISSION.md:51-57` |
| Source, 12 models, pinned dependencies and execution instructions | PASS | Tracked source/models/lockfile; updated README and reproduction guide, pending publication; `SUBMISSION.md:53-54` |
| Same-file regeneration | NOT VERIFIED | One successful inference, no second-run byte comparison; historical test hash not checked; `RULES.md`, rule 3 |
| Short technical document, PDF or Markdown, all required topics | PASS | `technical_report.md`; source-supported method, provenance, limitations, reproduction and evidence caveats; `SUBMISSION.md:62-73` |
| Technical-document final filename uses team name | NOT VERIFIED | Deliver `technical_report.md` as `Visca Barca.md`; final packaging not performed; `SUBMISSION.md:94-95` |
| No restricted data or reconstructive exports in public repo | PASS for current tracked-file inventory | No such tracked paths; all-history review and enrolled terms remain NOT VERIFIED; `RULES.md`, The data |
| Algorithm-generated labels; no hand labels/corrections on provided game | PASS for inspected prediction path | Existing predictor and prior successful run; complete historical conduct NOT VERIFIED; `RULES.md`, rule 1 |
| No private GT/human-produced match data; no public-GT training/tuning | PASS for inspected training/prediction path | External loaders only; complete historical conduct NOT VERIFIED; `RULES.md`, rule 2 |
| External sources accessible and documented | PASS for documented sources; full terms NOT VERIFIED | Public provider URLs, pinned Metrica commit, manifest; attribution in technical report; `RULES.md`, What you may use |
| Enrollment and accepted dataset terms | NOT VERIFIED | Requires user confirmation; `README.md:97-104`, `RULES.md`, The data |
| Registered team identity and 3-5 member eligibility | NOT VERIFIED | User confirmed team name Visca Barca; registration and member-count evidence not inspected; `RULES.md`, Eligibility and teams |
| Google account and emailed submission-form access | NOT VERIFIED | Requires user action; `SUBMISSION.md:77-98` |
| Form includes team name, CSV, public URL, technical document | NOT VERIFIED | Not submitted by this audit; `SUBMISSION.md:83-88` |
| One response per team; correct latest response | NOT VERIFIED | Requires user action; `SUBMISSION.md:97-98` |
| Submission before 17 November 2026 deadline | NOT VERIFIED | Date documented; exact cutoff/timezone and completed submission not established; `SUBMISSION.md:90` |

The public rule page permits some activities that the user's instructions prohibit here. This
audit follows the stricter user constraints: no provided-data access, AI uploads, video comparison,
competition-ground-truth evaluation, or new inference. Enrollment terms are not available here.

## Repository evidence and limits

All package modules, scripts, tests, twelve model files, checksum manifest and dependency manifests
are tracked in the audited checkout (HEAD `190fbde`). `.gitattributes` is still untracked from the
authorized CRLF fix and must be included in any future publication. New documentation likewise
requires a later user-authorized commit/push. No such operation was performed.

Tracked filenames contain no provided-data files, prediction CSVs, virtual environment, private-key
files or `.env` files. This is not a complete secret scan or Git-history certification. Current
working-tree status includes the four previously repaired model paths; their bytes match the
manifest and indexed originals. Publication must include the attribute rule to prevent recurrence.

Deterministic settings are implemented, but historical rebuild claims are not independent evidence
for this exact Windows CSV. Installation succeeded previously in the existing environment; a
fresh-machine `uv sync --locked` has not been repeated. See `reproduction.md` for local-only
second-run hash comparison and cache isolation.

## Delivery decision

The existing CSV is format-ready on prior evidence. The technical document is prepared, but its
team-name packaging is outstanding. The repository has adequate local inference materials; public
availability, publication of the fixes/docs, and exact byte reproduction remain unverified. External
score reproduction is optional evidence work, not a reason to delay an otherwise valid submission.
