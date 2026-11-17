# Visca Barca: submitted baseline technical report

This report describes source baseline `190fbdefcec376f674727a194cd8ac34315a7d70` and the
twelve shipped pretrained artifacts. It does not describe later temporal-isolation experiments.
No new inference, training or competition evaluation was run while preparing this publication.

## Inputs and preprocessing

At prediction time the baseline loads only the authorized tactical ATD TXT and original XML
slot-team metadata. The organizer source provides pitch dimensions, official half boundaries
and label definitions. Competition ground truth, videos, broadcast tracking, homography and
Smart Tagging are not prediction inputs. Coordinates become metres, missing values remain NaN,
and player gaps up to ten seconds are linearly interpolated before positions outside halves are
masked. Identity continuity inside a player slot is an assumption.

External training loaders use Metrica sample games 1-3 and seven DFL matches, with event-derived
Metrica labels and DFL provider possession/status labels. The two-second control rule is applied
to external reference labels. External halves have a 1,000-frame separator. Data is not shipped;
source URLs and checksums are in the baseline downloader and external checksum manifest.

## Classifiers and inference

LightGBM uses 300 trees, learning rate 0.05, 31 leaves, minimum child samples 200, seed 0
and eight training jobs. Features cover collective/player movement, team shape, ball proximity,
touch timing and ball advance at multiple centered scales. Dead-ball models use player signals.

TCNs operate at five steps per second with 64 channels and dilations 1 through 128. They use
normalized signals plus observation masks, residual convolutions, dropout 0.1 during training,
Adam and 1,500 training updates. Training seeds are 0/1/2. Team-swapped examples and averaged
swapped inference enforce symmetry. External augmentation simulates missing ball/player tracks.

P(dead) averages the clean and player-dropout LightGBM/TCN pairs, four classifiers in total.
P(home | live) averages the clean and player-dropout pairs of LightGBM and a three-seed TCN mean,
eight classifiers in total. Prediction loads pretrained states on CPU, uses evaluation mode and
no gradients, and does no fitting, calibration or tuning.

Three emissions combine these probabilities into home, away and dead. Full-timeline Viterbi
decoding charges 2.5 nats for team changes and 20 nats for live/dead changes. The two-second
control rule follows; the output writer masks all ungraded frames to X. The decoder is not
independently reset for each half or valid span. Interpolation, centered features, TCN context
and global decoding can carry boundary effects. This is an offline baseline, not a causal system.

## Evidence and limitations

The independently verified result is exact artifact reproduction: fresh baseline inference took
244.98 s on Windows CPU and matched the preserved Visca Barca CSV by complete SHA-256 and
binary comparison. Both files were 1,882,093 bytes. The official validator passed with exit 0.
Their SHA-256 is `f6f402152dfe300986243f5af15cae06ca0fd66f3239fbf02a9787574edfeaef`.
All twelve checkpoint hashes matched before and after. This demonstrates repeatability and CSV
format validity, not correctness of possession labels or generalization performance.

The historical README reported spellF1 0.816/0.792 for Metrica and 0.769/0.752 for DFL
(clean/player-dropout). Missing experiment scripts and per-fold records mean nested selection,
held-out exclusion, exact metric invocation and aggregate calculations remain unverified.
The historical experiment log is reported development history, not independent evidence.
The shipped models were fitted on the external training corpus: scoring those same matches with
these weights is **in-sample diagnostic only**. No such result is claimed as independent held-out
performance, and this publication reports no new competition accuracy measurements.

Provider label conventions, approximate event timing, tracking loss, slot identity changes,
interpolation, centered context and the control rule limit accuracy. Cache source provenance is
not validated. Identical retraining across hardware is not established. The original artifact's
full creation environment is not independently known. See [reproduction](reproduction.md) for
the verified Windows environment, lockfile, historical test-hash caveat and checkout safeguards.

## Provenance and distribution

Source code, models, tests and locked dependencies are unchanged from the baseline. This
standalone repository has fresh Git history and includes no experimental branches or source.
DFL provenance is Bassek et al. (2025), Scientific Data 12:195,
https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0 as documented by the project).
Metrica source is https://github.com/metrica-sports/sample-data; provider attribution and
responsible-use requirements apply, and blanket redistribution rights are not established here.
No external or competition datasets, predictions, private logs or credentials are distributed.
No competition ground-truth evaluation or manual prediction correction is part of this preparation.
