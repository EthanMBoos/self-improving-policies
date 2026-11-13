# Does Robometer grade better in its training orientation?

**No clear difference.** Robometer judged LIBERO videos about as well in the stored view as in the mirrored view it
was trained on. The orientation mismatch does not explain why it was too optimistic in the RL runs.

## The question

The RL runs found Robometer about 2× too optimistic: it passed about half of all failed episodes. The RL worker saves
the upright view the policy sees. Robometer was trained on LIBERO frames mirrored left-right from that view (found on
branch `stan`, commit 97c235f). If the mismatch confused it, the "too optimistic" result would be partly a bug.

## What was run

Robometer-4B graded the 96 labelled rollout videos already in the repo, in four orientations, on one H200 on PACE ICE,
2026-11-13. The videos are the π0.5 baseline evals and the LIBERO-90 candidate clips. Each file name carries LIBERO's
own success label. The "hard subset" leaves out the zero-shot base model, whose failures are easy because its arm
drifts away from the objects. It has 44 successes and 12 near-miss failures.

Orientations, relative to the stored upright video: `none` is as stored, `lr` is mirrored left-right (Robometer's
training view), `ud` is upside down, and `rot180` is the raw render.

To reproduce: `sbatch scripts/rl/pace/regrade_committed.sbatch`, see [docs/regrade-orientation.md](../../docs/regrade-orientation.md).

## Results

Full tables in [comparison.txt](comparison.txt). Hard subset, threshold 0.5:

| | Stored (`none`) | Mirrored (`lr`) | Upside down (`ud`) | Raw (`rot180`) |
|---|---|---|---|---|
| AUROC | 0.66 | 0.63 | 0.48 | 0.57 |
| Failures passed | 7 of 12 | 5 of 12 | 6 of 12 | 7 of 12 |
| Successes caught | 38 of 44 | 42 of 44 | 24 of 44 | 30 of 44 |

- **Mirrored and stored are about the same.** AUROC is 0.66 against 0.63 on the hard subset and 0.90 for both on all
  96 videos. The mirrored view passes 2 fewer failures, which is within noise with 12 failures.
- **The test can detect an orientation problem.** Upside-down frames clearly hurt: AUROC drops to 0.72 on all videos
  and 0.48 on the hard subset, which is chance level. A left-right mirror does not.
- **Robometer misses near-misses in every orientation.** It passes about half of the hard failures, consistent with
  the RL runs, where it passed about half of all failures.

## Caveats

- Only 12 hard failures, so one episode moves the false-positive rate by 8 points. This rules out a large orientation
  effect, not a small one.
- These are fine-tuned π0.5 rollouts, not the RL policy's episodes on LIBERO-10. Re-grading the RL episodes themselves
  would give a firmer number; `scripts/rl/pace/regrade.sbatch` does that once the recorded episodes are available.
