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

## The LIBERO fine-tune of Robometer on the same videos

Same 96 videos, same job, graded by `jesbu1/robometer-4b-fft-libero`, Robometer-4B fine-tuned on all five LIBERO
suites (2026-11-18, one H200). Full tables in [comparison_libero_finetune.txt](comparison_libero_finetune.txt). Hard
subset, threshold 0.5:

| | General, stored | General, mirrored | LIBERO fine-tune, stored | LIBERO fine-tune, mirrored |
|---|---|---|---|---|
| AUROC | 0.66 | 0.63 | 0.86 | 0.87 |
| Failures passed | 7 of 12 | 5 of 12 | 1 of 12 | 2 of 12 |
| Successes caught | 38 of 44 | 42 of 44 | 27 of 44 | 39 of 44 |

- **The fine-tune catches the near-misses the general model passes with confidence.** The general model scored five
  hard failures at 0.9 or more in both orientations: a drawer opened but apparently not the middle one, a mug put in
  the microwave with the door left open, both moka pots crowded on the burner, the right moka pot, and the mug and
  pudding task. The fine-tune rejects four of the five in either orientation.
- **The fine-tune does care about orientation.** In the stored view it rejects 17 of 44 successes, against 5 of 44
  in its mirrored training view. The general model barely notices the mirror; the fine-tune learned LIBERO's look
  closely, including which way it faces.
- **It was trained on these scenes.** Its training data covers all five LIBERO suites, LIBERO-90 included, so this is
  an in-distribution test, not a general one.
- **This does not contradict SeungTaek's loop result.** On the bowl task, the fine-tune ranked attempts well (AUROC
  0.90) but still approved about half the failures at 0.65, and the policy got worse the same way
  ([rft/README.md](../../rft/README.md), branch `stan`). Those failures are clean deliveries of the wrong identical
  bowl, harder than the near-misses here.

## Caveats

- Only 12 hard failures, so one episode moves the false-positive rate by 8 points. A bootstrap 95% interval for the
  hard-subset AUROC difference, mirrored minus stored, is -0.18 to +0.13 for the general model. This rules out a large
  orientation effect, not a small one.
- The openpi eval client keeps the last episode per task and outcome, so the videos are not a random sample.
- These are fine-tuned π0.5 rollouts, not the RL policy's episodes on LIBERO-10. Re-grading the RL episodes themselves
  would give a firmer number; `scripts/rl/pace/regrade.sbatch` does that once the recorded episodes are available.
