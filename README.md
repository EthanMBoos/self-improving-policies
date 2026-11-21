# Can a VLM Grade the Robot?

A robot policy can improve itself by fine-tuning on its own successful attempts, but something has to decide which
attempts succeeded. We test whether vision-language model (VLM) graders can do that, using pi0.5 in LIBERO, where the
simulator's success check gives ground truth.

Findings so far:

- **Off-the-shelf graders pass many failures**, almost all of them near-misses (the wrong drawer opened, a bowl on the
  plate's rim). The best general grader has AUROC 0.84 but passes 42-79% of failures at 90% recall.
- **What hurts the policy is passing failures, not rejecting successes.** Fine-tuning on true successes gains 7 points;
  mixing in failures erases that, while dropping half the successes costs about 1 point.
- **A real grader's damage matches random errors at the same rates**, so the number to watch is the share of failures
  in the data it keeps (1 - precision), not its accuracy.
- **A cheap fix looks promising:** a linear probe on frozen image features, trained on about 100 labelled rollouts,
  keeps a set with 9% failures where the general VLM grader keeps 23-30%. Fine-tunes on it are running.

## Layout

| Path | What |
|---|---|
| [`exp/`](exp/) | All experiment code and PACE job scripts. **Start at [`exp/README.md`](exp/README.md)** for the run order. |
| [`paper/`](paper/) | RA-L manuscript draft (`tectonic main.tex`), literature notes in `paper/lit/`. |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | The experiment plan. |
| [`docs/LITERATURE.md`](docs/LITERATURE.md) | Review of VLMs as robot reward models and success detectors, 2023-2026. |

Results data (rollouts, scores, checkpoints) stays on PACE under `~/scratch/ethan-exp`.
