# Experiment code (work in progress)

Code for the experiments in [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md), run on PACE ICE. Status as of 2026-11-20.

## Pipeline

1. `rollouts.py` + `pace/rollouts.sbatch`: run pi0.5 (`pi05_libero`) in LIBERO and save every episode (both cameras,
   state, actions, object positions, drawer/door/knob joint positions and goal predicates at start and end).
2. `grade.py` + `pace/grade.sbatch`: score each episode with a grader (Robometer, RoboReward prompt, Qwen yes/no).
3. `analyze.py`: grader vs. LIBERO success. Thresholds picked on a validation split (hash of the episode name),
   metrics on test. Also near-miss vs. obvious failures, overlap of passed failures, majority vote.
4. `features.py` + `probe.py` (`pace/features.sbatch`, `pace/probe.sbatch`): Experiment 2, linear probes on frozen
   SigLIP/DINOv2 features.
5. `replay_joints.py` (`pace/replay_joints.sbatch`): backfills joint positions for episodes recorded before
   `rollouts.py` saved them, by replaying the saved actions without rendering.
6. `sheets.py`: contact sheets for looking at episodes.

Run jobs with `pace/submit.sh <script> <sbatch options>` and watch them from a laptop with `pace/watch.sh <job ids>`.

## Where things stand

- Step 0 checks pass: LIBERO-Spatial 99/100 (openpi reports 98.8%), LIBERO-10 95% (openpi reports 92.4%).
  The policy samples actions, so the same seed and init reproduce the outcome but not the exact trajectory.
- Pool `pools/main`: LIBERO-90, 90 tasks x inits 0-4 = 450 episodes, 28% success. Of 324 failures, 302 are
  near-misses (an object moved, a drawer/door/knob moved, or part of the goal holds) and 22 are obvious.
- Experiments 4 and 2 are done on this pool. Experiment 1 (fine-tuning) has not started.

### Experiment 4: graders vs. LIBERO success (317 test episodes)

| Grader | AUROC (95% CI) | FPR at 90% recall |
|---|---|---|
| Robometer LIBERO fine-tune, true view (trained on these tasks) | 0.94 (0.91-0.96) | 0.29 |
| Robometer LIBERO fine-tune, mirrored view | 0.89 (0.84-0.93) | 0.31 |
| RoboReward-4B | 0.85 (0.80-0.90) | 0.70 |
| Robometer-4B general, true view | 0.84 (0.79-0.88) | 0.79 |
| Robometer-4B general, mirrored view | 0.82 (0.77-0.87) | 0.42 |
| Qwen3-VL-4B yes/no, last frame of both cameras | 0.72 (0.66-0.78) | 0.61 |
| Qwen3-VL-4B yes/no, 16-frame video | 0.70 (0.64-0.76) | 0.57 |
| Qwen3-VL-4B yes/no, last frame | 0.70 (0.63-0.76) | 0.67 |
| Qwen3-VL-4B with RoboReward's prompt | 0.61 (0.54-0.68) | 0.80 |

- No grader is safe alone at high recall: the best general one still passes 42% of failures at 90% recall.
- RoboReward's training matters: 0.85 vs. 0.61 for its base model with the same prompt.
- Graders pass mostly different failures (pairwise Jaccard 0.15-0.70). Rank-averaging all nine gives 0.87,
  majority vote passes 31% of failures at 96% recall; neither beats the best single grader.
- 7 of 220 test failures pass every grader, all near-misses: three "open the top drawer" episodes where the robot
  opened the middle drawer (16 cm) and left the top one shut, three books put in the caddy but not its front
  compartment, one bowl left half on the plate's rim.

### Experiment 2: linear probe on frozen features (AUROC)

| | Seen tasks (317 test) | Unseen tasks (135 test) |
|---|---|---|
| SigLIP probe, 25 labels | 0.66 | 0.65 |
| SigLIP probe, 100 labels | 0.87 | 0.71 |
| SigLIP probe, all labels (133 seen / 315 unseen) | 0.91 | 0.74 |
| DINOv2 probe, all labels | 0.88 | 0.76 |
| Shuffled labels (control) | 0.46-0.57 | 0.49-0.57 |
| Robometer LIBERO fine-tune | 0.94 | 0.95 |
| Robometer-4B general | 0.84 | 0.86 |
| RoboReward-4B | 0.85 | 0.74 |

- On seen tasks about 100 labels beats every zero-shot grader. On unseen tasks the probe stalls around 0.75 and the
  general Robometer is better. First + last frames beat last frame alone by up to 0.07.
- "Unseen" means unseen by the probe; the LIBERO fine-tuned Robometer was trained on all LIBERO tasks.

### Caveats

- One policy, one benchmark; test set of 97 successes / 220 failures (CIs about +-0.05).
- The joint backfill replay reproduces the recorded outcome for 425 of 450 episodes; 24 of the 25 mismatches are
  recorded successes, which the near-miss tagging does not use.
- SigLIP's text encoder takes 16 tokens, so the longest task descriptions are truncated in the probe features.

## Things that bit us, and the workaround

- **Image orientation.** Raw robosuite frames are upside down. The true camera view is `raw[::-1]`, which is what
  Robometer's LIBERO loader uses. openpi feeds the policy `raw[::-1, ::-1]`, a left-right mirror. `grade.py --view`
  picks one; it changes the general Robometer's AUROC by about 0.04.
- **Simulators starting at once hang.** Several processes creating EGL contexts on one GPU at the same time deadlock
  (1 starts in 9 s, 8 together never do). `rollouts.py` creates and closes envs under a file lock.
- **Model compile vs. rendering.** Compiling pi0.5 while simulators render on the same GPU stalled for over 10 minutes.
  `rollouts.sbatch` sends the server two dummy requests (compile takes ~20 s) before starting any simulator.
- **Websocket keepalive.** openpi runs inference inside the server's event loop, so it can't answer pings while
  compiling and connections drop with "1011 keepalive ping timeout". `serve.py` starts the server with pings off.
- **JAX takes all GPU memory** by default and starves EGL rendering: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.
- **Bad nodes.** Some PACE nodes have a GPU CUDA or EGL can't use. Jobs check first, append to
  `$EXP/node_health.tsv`, and exit fast; `submit.sh` excludes every node marked FAIL.
- **`--exclude` ignored.** Without a partition PACE spreads the job over three partitions and replaces the exclude
  list. Scripts set `-p coe-gpu`.
- **Scratch quota (300 GB, 1M files).** A full disk killed jobs mid-write. Jobs and `submit.sh` warn at 85% and refuse
  at 95%; rollouts check every 10 minutes. An episode is about 47 MB.
- **Robometer environment.** No committed lockfile upstream, so pin: torch 2.8.0+cu128, transformers 4.57.6,
  unsloth and unsloth_zoo 2026.11.4, no torchao, `use_unsloth: false` in the checkpoint config (`pace/infer_views.sh`),
  bf16 autocast.
- **SigLIP tokenizer needs protobuf >= 5**, but the Robometer env pins 3.20, so features run in a separate
  `$EXP/feat-venv` (torch 2.8.0+cu128, transformers 4.57.6, sentencepiece).
- **Near-misses on drawers looked like obvious failures** because only free objects were tracked. Fixture joints are
  now saved and counted (31 of 53 "obvious" failures were near-misses).
- **LIBERO prints an info line on import**, so anything parsing its stdout takes `tail -1`.

## Next

Experiment 1 (fine-tune on episodes kept by each grader, measure the effect of false positives), roughly 1-2 days of
GPU time. Optional: a grader aimed at near-misses (e.g. comparing start and end states), probes on a second suite.
