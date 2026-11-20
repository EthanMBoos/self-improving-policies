# Experiment code (work in progress)

Code for the experiments in [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md), run on PACE ICE. Status as of 2026-11-20.

## Pipeline

1. `rollouts.py` + `pace/rollouts.sbatch`: run pi0.5 (`pi05_libero`) in LIBERO and save every episode (both cameras,
   state, actions, object positions and goal predicates at start and end).
2. `grade.py` + `pace/grade.sbatch`: score each episode with a grader (Robometer, RoboReward prompt, Qwen yes/no).
3. `analyze.py`: grader vs. LIBERO success. Thresholds picked on a validation split (hash of the episode name),
   metrics on test. Also near-miss vs. obvious failures, overlap of passed failures, majority vote.
4. `features.py` + `probe.py`: Experiment 2, linear probes on frozen SigLIP/DINOv2 features.
5. `sheets.py`: contact sheets for looking at episodes.

Run jobs with `pace/submit.sh <script> <sbatch options>` and watch them from a laptop with `pace/watch.sh <job ids>`.

## Where things stand

- Step 0 checks pass: LIBERO-Spatial 99/100 (openpi reports 98.8%), LIBERO-10 95% (openpi reports 92.4%).
  The policy samples actions, so the same seed and init reproduce the outcome but not the exact trajectory.
- Pool `pools/main`: LIBERO-90, inits 0-4. 328 episodes graded (tasks 0-67), 33% success. Tasks 60-89 are being
  finished.
- Interim results on those 328 (237 test episodes):

  | Grader | AUROC (95% CI) | FPR at 90% recall |
  |---|---|---|
  | Robometer LIBERO fine-tune, true view (trained on these tasks) | 0.96 (0.94-0.98) | 0.20 |
  | Robometer-4B general, true view | 0.89 (0.85-0.93) | 0.75 |
  | Robometer-4B general, mirrored view | 0.85 (0.80-0.90) | 0.35 |
  | RoboReward-4B | 0.84 (0.78-0.90) | 0.68 |
  | Qwen3-VL-4B yes/no, last frame of both cameras | 0.77 (0.71-0.83) | 0.52 |
  | Qwen3-VL-4B with RoboReward's prompt | 0.64 (0.56-0.71) | 0.80 |

  Different graders pass mostly different failures (Jaccard 0.16-0.64); 4 of 153 test failures pass all of them.

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
- **LIBERO prints an info line on import**, so anything parsing its stdout takes `tail -1`.

## Next

Grade tasks 60-89, run the Experiment 2 probes, full Experiment 4 tables, look at shared false positives.
Experiment 1 (fine-tuning) only if time allows.
