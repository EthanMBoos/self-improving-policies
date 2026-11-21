# Experiment code

Code for the experiments in [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md), run on Georgia Tech's PACE ICE cluster.
Everything heavy runs as a Slurm job; nothing runs on the login node.

## Conventions

- Code is synced to `$CODE = ~/scratch/ethan-exp/code` and data lives in `$EXP = ~/scratch/ethan-exp` (set by
  `pace/env.sh`). Job scripts are submitted from `$EXP`, and paths like `pools/main` are relative to it.
- Submit with `pace/submit.sh <script> <sbatch options>`. It checks scratch space and skips nodes logged as broken in
  `$EXP/node_health.tsv`. Options go before the script.
- Watch jobs from a laptop with `pace/watch.sh <job ids>`. It exits and prints a summary when a job ends.
- A pool is a directory of episodes: `pools/<name>/episodes/<name>/{meta.json, scene.json, episode.npz, *.mp4}`, plus
  `episodes.jsonl` (one line per episode) and `scores/<grader>.jsonl` (one line per episode per grader).

## Run order

| Step | Run | Produces |
|---|---|---|
| 0. Setup, once | `sbatch pace/setup.sbatch`, `sbatch pace/setup_graders.sbatch`, then `pace/infer_views.sh` | openpi and LIBERO envs, grader env and checkpoints |
| 1. Roll out the policy | `pace/rollouts.sbatch` (runs `serve.py` + `rollouts.py`) | `pools/<name>/episodes` |
| 2. Grade the episodes | `pace/grade.sbatch` (runs `grade.py`) | `pools/<name>/scores/*.jsonl` |
| 3. Experiment 4: graders vs. ground truth | `analyze.py` | grader table (AUROC, FPR, near-miss breakdown) |
| 4. Experiment 2: linear probes | `pace/features.sbatch` (`features.py`), then `pace/probe.sbatch` (`probe.py`) | `features_*.npz`, probe table |
| 5. Experiment 1: fine-tune on graded data | `exp1_select.py`, then `pace/exp1_train.sbatch`, then `pace/rollouts.sbatch` with `CKPT` set | dataset lists, checkpoints, evaluation pool |
| 6. Part 2: high-precision graders | `phase_a.py`, then the same train and evaluation jobs | precision table, dataset lists |

Example commands, run from `$EXP` on the login node (each submits a job):

```bash
S=code/pace/submit.sh
# 1. 19 tasks x inits 10-39 with the base pi0.5 checkpoint
$S code/pace/rollouts.sbatch --export=ALL,OUT=pools/exp1_train,SUITE=libero_90,TASKS="0 2 12",INITS="$(seq -s ' ' 10 39)"
# 2. score them with Robometer-4B, true camera view
$S code/pace/grade.sbatch --export=ALL,POOL=pools/exp1_train,SETTINGS="robometer:$EXP/ckpt/Robometer-4B:true:agent"
# 3. compare every scored grader against LIBERO's success check (CPU, quick)
robometer/.venv/bin/python code/analyze.py --pool pools/main pools/main/scores/*.jsonl
# 5. build the dataset lists, fine-tune one, evaluate it on held-out inits
robometer/.venv/bin/python code/exp1_select.py --pool pools/exp1_train --scores <train scores> \
    --calib-pool pools/main --calib-scores <main scores> --out exp1/sets
$S code/pace/exp1_train.sbatch --export=ALL,RUN=clean            # optional: SEED=43,EXP_NAME=clean_s43
$S code/pace/rollouts.sbatch --export=ALL,OUT=pools/exp1_eval,SUITE=libero_90,TASKS="...",INITS="$(cat exp1/eval_inits.txt)",\
    TAG=clean,CONFIG=pi05_libero_lora,CKPT=$EXP/checkpoints/pi05_libero_lora/clean/1999,NOFRAMES=1
```

Each job script lists its settings at the top. `rollouts.sbatch` checks that the GPU can render before starting
and resubmits itself elsewhere if it can't. `exp1_train.sbatch` builds its dataset on the node's local disk from
`exp1/sets/<RUN>.txt`.

## Files

| File | Role |
|---|---|
| `rollouts.py`, `serve.py` | LIBERO client that runs and saves episodes; openpi policy server with keepalive pings off |
| `egl_probe.py` | Render check that `rollouts.sbatch` runs before starting |
| `grade.py` | Robometer, RoboReward-prompt and Qwen yes/no graders |
| `analyze.py` | Thresholds on validation, metrics on test, near-miss tagging, grader overlap |
| `features.py`, `probe.py` | Frozen SigLIP/DINOv2 features; linear probes vs. number of labels |
| `exp1_select.py` | Experiment 1 datasets (Clean, FP/FN sweep, real grader, matched random, strict grader) |
| `phase_a.py` | Part 2: failure share of each grader and selection rule; writes dataset lists |
| `make_dataset.py`, `exp1_config.py`, `train.py` | LeRobot dataset builder, LoRA training config, openpi training entry point |
| `replay_joints.py`, `sheets.py` | One-off tools: backfill fixture joints for old episodes; contact sheets for viewing episodes |
| `pace/env.sh`, `submit.sh`, `watch.sh` | Paths and checks shared by all jobs; submit wrapper; job watcher |

## Results so far

- Step 0 checks pass: LIBERO-Spatial 99/100 (openpi reports 98.8%), LIBERO-10 95% (openpi reports 92.4%).
  The policy samples actions, so the same seed and init reproduce the outcome but not the exact trajectory.
- Pool `pools/main`: LIBERO-90, 90 tasks x inits 0-4 = 450 episodes, 28% success. Of 324 failures, 302 are
  near-misses (an object moved, a drawer/door/knob moved, or part of the goal holds) and 22 are obvious.
- Experiments 4 and 2 are done on this pool. Experiment 1's first round (base, Clean, FP 30, real grader, matched)
  is done on a separate training pool.

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

### Experiment 1, first round: fine-tune on what each grader keeps

Training pool `pools/exp1_train`: the 19 LIBERO-90 tasks the base policy solves sometimes (1-4 of 5 in `pools/main`),
inits 10-39, 570 episodes, 52% success; 264 of 273 failures are near-misses. The real grader is the general
Robometer-4B (true view) at the balanced-accuracy threshold from `pools/main` validation: on this pool it passes 35%
of failures and drops 26% of successes (AUROC 0.77). Matched keeps the same number of successes and failures per
task as the real grader, chosen at random.

| Dataset | Episodes | Frames | Failures in it (episodes) |
|---|---|---|---|
| Clean (true successes) | 297 | 67.5k | 0% |
| FP 30 (plus 30% of failures) | 379 | 100.3k | 22% |
| Real grader | 315 | 92.7k | 30% |
| Matched | 315 | 93.2k | 30% |

Failures run to the 400-step limit, so their share of frames is far larger than their share of episodes.

Every run: `pi05_libero_lora` (LoRA on both Gemma models, from the `pi05_libero` checkpoint), batch 32, 2000 steps,
cosine LR 5e-5, one seed. About 47 min on an H200 (108 GB of GPU memory). Evaluation: the same 19 tasks x held-out
inits 5-9 and 40-49 = 285 episodes per policy, LIBERO's check as judge. CIs are bootstrap over paired episodes.

| Policy | Success | vs. base |
|---|---|---|
| Base pi0.5 | 147/285 = 51.6% | |
| Clean | 168/285 = 58.9% | +7.4 (+1.4 to +13.0) |
| Real grader | 150/285 = 52.6% | +1.1 (-4.9 to +7.0) |
| Matched | 150/285 = 52.6% | +1.1 (-4.6 to +6.7) |
| FP 30 | 138/285 = 48.4% | -3.2 (-9.1 to +2.5) |

- Clean beats base, so the pipeline works and there is room to improve.
- Clean beats the real grader by 6.3 points (+1.1 to +11.9): the grader's mistakes erase the gain.
- Real grader and matched tie exactly (0.0, -5.6 to +5.3): this round, the number of mistakes mattered, not which
  episodes. Per-task differences go both ways (task 79: real grader 9/15 vs. matched 2/15; task 2: 7 vs. 11).
- One seed per dataset; a gap of about 6 points is the smallest this setup resolves.

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
- **EGL fails on some GPUs, not nodes.** About 1 in 3 rollout jobs gets a GPU where EGL falls back to software
  ("Cannot initialize a EGL device display"); the same node works on its next job. Render failures are logged as
  "bad" (not FAIL) and the job resubmits itself elsewhere, up to 4 attempts. Every retry so far passed on attempt 2.
- **Data loader workers re-run the script.** openpi's loader uses spawn, so `train.py` needs a `__main__` guard or
  it hangs.
- **The datasets library copies every training dataset into its cache** (~9 GB each, on scratch by default).
  `exp1_train.sbatch` points `HF_DATASETS_CACHE` at the node's local disk.
- **Checkpoints with optimizer state are ~9 GB.** Training saves only the final step and deletes `train_state`
  afterwards; serving needs only `params` and `assets`.
- **Reading npz frames one at a time** decompresses the whole array each time (1 min per episode). Load each array once.
- **One H200 ran at 89 C with clocks at 450 MHz**, so its fine-tune took 2 h 17 min instead of 47 min. Same result,
  just slower.
- **LIBERO prints an info line on import**, so anything parsing its stdout takes `tail -1`.

