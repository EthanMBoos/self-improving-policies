# self-improving-policies

Can a VLM grade the robot? We use vision-language models as reward models for self-improving robot policies:
rejection-sampling fine-tuning of π0.5 in LIBERO, where the pass/fail label on each rollout comes from a VLM
instead of the simulator.

This branch covers the first pipeline step: getting the π0.5 checkpoints from
[openpi](https://github.com/Physical-Intelligence/openpi) running in the LIBERO simulator and measuring baselines.

## Baseline results

Evaluated on one RTX 5090 with openpi's LIBERO eval client (`examples/libero/main.py`, seed 7, replan every 5 steps).

| Model | Trials/task | Spatial | Object | Goal | LIBERO-10 | Average |
|---|---|---|---|---|---|---|
| π0.5-LIBERO (`pi05_libero`) | 10 | 100.0 | 100.0 | 97.0 | 92.0 | **97.2** |
| π0.5-LIBERO, published by openpi | 50 | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |
| π0.5-base (`pi05_base`), zero-shot | 2 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |

Each suite has 10 tasks, so 10 trials/task = 100 episodes per suite (roughly ±3 points). Per-task breakdowns and
raw logs are in [results/libero_eval/](results/libero_eval/).

- **π0.5-LIBERO reproduces the published numbers.** Its weakest tasks are "put both moka pots on the stove" (6/10)
  and "open the top drawer and put the bowl inside" (8/10).
- **π0.5-base gets 0% zero-shot.** The arm moves but drifts away from the objects. It was never trained on LIBERO's
  action space, so this is expected.

![pi05_base zero-shot (top) vs pi05_libero (bottom) on "pick up the black bowl from table center and place it on the plate"](results/libero_eval/base_vs_libero_frames.png)

*Same task. Top: π0.5-base zero-shot fails. Bottom: π0.5-LIBERO succeeds. Frames are rotated 180°, which is how
the model sees them.*

**What this means for the project:** rejection-sampling fine-tuning needs a starting policy that succeeds some of
the time. π0.5-LIBERO (~97%) has almost no headroom, and π0.5-base (0%) produces no successes to keep. We need an
intermediate starting point, e.g. `pi05_base` fine-tuned on a subset of the LIBERO demos or for fewer steps, or
π0.5-LIBERO evaluated on harder or held-out tasks (LIBERO-90, perturbed scenes).

## Reproducing

### Requirements

- Linux with an NVIDIA GPU. Tested on an RTX 5090 (32 GB), driver 580, CUDA 13.0. The policy needs well under
  16 GB, but JAX preallocates 75% of GPU memory by default.
- [`uv`](https://docs.astral.sh/uv/) and `git`.
- About 40 GB of disk: 25 GB checkpoints, 12 GB Python environments.

### 1. Clone

```bash
git clone -b stratos https://github.com/EthanMBoos/self-improving-policies.git
cd self-improving-policies
```

### 2. Download the checkpoints (~25 GB)

```bash
uv run scripts/download_checkpoints.py
```

This downloads `pi05_base` and `pi05_libero` from openpi's public bucket `gs://openpi-assets/checkpoints/` into
`checkpoints/`, which is gitignored, and verifies every file's checksum. No Google Cloud account is needed. Re-running
skips files that are already valid. Use `--out <dir>` to store them elsewhere, e.g. on scratch storage on PACE.

The checkpoints are JAX/orbax format. openpi has a script to convert them to PyTorch if we need that later.

### 3. Set up openpi and LIBERO

```bash
scripts/setup_openpi.sh
```

This clones openpi (pinned commit) into `third_party/openpi` and creates two environments:

| Environment | Python | Role |
|---|---|---|
| `third_party/openpi/.venv` | 3.11, JAX | **Policy server**: loads π0.5 on the GPU and serves actions over a local websocket (port 8000) |
| `third_party/openpi/examples/libero/.venv` | 3.8 | **Eval client**: runs the LIBERO simulator, queries the server, records success/failure |

They are separate processes because LIBERO/robosuite are pinned to Python 3.8 while openpi needs 3.11. The script
also writes a project-local LIBERO config to `third_party/libero_config/`. The eval script points LIBERO at it with
`LIBERO_CONFIG_PATH`, so any existing `~/.libero` from another project is left alone.

### 4. Evaluate π0.5-LIBERO

Terminal 1 (policy server):

```bash
cd third_party/openpi
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config pi05_libero --policy.dir ../../checkpoints/pi05_libero
```

Wait for `server listening on 0.0.0.0:8000`. Then in terminal 2, from the repo root:

```bash
scripts/run_libero_eval.sh outputs/libero_eval/pi05_libero_10trials 10
```

This runs all four suites in parallel against the server and prints the summary table, which is also written to
`<out_dir>/summary.txt`. Arguments are `<out_dir> [trials_per_task=10] [suites...]`. For example,
`scripts/run_libero_eval.sh outputs/test 1 libero_spatial` is a quick 10-episode smoke test.

Runtime on the RTX 5090: about 30 minutes at 10 trials/task. LIBERO-10 is the bottleneck, at around 19 s per
episode. The full protocol of 50 trials/task takes about 2.5–3 hours.

### 5. Evaluate π0.5-base zero-shot

Stop the first server, then in terminal 1 from the repo root:

```bash
third_party/openpi/.venv/bin/python scripts/serve_pi05_base_zeroshot.py
```

Terminal 2:

```bash
scripts/run_libero_eval.sh outputs/libero_eval/pi05_base_zeroshot_2trials 2
```

`pi05_base` ships no LIBERO normalization stats, so this server combines the base weights with the `pi05_libero`
data pipeline and the LIBERO norm stats bundled with the `pi05_libero` checkpoint. It keeps π0.5's default
`discrete_state_input=True` (robot state as text tokens), which is how the base model was pretrained. The
`pi05_libero` fine-tune turns that off.

### Troubleshooting

- **`EGLError` tracebacks in the client logs** at the end of each task are harmless. They come from robosuite
  tearing down its render context. If rendering itself fails, set `MUJOCO_GL=glx`.
- **GPU memory:** JAX reserves 75% of the GPU at startup. To share the GPU, start the server with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` (or similar).
- **Videos:** the client saves one rollout video per task and outcome to `<out_dir>/videos_<suite>/`. Later
  episodes overwrite earlier ones. `outputs/` is gitignored.

## Repository layout

```
scripts/
  download_checkpoints.py     download + verify pi05 checkpoints from gs://openpi-assets
  setup_openpi.sh             clone openpi, build server/client envs, write LIBERO config
  run_libero_eval.sh          run the LIBERO eval client on the four suites in parallel
  serve_pi05_base_zeroshot.py serve pi05_base with the LIBERO pipeline (zero-shot)
  summarize_libero.py         per-task / per-suite success rates from client logs
results/libero_eval/          summaries + logs of the runs above
checkpoints/                  (gitignored) model weights
third_party/                  (gitignored) openpi + LIBERO + environments
outputs/                      (gitignored) eval logs and videos
```
