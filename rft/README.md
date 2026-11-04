# The RFT loop

Rejection-sampling fine-tuning of pi0.5 on one held-out LIBERO-90 task, run twice: with the simulator as the judge
(control) and with Robometer-4B as the judge (experiment). The simulator scores every episode in both; in the
experiment it is logged and never used to select. Everything pre-registered is in `configs/rft/<task>.json`.

## Pieces

| File | Runs in | Does |
|---|---|---|
| `scripts/rollout_worker.py` | LIBERO client env (Python 3.8) | N attempts of one task against the policy server, each from a seeded random layout; writes frames, videos, and `episodes_extra.jsonl` |
| `scripts/rollout_and_log.py` | openpi env | Launches the worker, writes `metrics.json` (true success with 95% interval), optionally packs a LeRobot dataset |
| `rft/lerobot_pack.py` | openpi env | Raw episodes to a LeRobot v2.1 dataset in openpi's LIBERO schema plus a `sim_success` column and the sidecar |
| `scripts/finetune_round.py` | openpi env | Full fine-tune of a checkpoint on a dataset with the round config; runs openpi's own trainer |
| `grader/programs/robometer_grade.py` | Robometer env | The grader program (from branch `stratos`); last-frame success probability per video |
| `grader/robometer.py` | any | Grades a rollout directory through the program above and writes `grader_*` fields into the sidecar |
| `grader/audit.py` | any | Confusion matrix against the simulator at several thresholds, the D5 verdict, false positives to watch |
| `rft/select.py` | any | The only place a verdict becomes training data; one field per judge |
| `scripts/loop.py` | any (orchestration only) | Rounds: rollouts, grade, select, pack, fine-tune, exam, `metrics.json`; resumable |
| `analysis/plot_curves.py` | openpi env | True success per round for both loops plus judged success |

## Setup on sci-ai (once)

```
source /sci-ai-pool/data/shared/slee3321/env.sh      # WORK, caches, CUDA_VISIBLE_DEVICES
cd $WORK/self-improving-policies
scripts/setup_openpi.sh                               # done already
uv run scripts/download_checkpoints.py                # done already
scripts/setup_robometer.sh                            # grader env + Robometer-4B; add --in-domain for the second grader
```

Port 8000 is taken on sci-ai; the configs use 8001. Nothing else may be using the card while a round trains.

## Smoke test (minutes, before any real round)

```
third_party/openpi/.venv/bin/python scripts/loop.py --config configs/rft/middle_bowl.json --judge sim --dry-run
cp configs/rft/middle_bowl.json configs/rft/smoke.json   # then edit: run_name smoke, rollouts_per_round 12, min_kept 2, exam_episodes 10, train.steps 20, train.warmup_steps 2, train.save_interval 20
python3 scripts/loop.py --config configs/rft/smoke.json --judge sim --rounds 1
```

A passing smoke test shows: a round_0 exam metrics.json, 12 logged rollouts with videos and `frames.npz`, `kept.json`,
a packed `lerobot/rft/train` dataset, a checkpoint directory with `params/` and `assets/`, and a round_1 exam.
Then grade and audit the smoke rollouts to check the grader end to end:

```
python3 -c "from grader.robometer import grade_rollouts; from rft.config import LoopConfig; import pathlib; grade_rollouts(pathlib.Path('runs/smoke/sim/round_1/rollouts'), LoopConfig.load('configs/rft/smoke.json').grader)"
python3 grader/audit.py runs/smoke/sim/round_1/rollouts --config configs/rft/smoke.json
```

## The real runs

```
tmux new -s rft
python3 scripts/loop.py --config configs/rft/middle_bowl.json --judge sim --audit-with robometer   # control, graded for the audit
python3 grader/audit.py runs/middle_bowl/sim/round_1/rollouts --config configs/rft/middle_bowl.json   # D5 gate
python3 scripts/loop.py --config configs/rft/middle_bowl.json --judge robometer                     # experiment
third_party/openpi/.venv/bin/python analysis/plot_curves.py runs/middle_bowl
```

The control loop must show rising true success before the experiment loop runs (non-negotiable 4), and the audit's
precision at 0.65 must be at least 0.5 (D5). Re-running a loop command resumes at the first unfinished stage.

## Round layout

`runs/<run_name>/<judge>/round_k/` holds the attempts made by the round k-1 checkpoint (`rollouts/`), the judge's
selection (`kept.json`), the kept episodes as a dataset (`lerobot/rft/train`, plus `rft/all` for every attempt),
the round k checkpoint (`checkpoint/pi05_rft/round_k/<step>`), the round k exam (`exam/`), and `metrics.json`.
`round_0/` holds only the base checkpoint's exam. So `metrics.json` in `round_k` describes the brain after k
fine-tunes, and the plot's x axis is k.

## Numbers that must not be hardcoded

Rounds, rollouts per round, minimum kept, exam size, seeds, the grader threshold, and the round's training settings
all live in the config. `tests/` checks that seeds never collide and that selection reads only the judge's field:

```
python3 -m unittest discover -s tests
```
