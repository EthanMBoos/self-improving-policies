# Re-grade the RL episodes with the frame orientation fixed

The RL runs found Robometer about 2× too optimistic: it passed about half of all failures. But the RL worker records
the upright view the policy sees, and Robometer was trained on LIBERO frames mirrored left-right from that view
(SeungTaek found this on branch `stan`, commit 97c235f). This job re-grades the recorded episodes in each orientation
and compares each against LIBERO's own success check. It needs no simulator and no training, only Robometer on a GPU.

If the mirrored orientation (`lr`) agrees with LIBERO much better than the stored one (`none`), part of the "too
optimistic" finding was a preprocessing bug. If all orientations score about the same, the finding stands.

## Quick check first: the videos already in git

This needs nothing from stratosreppas. The repo already has 96 labelled rollout videos in the same upright view: the
π0.5 baseline evals and SeungTaek's LIBERO-90 candidate clips. The outcome is in each file name.

| Source | Successes | Failures |
|---|---|---|
| Fine-tuned π0.5 on the four LIBERO suites | 40 | 7 |
| LIBERO-90 candidate clips | 4 | 5 |
| Base π0.5, zero-shot | 0 | 40 |

Do the one-time setup in step 2 below, then submit from the login node:

```bash
cd ~/scratch/self-improving-policies
sbatch scripts/rl/pace/regrade_committed.sbatch
```

It grades all 96 videos in four orientations, which should take minutes. Results are in
`outputs/rl/offline/committed/comparison.txt`, as two tables:

- **All videos.** The 40 base-model failures are easy: its arm drifts away from the objects. Any orientation should
  reject them, so this table flatters every row.
- **Hard subset.** Without the base model, 44 successes and 12 near-miss failures. This is the one to read. Compare the
  false-positive rate and AUROC of `lr`, Robometer's training view, against `none`, the stored view the RL runs used.

With only 12 hard failures, one episode moves the false-positive rate by 8 points. A big gap between `lr` and `none`
is a strong hint. A small one means nothing either way, and the full re-grade below settles it.

## What runs where

| Step | Where | Time |
|---|---|---|
| 1. Get the recorded episodes from stratosreppas | their machine | depends on upload |
| 2. Set up the repo and Robometer | PACE ICE, interactive GPU job | about 20 min, once |
| 3. Pilot: 1,000 episodes × 4 orientations | PACE ICE, batch job | about 1 hour, a guess |
| 4. Full re-grade in the winning orientation | PACE ICE, batch job | a few hours per run |

Any ICE GPU with CUDA 13 works. Robometer-4B needs about 13 GB at batch 16, so an H100 is not required. An A100, L40S,
H100 or H200 are all fine. The job script asks for any CUDA 13 GPU, which excludes the 16 GB V100s.

## 1. Get the recorded episodes

The episodes are not in git. Ask stratosreppas to run this in `outputs/rl/runs/` on their machine. It keeps only what
grading needs, the episode lists and third-person videos, and leaves out wrist videos and simulator states:

```bash
tar czf rl_runs_for_regrade.tgz --exclude=wrist.mp4 --exclude=final_state.npy --exclude=meta.json \
  oracle/episodes.jsonl oracle/goal_report.jsonl oracle/episodes \
  robometer_soft/episodes.jsonl robometer_soft/goal_report.jsonl robometer_soft/episodes \
  robometer_bin/episodes.jsonl robometer_bin/goal_report.jsonl robometer_bin/episodes
```

`tar` warns if a `goal_report.jsonl` is missing but still writes the archive. Copy it to `~/scratch` on ICE, for example
with `scp rl_runs_for_regrade.tgz YOUR_USERNAME@login-ice.pace.gatech.edu:scratch/`, then unpack it on ICE:

```bash
mkdir -p ~/scratch/rl_runs && tar xzf ~/scratch/rl_runs_for_regrade.tgz -C ~/scratch/rl_runs
```

The episode lists hold absolute paths from stratosreppas's machine. `grade_offline.py` notices that the videos are not
there and rewrites the paths onto the unpacked run directory.

## 2. Set up on PACE ICE

Log in from the campus network or the [Georgia Tech VPN](https://vpn.gatech.edu/):

```bash
ssh YOUR_USERNAME@login-ice.pace.gatech.edu
```

Clone into scratch, since the home quota is too small for the 9 GB model and the Python environment. Then install `uv`
if it is missing:

```bash
cd ~/scratch
git clone -b regrade-orientation https://github.com/EthanMBoos/self-improving-policies.git
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

Do the setup on a GPU node, so the final CUDA check means something:

```bash
srun --gres=gpu:1 --constraint=cuda13 --cpus-per-task=8 --mem=48G --time=01:00:00 --pty bash
source ~/scratch/self-improving-policies/scripts/rl/pace/env.sh
cd "$SIP_REPO"
scripts/setup_robometer.sh
exit
```

The setup ends with `robometer ok, cuda True`. It puts Robometer's code and environment in `third_party/robometer/` and
the weights in `checkpoints/Robometer-4B-infer`. `exit` releases the GPU.

## 3. Pilot: which orientation does Robometer grade best?

From the login node, submit the pilot on the oracle run. Its episodes were never shaped by Robometer's reward, so they
are the fairest test:

```bash
cd ~/scratch/self-improving-policies
sbatch --export=ALL,RUN=$HOME/scratch/rl_runs/oracle scripts/rl/pace/regrade.sbatch
squeue -u $USER
```

The log goes to `regrade-<jobid>.out` in the directory you submitted from. Each orientation grades the same 1,000
random episodes. The last lines are a table like this one, also saved as `comparison.txt` under
`outputs/rl/offline/oracle_n1000/`:

```text
1000 shared episodes, true success 27.0%, threshold 0.5

variant                                   says ok   prec  recall    FPR  AUROC  best bal acc
--------------------------------------------------------------------------------------------
none (flip none)                            ...
lr (flip lr)                                ...
ud (flip ud)                                ...
rot180 (flip rot180)                        ...
```

How to read it:

- **`says ok` against true success** is the optimism. The RL runs had about 60% against about 27%.
- **AUROC** is the cleanest single number, since it does not depend on the threshold. The RL runs had 0.73 to 0.77.
- **`lr` is the orientation Robometer was trained on.** If it clearly beats `none` on AUROC and false-positive rate,
  the orientation bug explains part of the result.
- **`rot180` is the raw render**, which is what Robometer's own online LIBERO wrapper feeds it. `ud` is there in case
  the RL videos turn out not to be upright.

The `none` row should land close to the RL README's offline numbers for the oracle run. If it is far off, check the
setup before trusting the other rows.

## 4. Full re-grade in the winning orientation

Re-grade every episode of each run, with `none` as the baseline and the winner, here `lr`:

```bash
for run in oracle robometer_soft robometer_bin; do
  sbatch --export=ALL,RUN=$HOME/scratch/rl_runs/$run,SAMPLE=0,FLIPS="none lr" scripts/rl/pace/regrade.sbatch
done
```

For the two Robometer runs the table gets an extra row: the scores Robometer gave during training. The script also
checks that the offline `none` grading reproduces them.

For the full plots and false-positive gallery of a re-grade, run the existing analysis on it:

```bash
uv run --no-project --with numpy --with matplotlib --with imageio python \
  scripts/rl/analyze_grader.py outputs/rl/offline/oracle_n0/lr
```

## Copy the results home

On your own computer:

```bash
scp -r YOUR_USERNAME@login-ice.pace.gatech.edu:scratch/self-improving-policies/outputs/rl/offline ~/Downloads/
```

Scratch is not backed up and old files get cleared, so copy anything you want to keep.

## What changed in the code

- `scripts/rl/graders/robometer_grade.py` takes `--flip none|lr|ud|rot180`. The default is `none`, so the RL
  pipeline behaves as before.
- `scripts/rl/grade_offline.py` takes `--flip`, plus `--sample N --seed S` for a pilot subset. It rewrites video paths
  when a run was copied from another machine.
- `scripts/rl/compare_orientations.py` prints the comparison table. `--source` restricts it to a subset.
- `scripts/rl/videos_to_episodes.py` turns labelled rollout videos into an episode list for `grade_offline.py`.
- `scripts/rl/pace/` holds the PACE environment file and the batch job. `scripts/setup_robometer.sh` and the LIBERO-90
  candidate clips come from branch `stan` unchanged.
