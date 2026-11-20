# Three side experiments, built from scratch

## Ground rules

- **Work on a fresh `ethan` branch cut from `main`:**

  ```bash
  git checkout main && git pull
  git checkout -b ethan
  ```

- **Don't build on the other branches.** Their scripts and numbers are useful
  pointers, but treat them as claims to re-check:
  - Robometer passing 58.5% of failures
  - AUROC 0.66 vs 0.63 on the orientation check
  - AUROC 0.87 for the LIBERO fine-tune
  - the wrong-bowl hack under PPO

  If they hold up in your own setup, good. If they don't, that's a finding too.
- **Build the smallest thing that works,** using the official repos directly:
  openpi for π0.5, LIBERO, and each grader's own release.
- **Check every piece by eye before trusting it.** Watch videos, look at frames,
  and spot-check labels. Most wrong results in this kind of work come from a
  silent pipeline bug, not from the model.
- **Write down every choice** (seed, checkpoint, camera, image orientation,
  threshold), so you can tell later which ones mattered.

---

## Step 0: build and check the pieces

### Rollouts

Write one short script that loads a π0.5 checkpoint with openpi, runs it in
LIBERO, and saves the following for each episode:

- the third-person and wrist videos
- the task text
- the seed
- LIBERO's success flag at the end
- the observations, robot state and executed actions at every step, in the
  format openpi trains on. Experiment 1 fine-tunes on these, so save them now
  rather than collecting everything again later.

Check before trusting:

- **Labels:** watch about 20 saved videos, half labelled success and half failure,
  and confirm the label matches what you see. Note any case where LIBERO's check
  itself looks wrong. Your whole setup treats it as ground truth, so you need to
  know how good it is.
- **Images:** look at a raw frame and a saved frame side by side, so you know the
  orientation and colour channels yourself. Don't rely on anyone's note about
  flips.
- **Baseline success rate:** re-run a few episodes with the same seed and check
  that you get the same outcome, or at least know how much it varies. Then compare
  the success rate to what openpi reports for that checkpoint. A large gap means
  something is set up wrong.

### The pool

Fine-tuned π0.5 succeeds about 97% of the time on the standard LIBERO suites. At
that rate almost every episode succeeds, so grader mistakes barely matter. Run a
quick 50-episode check to find a setting in the 30–60% range. Candidates:

- LIBERO-90
- a less-trained checkpoint
- a task subset where it struggles

Check which suites each grader was trained on before choosing. The current
Robometer LIBERO fine-tune includes LIBERO-90, so on LIBERO-90 it is being
tested on tasks it has seen. Either pick tasks no grader trained on, or label
those results "seen tasks".

**What the failures look like matters as much as how many there are.** A failure
where the arm drifts away is easy for every grader. The interesting failures are
near-misses from a competent policy. In an earlier check on committed videos, one
grader's AUROC dropped from 0.96 to 0.86 once a weak policy's drifting failures
were removed. Tag each failure (near-miss or obvious) and report grader numbers
on the near-miss subset as well as overall.

Start with a few hundred episodes across several tasks that each have both
successes and failures. Grow the pool (for the 1,000-episode probe or for
fine-tuning) only once the pipeline gives useful numbers. All three experiments
use this one pool.

### Splits and thresholds

- Split by whole episode, never by frame.
- Hold out a small validation set. Choose each grader's pass threshold on it,
  freeze the threshold, then report on the test set only.
- Report AUROC, false-positive rate, success recall, and precision on the
  accepted episodes (the share of passed episodes that really succeeded). That
  last number is how clean a filtered training set would be.

### Graders

Install each grader fresh from its official release, then hand-pick about 10
episodes:

- 3 obvious successes
- 3 obvious failures, such as the arm wandering off
- 4 near-misses you have watched yourself

Run each grader on these first. If it can't separate the obvious cases, assume
the setup is wrong until you've ruled that out: check orientation, camera,
frame sampling, prompt and preprocessing against the official example. In an
earlier check, mirroring the frames moved one grader's recall from 0.61 to 0.89
on the same videos. Only once the integration matches the official example is a
miss on obvious cases a finding about the model.

---

## 1. Which grader mistake hurts more: passing failures or rejecting successes?

**Question.** When you fine-tune only on episodes judged successful, a false
positive puts a failed episode into the training set labelled as a success. A
false negative only throws away a good episode. Theory from LLM work says false
positives should hurt more (arXiv 2411.17501), but nobody has measured it for
robots.

**Setup.** Use LIBERO's own success check as the grader, and corrupt it on
purpose. Every dataset comes from the same pool, so you roll out once, then only
fine-tune and evaluate.

| Dataset | What gets kept |
|---|---|
| No fine-tune | Nothing. The starting checkpoint, as a baseline |
| Clean | True successes only |
| FP 10 / 30 / 50 | Successes, plus 10 / 30 / 50% of failures relabelled as success |
| FN 10 / 30 / 50 | Successes, with 10 / 30 / 50% of them dropped |
| Real grader | Whatever a VLM grader passes, at its measured false-positive rate |
| Random flips matched to the real grader | Random flips at the real grader's false-positive *and* false-negative rates |

**Start small.** First run No fine-tune, Clean, one noisy condition and the real
grader. Run the full sweep only if those results are interesting.

**Fine-tuning.** Use openpi's supported LoRA training path (the JAX configs).
openpi's PyTorch implementation doesn't support LoRA, so don't write a trainer.
Use the same starting checkpoint, step count and settings for every row.

- **First check: does Clean beat No fine-tune?** This can fail for real reasons:
  an already strong policy may not improve on its own successes. Rule out a
  pipeline bug first. If Clean still doesn't help, the setting has no room to
  improve, and noise has nothing to erode. Move to harder tasks or a weaker
  checkpoint rather than running the rest of the table.

**The real-grader row.** VLM mistakes are not random; they cluster on
near-misses. Compare the real-grader policy with random flips matched to *both*
its false-positive and false-negative rates, keeping the task mix roughly the
same. Matching only the false-positive rate would change two things at once.

- **If the real-grader policy is worse:** structured mistakes do more damage than
  random ones.
- **If the two match:** the error rate alone tells the story.

**Control for dataset size.** Dropping successes shrinks the data. Either cap every
dataset at the same number of episodes, or report the size next to each point.
For every dataset, report its size, its task mix, and the share of failures in
it.

**Measure.** Evaluate each policy on LIBERO, with the oracle as the judge. Plot:

- policy success against the false-positive rate
- policy success against the false-negative rate
- the real-grader run on the same plot

**Use enough evaluation episodes to see a difference.** A 3–5 point gap in
success rate needs hundreds of evaluation episodes per policy to stand out from
noise. Use several seeds and report confidence intervals. Without them, the
comparison will most likely come out as noise.

**Cost.** One rollout pass for the pool, plus 4 fine-tunes to start (about 10
for the full sweep) and their evaluations.

---

## 2. How much of your own experience beats a big general grader?

**Question.** The leading labs train graders on their own robot data rather than
using big general models. How many labelled episodes of your own does a tiny
grader need to beat a 4B general one?

**Setup.**

- Run a frozen image encoder over every episode in the pool. SigLIP is a natural
  choice, since it's π0.5's own vision encoder. For a second data point, try an
  unrelated encoder such as DINOv2.
- For each episode, take features for the first frame and last frame from both
  the third-person and wrist cameras, plus a text embedding of the task.
- Train a logistic regression on top to predict LIBERO's success label.
- Train on 50, 200 and 1,000 episodes.

**Test splits.**

- **Held-out episodes on the same tasks.** This is the easy case. In simulation
  the renders look the same from run to run, so expect the probe to do well.
- **Held-out tasks.** Hide some tasks completely during training. This is the real
  test, because it asks whether the probe learned "success" or just memorised each
  task's end scene.

**Compare against:**

- the general VLM grader or graders from Experiment 4
- a probe trained on shuffled labels, as a floor
- a "last frame only" probe, to see whether the start frame adds anything

Report AUROC and the false-positive rate at 90% recall.

**Things to try once it works:**

- a tiny MLP instead of logistic regression
- adding middle frames
- wrist camera only vs third-person only, especially on near-misses
- giving the probe the difference between first-frame and last-frame features

**Cost.** Small. Feature extraction is one GPU job, and training takes minutes on
a CPU.

---

## 4. Do different graders fail on the same episodes?

**Question.** A 2026 paper argues that averaging several graders can't remove
mistakes they all share (arXiv 2608.08002). If every grader passes the same
near-misses, adding graders won't help, and you need more evidence instead, such
as more camera views or more frames.

**Graders (all free, open weights, run on PACE).** Pin the exact checkpoint for
each one, and use the official inference code where a release has it.

| Grader | Type | Notes |
|---|---|---|
| Robometer-4B (general) | Trained, dense progress plus success | Install fresh from the official release |
| Robometer LIBERO fine-tune | Trained, in-domain | Same; check what data it saw |
| RoboReward 4B or 8B | Trained, end-of-episode 1–5 score | Check that the weights are public on HuggingFace |
| Qwen3-VL-4B base | Zero-shot yes/no | Robometer-4B's base model. Check RoboReward's base and run that too if it differs |
| TOPReward-style | Zero-shot, reads token probabilities | Use the probability of "yes" instead of the text answer. Reported useful zero-shot (arXiv 2602.19313), so treat it as a real baseline |
| GVL-style | Zero-shot, shuffled-frame progress | Run with Qwen3-VL instead of GPT |
| Your probe from Exp 2 | Trained on your own data | For comparison |

The three zero-shot graders are just different prompts on one Qwen3-VL model, so
they share one script. For these, read each paper's method section and write
your own version rather than reusing anyone's wrapper code.

**Measure.**

- A pass/fail table: graders against failed episodes.
- How often each pair of graders passes the *same* failures.
- How many failures *every* grader passes. These are the shared blind spots. Pull
  frames from them and look at what they have in common.
- Whether a majority vote or an average beats the best single grader.

**Comparisons worth calling out:**

- **Qwen3-VL base vs the trained graders built on it.** A September 2026 benchmark
  found trained graders scoring below their own base model.
- **RoboReward vs Robometer.** RoboReward learned failures from fake relabelled
  ones. Robometer learned from real failed runs. Do they miss different things?
- **Image orientation.** Run each grader on the stored and mirrored images. You
  can re-check the earlier orientation result for free while you're at it.

**Cost.** Mostly setup, then one job per grader on the pool.

---

## Suggested order

1. **Step 0.** Rollout script, pool and grader sanity checks. Don't skip the
   watching-videos part.
2. **Experiment 2.** Cheapest, and gives a result fast.
3. **Experiment 4.** Runs on the pool with no training.
4. **Experiment 1.** The most work: fine-tuning plus large evaluations.
