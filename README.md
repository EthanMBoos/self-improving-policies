# VLM graders work once they've seen failures

A VLM can grade a robot, but only after it has been trained on that kind of
robot failing. From 2023 to 2026 the field moved from asking general models
whether an image matches a caption to training small (4B–8B) robot reward
models. Training on failures and near-misses is what made them work. Model size
and extra reasoning mattered much less. The open problem is the false positive:
a failure the grader passes. False positives, not overall accuracy, decide
whether RL against a learned reward works or turns into reward hacking.

## How the field moved

| Period | Idea | Work |
|---|---|---|
| 2022–23 | Image–text similarity as reward | VLM-RM [1], RoboCLIP [2] |
| 2023 | Success detection as yes/no question | SuccessVQA [3] |
| 2023–24 | LLM-written reward code, VLM preferences | Eureka [4], RL-VLM-F [5] |
| Late 2024 | Zero-shot progress from shuffled frames | GVL [6] |
| 2025 | Sim RL on binary success | SimpleVLA-RL [7], πRL [8] |
| Late 2025 | Self-improvement on human labels | π*0.6 [9] |
| 2026 | Robot reward models trained on failures, plus audits | RoboReward [10], Robometer [11], FailBench [12], RoboRef [13] |

## Core findings

**1. Zero-shot general VLMs are weak graders, but not useless.**

"General" here means a general-purpose VLM such as Qwen or Gemini, not a robot
reward model trained across many robots, such as Robometer-4B.

- Zero-shot Flamingo was at chance (50%) on success detection. It only worked
  after fine-tuning on labels [3].
- VLM-RM worked only for coarse poses, and only at the largest CLIP size [1].
- VLM preference labels were least reliable when two states were close in
  progress [5], which is exactly the near-miss case.
- GVL works on clean demos but fails on messy data [6]. Its metric (VOC) is
  measured on successes, so it doesn't test whether a grader rejects failures.
- **Counterpoint:** TOPReward reads progress from a general VLM's token
  probabilities and reports rewards good enough to improve behaviour cloning
  [24]. On FailBench the best model was a general one (next finding) [12]. Treat
  general VLMs as real baselines, not as predetermined losers.

**2. Training on failures is what fixes it.**

- Robot datasets are mostly successes, so models trained on them learn to say
  "success" [14]. Every serious reward model now manufactures failures:
  - RoboReward relabels successes with wrong instructions and cuts them short [10].
  - Robometer adds real failed runs [11].
- Removing RoboReward's fake failures raised its error from 0.845 to 1.450 [10].
- Robometer on LIBERO: a demo-only model ranked progress well (VOC 0.96) but
  barely separated successes from failures (gap 0.11). Adding 1,929 failures
  raised the gap to 0.46 [11].
- Robometer-4B without LIBERO fine-tuning ranks progress within LIBERO episodes
  near chance (Kendall 0.03). Fine-tuning raises it to 0.85 [15]. This measures
  progress ranking, not pass/fail, so it can coexist with good success/failure
  separation.
- **Counterpoint:** on FailBench, every fine-tuned grader scored below its own base
  model, and general Gemini 3 Flash led at about 0.77 balanced accuracy [12]. The
  gains from fine-tuning seem to stay inside the domain the model was trained on.

**3. False positives, not accuracy, decide whether RL works.**

- RoboRef reports a −0.87 rank correlation between a grader's offline accuracy
  and RL success across six reward models, and a mean reward of 0.72 from
  Robometer-4B on failed episodes [13]. We couldn't verify this source, so treat
  it as unconfirmed. The points below don't depend on it.
- Reward hacking is reported by several groups:
  - RoboReward gave maximum reward for picking up the wrong object [11].
  - A VLM declared success too early, and the agent exploited it [16].
  - Policies trained in a world model produced rollouts that look right but
    aren't, and these fooled the VLM [17].
- RL creates *new* rewarded failures, so auditing the grader before training
  undercounts what the trained policy will exploit [18].
- With an imperfect verifier, resampling hits a ceiling set by the verifier's
  false-positive rate [19]. That result is about LLM resampling at inference
  time, so it motivates the question for robots but doesn't answer it. No robot
  paper has measured how grader false positives vs false negatives affect
  fine-tuning on filtered episodes.

**4. Why near-misses fool graders.**

- **Bias toward "success":** graders say success when the evidence is ambiguous,
  and more reasoning doesn't fix it [12].
- **What the camera can see:** if the camera can't tell a closed door from a nearly
  closed one, no model can. Extra views help (about +9 points [20]).
- **Ensembles:** they help less than expected because graders share their
  errors [21].

**5. Leading labs don't use VLM graders as the training reward.**

- Physical Intelligence: human episode labels plus a learned value function [9].
- DeepMind: a model trained on the robot's own demos to predict the steps left
  in the task [22].
- Simulator RL: the simulator's own success check [7, 8].
- DeepMind's world-model evaluation: humans scored the videos [23].
- Learned rewards do give real gains in academic RL, but still trail human reward
  by 10–25 points [10].

## What it means for the premises

- **"Verifying is easier than doing"** holds only when:
  - success is visible to the camera,
  - the grader has enough views,
  - failures aren't near-misses, and
  - the false-positive rate is low relative to the policy's success rate.

  It also wears down under RL, because RL pushes rollouts into the grader's blind
  spots.
- **Bitter Lesson:** what scales is learning from the robot's own successes *and
  failures*, not using the biggest model without training.
- **Big World Hypothesis:** supported. Graders need adapting to each new setting.

## References

Marked † where we only partly checked the source.

1. VLM-RM, Rocamonde et al. — arXiv 2310.12921
2. RoboCLIP, Sontakke et al. — arXiv 2310.07899
3. SuccessVQA, Du et al. — arXiv 2303.07280
4. Eureka, Ma et al. — arXiv 2310.12931
5. RL-VLM-F, Wang et al. — arXiv 2402.03681
6. GVL, Ma et al. — arXiv 2411.04549
7. SimpleVLA-RL — arXiv 2509.09674
8. πRL, Chen et al. — arXiv 2510.25889
9. π*0.6 / RECAP, Physical Intelligence — arXiv 2511.14759
10. RoboReward, Lee et al. — arXiv 2601.00675
11. Robometer, Liang et al. — arXiv 2603.02115
12. FailBench — arXiv 2609.03611 † (numbers taken from a summary of v1)
13. RoboRef, NeurIPS 2026 — roboref-reward.github.io † (not verified; no arXiv
    version found)
14. Tian, Wu, Bajcsy, embodied reward models over-reward — arXiv 2606.01036
15. eVTA0 — arXiv 2609.33653
16. Demo2Reward — arXiv 2606.00083
17. RAW-Dream — arXiv 2605.12334
18. Multimodal reward hacking in RL, Yao et al. — arXiv 2607.09492
19. Inference Scaling fLaws — arXiv 2411.17501
20. Robo-Dopamine — arXiv 2512.23703
21. Shared errors in judge ensembles — arXiv 2608.08002
22. Self-Improving Embodied Foundation Models, DeepMind — arXiv 2509.15155
23. Veo world-model policy evaluation, DeepMind — arXiv 2512.10675
24. TOPReward — arXiv 2602.19313
