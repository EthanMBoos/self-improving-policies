# Grader audit: robometer on 300 episodes

Simulator success 123/300 = 41.0% [36, 47]

| threshold | tp | fp | fn | tn | precision | recall | fpr | grader rate | gap |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 118 | 137 | 5 | 40 | 0.46 | 0.96 | 0.77 | 0.85 | 0.44 |
| 0.65 | 112 | 96 | 11 | 81 | 0.54 | 0.91 | 0.54 | 0.69 | 0.28 |
| 0.85 | 36 | 21 | 87 | 156 | 0.63 | 0.29 | 0.12 | 0.19 | -0.22 |

AUROC 0.744, Brier 0.285, ECE 0.286

**D5 at threshold 0.65: precision 0.54 versus floor 0.5: PASS.** experiment loop may run

## False positives to watch (simulator failed, highest grader score)

- ep_0002076 seed 2076: score 0.914, 400 steps, `ep_0002076/agentview.mp4`
- ep_0002258 seed 2258: score 0.914, 400 steps, `ep_0002258/agentview.mp4`
- ep_0002145 seed 2145: score 0.910, 400 steps, `ep_0002145/agentview.mp4`
- ep_0002054 seed 2054: score 0.906, 400 steps, `ep_0002054/agentview.mp4`
- ep_0002173 seed 2173: score 0.906, 400 steps, `ep_0002173/agentview.mp4`

## False negatives (simulator succeeded, lowest grader score)

- ep_0002146 seed 2146: score 0.320, 210 steps, `ep_0002146/agentview.mp4`
- ep_0002099 seed 2099: score 0.443, 88 steps, `ep_0002099/agentview.mp4`
- ep_0002004 seed 2004: score 0.461, 86 steps, `ep_0002004/agentview.mp4`
- ep_0002210 seed 2210: score 0.488, 133 steps, `ep_0002210/agentview.mp4`
- ep_0002119 seed 2119: score 0.490, 88 steps, `ep_0002119/agentview.mp4`
