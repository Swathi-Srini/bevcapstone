# Paper working outline

## Research question
**Can an autonomous driving policy maintain route-following performance and optimise energy consumption when its visual perception is progressively degraded?**

Operationalised (every term is a measured quantity):
- *Progressively degraded perception*: fog / rain intensity swept from 0 to 1; effect on perception **measured**, not assumed: the share of vehicle detections the unmodified YOLOv8 pipeline keeps (Fig. 2; rain 1.0 keeps about 4%, fog 1.0 about 27%). Weather has no dynamics effect (friction constant), so it acts only through perception.
- *Route-following*: route completion, success (arrive at destination), failure (crash + off-road), lateral error.
- *Energy*: estimated traction Wh/km (`bev_rl/energy.py`, declared virtual vehicle; an estimate, not battery electricity; no regeneration).
- *Maintain*: route completion and success stay within a stated tolerance of the clear-weather value as degradation grows (curve, Fig. 7), compared to fair baselines.
- *Optimise energy*: lower Wh/km than baselines at equal or better route completion; Pareto plot success vs Wh/km.

## Sub-questions
- RQ1 How fast does a policy trained under perfect perception (previous best `ov_split`) degrade as detections disappear? (baseline curve)
- RQ2 Does coupling caution to measured detector reliability (c_auto) flatten that curve? (ours vs baseline vs no-caution control)
- RQ3 Is the gain from the caution mechanism itself rather than from training with dropout? (control `ctrl_nocaut`, same data/architecture)
- RQ4 What does safety cost in energy/time? (c=0 vs c=1 vs auto; Pareto)
- RQ5 Can RL fine-tuning with an energy + robustness reward move the Pareto front further? (reward-combination ablation, `ppo_caution.py`; currently no gain from earlier PPO, so reported as is)

## Claimed contributions (only what measurements support at submission)
1. Measured degradation curve of a real detector pipeline under fog/rain and its use as the perception-reliability signal (Fig. 2).
2. Reliability-coupled caution conditioning: one policy, caution input `c = 1 - retention(weather)`, caution-aware privileged teacher (C4/C5).
3. Progressive-degradation protocol with fair baselines, fresh seeds, Wilson CIs (Fig. 5, 7).
4. Combinable energy + robustness RL reward and its ablation (C7), including negative results.
Do NOT claim: physical fog/rain effects on dynamics; real-vehicle energy; anything not in the figures.

## Section plan
1 Introduction (RQ, gap from `literature/`), 2 Related work, 3 Method (Fig. 1; BEV overlay; caution; teacher; RL reward), 4 Experimental protocol (seeds, baselines, metrics), 5 Results (Fig. 2-7), 6 Ablations (scalar split, overlay, caution vs control, reward terms), 7 Limitations (see `KNOWN_LIMITATIONS.md`: simulated detector dropout from measured retention rather than the camera pipeline in the loop; privileged teacher; estimated energy), 8 Conclusion.

## Status tracker
| Item | Status |
|---|---|
| Fig 1-4, 6 | generated (from existing results) |
| Fig 5, 7 (main results) | waiting for evals: baseline running; caution + control training |
| RL reward ablation | queue running (`rl_queue.sh`) |
| Progressive sweeps (rain, fog) | script ready (`eval_caution --sweep rain|fog`), runs after training |
| Literature | `literature/`: several items snippet-level, must be read before citing |
