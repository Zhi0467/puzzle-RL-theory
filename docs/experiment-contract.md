# Puzzle Experiment Contract

Last updated: 2026-04-10 23:35 UTC

## Goal

Build a small, fully auditable puzzle pipeline that can tell apart:

1. fixed-dataset imitation,
2. online self-improvement from self-generated data,
3. verifier-filtered search plus distillation,
4. outcome-reward RL,
5. dense step-wise credit assignment,
6. and later diversity-preserving RL.

The target question is not "which method gets the best benchmark number?" It is:

- when do we only sharpen an already-present solution distribution, and
- when do we actually enlarge the set of successful trajectories available to the model?

## Current repo reality

The current codebase already supports a useful minimal maze setting:

- `pretrain`: random-walk-style sequence modeling;
- `midtrain`: DFS-trace supervision with a DFS path target;
- `eval.py`: path validity and path optimality on fresh random mazes;
- graph generation and shortest-path verification in `data/maze_gen.py`.

What is missing today:

- multi-sample logging per prompt;
- pass@k and diversity metrics;
- explicit trace-validity metrics beyond final path validity;
- online self-generation loops;
- RL or search-plus-distill baselines.

## First experiment family

### Domain

Maze graphs from the existing generator.

### Train / test split ideas

1. In-distribution:
   - same node count and edge probability as training.
2. Mild OOD:
   - larger graphs,
   - sparser graphs,
   - denser graphs.
3. Structural OOD:
   - different maze-generation regimes or graph families,
   - later a second puzzle family such as Sudoku or constrained graph planning.

### Base objects to log on every prompt

1. prompt graph and `(s, t)`;
2. all sampled trajectories;
3. final proposed path;
4. path validity;
5. shortest-path gap;
6. base-model log-probability of each sampled trajectory;
7. trace features needed for later diversity and compression analysis.

## Baseline matrix

### Baseline A: fixed-data SFT

- Existing `pretrain` and `midtrain` pipeline.
- Purpose: measure what pure imitation can do on the current trace format.

### Baseline B: best-of-N plus behavior cloning

- Sample `N` trajectories from the current model.
- Keep only verifier-positive trajectories.
- Distill them with ordinary SFT.
- Purpose: test how much of the gain comes from search plus filtering alone.

### Baseline C: reward-free online self-training

- SePT-style loop:
  - low-temperature self-generation,
  - standard-temperature supervised update,
  - refresh data online after each update round.
- Purpose: isolate the value of on-policy data refresh without any reward model.

### Baseline D: binary-outcome RLVR

- Sparse reward on final path correctness, optionally with optimality bonus later.
- Purpose: the standard "RL helps reasoning" object.

### Baseline E: step-wise supervised RL / dense credit

- Convert expert paths or expert action traces into step-wise targets.
- Reward action similarity or partial progress even when the final rollout is wrong.
- Purpose: probe the regime where plain RLVR fails because pass@k is effectively zero.

### Baseline F: diversity-preserving RL

- Later-stage extension only after Baselines A-E are stable.
- Candidate objects:
  - explicit diversity bonus in token space,
  - latent-policy or diffusion-style reasoning variants if the repo grows that direction.
- Purpose: test whether preserving diversity lets RL improve pass@1 without sacrificing pass@k.

## Metrics that matter

### Standard outcome metrics

1. pass@1
2. pass@k
3. valid-path rate
4. optimal-path rate

### Diversity metrics

1. unique valid solutions per prompt
2. entropy over distinct path classes
3. average pairwise trajectory distance
4. pass@k curve shape, not just one `k`

### Support-expansion diagnostics

1. base-policy log-probability of successful trajectories found by each method
2. fraction of successful trajectories that were already seen under large-k base sampling
3. compression test:
   - distill the discovered successful trajectories back into a plain SFT model,
   - then measure how much of the gain survives
4. OOD transfer after matched in-domain training

### Process diagnostics

1. trace validity, not only final-answer validity
2. shortest-path regret
3. error taxonomy:
   - invalid edge,
   - valid but non-terminal path,
   - valid but non-optimal path,
   - search failure despite nearby valid alternatives

## Decision rules

### Evidence for mostly mode elicitation / sharpening

Interpret the result as mostly sharpening if most of the following happen:

1. best-of-N plus cloning matches RL;
2. reward-free online self-training matches most of the gain;
3. RL improves pass@1 but hurts pass@k;
4. successful RL trajectories already have moderate or high base-model likelihood;
5. gains compress cleanly back into a supervised dataset.

### Evidence for genuine support expansion

Treat the result as stronger evidence for support expansion only if several harder conditions hold:

1. RL finds successful trajectories that large-k base sampling almost never finds;
2. those trajectories have consistently low base-model likelihood before training;
3. pass@1 and pass@k both improve, or at least pass@k does not collapse;
4. gains survive on structural OOD splits;
5. the effect does not disappear when matched search-plus-distill and reward-free online baselines are included.

### Evidence that dense credit assignment matters

Call the regime "credit assignment limited" if:

1. pass@k for the base model and RLVR is near zero;
2. binary-outcome RLVR barely moves;
3. step-wise reward or action-level supervision produces clear gains.

## Implementation order

1. Add multi-sample evaluation to the current maze repo:
   - `k` rollouts,
   - pass@k,
   - diversity logging,
   - base-logprob logging.
2. Add best-of-N filtering plus distillation.
3. Add reward-free online self-training.
4. Add sparse binary-outcome RLVR.
5. Add dense step-wise reward baseline.
6. Only then ask whether latent or diversity-preserving RL is worth the extra complexity.

## What would count as a successful first stage

The first stage succeeds if the repo can produce one clean figure or table answering:

- how much of the observed gain comes from search/filtering,
- how much comes from online refresh,
- and whether sparse RL adds anything once those baselines are present.

That is already enough to sharpen the scientific object before expanding to Sudoku or larger agentic settings.
