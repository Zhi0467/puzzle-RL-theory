# Puzzle Experiment Contract

Last updated: 2026-04-12 05:22 UTC

## Goal

Build a small, fully auditable puzzle pipeline that can tell apart:

1. fixed-dataset imitation,
2. online self-improvement from self-generated data,
3. verifier-filtered search plus distillation,
4. on-policy supervision on model-generated rollouts,
5. outcome-reward RL,
6. dense step-wise credit assignment,
7. and later diversity-preserving RL.

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

### Baseline D: on-policy supervision or distillation

- Roll out from the current student policy.
- Supervise those rollouts with a privileged teacher:
  - exact solver annotations where available,
  - or a demonstration-conditioned / search-conditioned teacher later.
- Purpose: test whether on-policy supervision already reproduces gains that might otherwise be attributed to reward optimization.

### Baseline E: binary-outcome RLVR

- Sparse reward on final path correctness, optionally with optimality bonus later.
- Split KL-regularized and zero-KL variants explicitly once the RL path exists.
- Purpose: the standard "RL helps reasoning" object, separated from both search and on-policy supervision.

### Baseline F: step-wise supervised RL / dense credit

- Convert expert paths or expert action traces into step-wise targets.
- Reward action similarity or partial progress even when the final rollout is wrong.
- Purpose: probe the regime where plain RLVR fails because pass@k is effectively zero.

### Baseline G: diversity-preserving or exploration-first RL

- Later-stage extension only after Baselines A-F are stable.
- Candidate objects:
  - explicit diversity bonus in token space,
  - selective entropy control on narrow decision surfaces,
  - archive-and-return search for rare successful branches,
  - novelty bonuses over partial states,
  - latent-policy or hidden-state exploration variants if the repo grows that direction.
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
5. semantically clustered successful-solution families, not just string-level variety

### Support-expansion diagnostics

1. base-policy log-probability of successful trajectories found by each method
2. fraction of successful trajectories that were already seen under large-k base sampling
3. compression test:
   - distill the discovered successful trajectories back into a plain SFT model,
   - then measure how much of the gain survives
4. OOD transfer after matched in-domain training
5. trajectory-cluster novelty:
   - compare successful trace/action clusters from base sampling, early RL checkpoints, and the final policy
   - ask whether RL only amplifies early-found clusters or actually adds new successful ones
6. time-to-first-success on a fixed hard slice
7. partial-state coverage before the first successful full rollout

### Process diagnostics

1. trace validity, not only final-answer validity
2. shortest-path regret
3. error taxonomy:
   - invalid edge,
   - valid but non-terminal path,
   - valid but non-optimal path,
   - search failure despite nearby valid alternatives
4. recovery after an early wrong move once interactive environments are added

## Decision rules

### Primary operational criterion

Claim ``support expansion'' only if, on a fixed hard slice, the post-trained policy emits trace-valid successful trajectories at a meaningful rate that the pre-trained model still cannot reach under larger-k search and positive-support distillation controls.

### Primary operational criterion for "RL added more than on-policy supervision"

Treat reward optimization as doing something distinct only if RL still wins after:

1. matched search-plus-distill controls;
2. matched reward-free online refresh controls;
3. matched on-policy supervision or distillation on the same rollout distribution.

### Evidence for mostly mode elicitation / sharpening

Interpret the result as mostly sharpening if most of the following happen:

1. best-of-N plus cloning matches RL;
2. reward-free online self-training matches most of the gain;
3. RL improves pass@1 but hurts pass@k;
4. successful RL trajectories already have moderate or high base-model likelihood;
5. gains compress cleanly back into a supervised dataset;
6. matched on-policy supervision reproduces the same gain.

### Evidence for genuine support expansion

Treat the result as stronger evidence for support expansion only if several harder conditions hold:

1. the base model still has near-zero success on the hard slice even after aggressive best-of-N or search-based decoding;
2. the post-RL advantage survives matched best-of-N plus cloning and reward-free online-refresh baselines;
3. successful post-RL traces live in clusters that were absent or vanishingly rare in base samples and early RL checkpoints;
4. step-level trace validity improves, not only final-answer accuracy;
5. pass@1 and pass@k both improve, or at least pass@k and diversity do not collapse;
6. gains survive on structural OOD splits.
7. matched on-policy supervision still leaves a residual RL advantage.

### What does not count on its own

Do not treat the following as support-expansion evidence by themselves:

1. higher final-answer accuracy alone;
2. higher pass@1 when large-k base search already nearly matches the result;
3. good trajectories that search-plus-distill can already recover;
4. gains reproduced by reward-free online self-training;
5. longer or more natural-looking reasoning traces.
6. gains reproduced by matched on-policy supervision on the same rollout support.

### Evidence that zero-KL versus KL is part of the object

Treat the KL setting as a real mechanism question only if:

1. the zero-KL and KL variants behave differently under matched rollout budgets;
2. the difference survives matched clipping / optimizer settings;
3. any zero-KL gain is not just an artifact of drifting into illegible or verifier-exploiting behavior.

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
4. Add on-policy supervision or distillation on student rollouts.
5. Add sparse binary-outcome RLVR, with KL and zero-KL variants separated.
6. Add dense step-wise reward baseline.
7. Only then ask whether exploration-first or diversity-preserving RL is worth the extra complexity.

## What would count as a successful first stage

The first stage succeeds if the repo can produce one clean figure or table answering:

- how much of the observed gain comes from search/filtering,
- how much comes from online refresh,
- whether on-policy supervision already captures the remaining gain,
- and whether sparse RL adds anything once those baselines are present.

That is already enough to sharpen the scientific object before expanding to Sudoku or larger agentic settings.
