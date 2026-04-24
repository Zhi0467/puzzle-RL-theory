# Puzzle Experiment Contract

Last updated: 2026-04-24 23:06 UTC

## Goal

Build a small, fully auditable puzzle pipeline that can discover statistics
whose training-time dynamics distinguish:

1. random-valid-trajectory pretraining,
2. solver-trace SFT,
3. and REINFORCE-style binary-reward RL.

The target question is not "which method gets the best benchmark number?" or
"which train/test-time scaling trick works best?" It is:

- which statistics behave like monotone entropy-like variables;
- which statistics behave like nonmonotone complexity-like variables;
- which statistics distinguish pretrain -> SFT from pretrain -> RL;
- and which statistics survive null dynamics and artifact checks.

The older support-expansion question remains downstream. First we need a
trustworthy observable surface.

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

### Standard branch split

Use one shared pretrain checkpoint as the vessel:

1. `pretrain`: random valid puzzle trajectories, not solver traces.
2. `sft`: solver-trace supervision from the pretrain checkpoint.
3. `reinforce`: sparse binary reward from the same pretrain checkpoint.
4. Optimizer robustness branches: AdamW and Muon.

Keep architecture, tokenizer/action format, generator, verifier, probe prompts,
rollout count, temperature, max length, and checkpoint cadence fixed.

### Measurement schedule

At dense checkpoints, save:

1. model weights and optimizer state;
2. fixed-probe multi-sample rollouts;
3. parsed puzzle-state traces;
4. logits/logprobs and length-normalized base-pretrain logprob;
5. reward and validity labels;
6. a small activation dump on fixed prefix positions;
7. optional matched-gradient diagnostics on a fixed microbatch.

Plot statistics as time series aligned by optimizer step and, for RL, rollouts
consumed. Final score tables are secondary.

## First Statistics To Try

### Statistic 1: trace-validity phase curve

For every checkpoint:

1. fraction of rollouts with all legal transitions;
2. fraction that terminate correctly;
3. invalid-action rate by step index;
4. shortest-path regret and distance-to-target trajectory.

Purpose: separate final-answer success from process validity and identify
whether RL has a sudden validity transition that SFT does not.

### Statistic 2: support occupancy entropy and first-visit rate

For every checkpoint:

1. entropy over visited graph states or state-action pairs;
2. first-visit rate relative to pretrain rollouts;
3. first-visit rate relative to earlier checkpoints;
4. bootstrap intervals under fixed rollout budget.

Purpose: detect exploration or support expansion before final reward rises.

### Statistic 3: trajectory-cluster birth/death plus base logprob

Cluster parsed trajectories, not raw strings. Track:

1. cluster mass over time;
2. newly born clusters above a fixed threshold;
3. dead clusters below that threshold;
4. successful clusters;
5. length-normalized logprob under the original pretrain checkpoint.

Purpose: distinguish amplification of early clusters from birth of new
successful trajectory families.

### Statistic 4: policy-field apparent complexity

For a fixed set of coarse puzzle states, evaluate next-action distributions,
quantize the resulting policy-field matrix, and compress it after puzzle-causal
coarse-graining.

Purpose: create a Coffee-Automaton-style candidate statistic. Early policies may
be simple and local, intermediate policies may encode many contingencies, and
late policies may compress again into a more deterministic strategy.

Artifact checks:

1. compression algorithm sweep;
2. quantization sweep;
3. row/column permutation controls;
4. rollout-order shuffle;
5. random legal-walk policy field.

## Baseline matrix

This matrix is now a later control surface, not the first milestone. Introduce
these branches only after the standard pretrain/SFT/RL vessel has produced
candidate statistics worth stress-testing.

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

1. Add statistics-oriented logging to the current maze repo:
   - `k` rollouts,
   - pass@k,
   - parsed state/action traces,
   - trace-validity curves,
   - support occupancy entropy,
   - first-visit rates,
   - trajectory-cluster birth/death,
   - diversity logging,
   - base-logprob logging.
2. Run the standard vessel: random-valid pretrain, solver-trace SFT, and binary-reward REINFORCE.
3. Add policy-field apparent complexity and one representation-rank diagnostic.
4. Run null and artifact checks.
5. Only then add best-of-N filtering, reward-free self-training, on-policy supervision, KL/zero-KL RLVR, dense credit, or exploration-first methods as perturbations.

## What would count as a successful first stage

The first stage succeeds if the repo can produce one clean figure or table answering:

- which candidate statistics have robust time-series shapes;
- which ones distinguish SFT from RL across AdamW and Muon;
- which ones flatten under random-reward or local-validity nulls;
- and which ones are artifacts of clustering, compression, temperature, or rollout budget.

That is already enough to sharpen the scientific object before expanding the
method grid, adding Sudoku, or claiming anything about support expansion.
