# Statistical Dynamics Reframe

Last updated: 2026-04-24 23:06 UTC

## Why this note exists

The current theory surface was still too close to a method comparison: try
search, SFT, self-training, RL, dense credit, and exploration variants on a
puzzle-pretrained transformer, then compare which works. That is useful later,
but it is not the scientific object.

The corrected object is a discovery problem. Use a deliberately standard puzzle
training pipeline as a controlled dynamical system, then look for statistics or
mathematical objects whose trajectories distinguish imitation-like learning from
RL-like exploration or selection.

For method branch `m`, checkpoint `t`, fixed prompt set `P`, and rollouts
`tau ~ pi_{m,t}(.|x)`, the unit of analysis is a time series:

```text
Phi_m(t) = observable({rollouts}, logits, gradients, activations, optimizer state)
```

The project should treat a statistic as interesting only if its training-time
shape is reproducible, interpretable, and robust to null variants and measurement
artifact checks.

## Methodological lesson from the Coffee Automaton

The useful analogy in `1405.6903` is methodological, not topical.

1. Choose a simple controlled dynamics.
   The Coffee Automaton uses a small stochastic cellular automaton, not a
   realistic fluid simulator. The puzzle analogue is a fixed generator, verifier,
   tokenizer, architecture, optimizer family, and checkpointing schedule.

2. Separate monotone entropy-like variables from complexity-like variables.
   Entropy rises as coffee and cream mix, but the interesting object is a
   coarse-grained apparent complexity that rises during intermediate structure
   formation and falls near equilibrium. In the puzzle pipeline, final reward,
   loss, or pass@1 may be entropy-like: necessary but not mechanistic. Candidate
   complexity-like objects include support occupancy, trajectory-cluster
   birth/death, policy-field apparent complexity, representation rank, and
   prefix-to-future predictability.

3. Tie coarse-graining to causal structure.
   The automaton coarse-grains local spatial neighborhoods because the dynamics
   are local. For puzzles, coarse-graining should use puzzle states and legal
   transitions, not raw token string similarity alone.

4. Compare against null dynamics.
   The paper uses an interacting versus non-interacting comparison, and the
   adjusted metric flattens the non-interacting case. Puzzle nulls should
   preserve surface marginals while breaking global solution structure.

5. Assume first measurements are artifact-prone.
   The Coffee Automaton paper explicitly finds a threshold-border artifact and
   adjusts the metric. Puzzle analogues include tokenizer artifacts, compression
   serialization, clustering thresholds, rollout budgets, temperature, max-length
   truncation, action masks, and probe shortcuts.

## Standard Pipeline Vessel

The first discovery pass should use only the standard branches needed to expose
dynamics:

1. Pretrain on random valid puzzle trajectories.
   This teaches syntax, local validity, and puzzle-state representation without
   installing a solver trace family.

2. Branch to SFT on solver traces.
   This is the imitation-like branch.

3. Branch to REINFORCE-style binary reward RL.
   This is the sparse verifier branch: reward 1 for a valid solved trajectory,
   reward 0 otherwise.

4. Run the same branch split under AdamW and Muon.
   The optimizer split is a robustness check on statistics, not a leaderboard.

Hold fixed the architecture, tokenizer/action format, generator, verifier, probe
prompts, rollout temperature, rollout count, max length, and checkpoint cadence.

## Candidate Statistic Families

### Scalar and optimizer dynamics

Operational objects:

- supervised loss on a fixed solver-trace probe set;
- NLL on a random-valid-trajectory probe set;
- binary reward rate;
- policy entropy, both full-token and legal-action-restricted;
- KL to the pretrain checkpoint and to the previous checkpoint;
- gradient norm and update norm;
- cosine alignment between SFT gradients and RL gradients on matched prompts;
- AdamW/Muon update geometry: layerwise update norms, cosine similarity, and
  effective rank;
- approximate Fisher trace, diagonal Fisher norm, or sharpness proxy on fixed
  probe batches.

Potentially interesting pattern:

- SFT shows fast loss decrease, entropy contraction, and high alignment with
  solver-trace gradients.
- RL shows a noisy expansion phase, lower early gradient alignment with SFT,
  then alignment or contraction after successful clusters appear.
- Loss and reward may be monotone, while update rank, Fisher geometry, or
  legal-action entropy has a rise-and-fall shape.

Main artifacts:

- entropy depends on temperature, action masking, invalid-token mass, and
  sequence length;
- KL can mostly measure formatting drift;
- Fisher and sharpness proxies depend strongly on batch, length, and reward
  normalization;
- AdamW/Muon differences can become optimizer stories unless treated only as
  robustness checks.

### Behavior and process dynamics

Operational objects:

- pass@1 and pass@k;
- trace-validity rate: every transition is legal, not only final answer valid;
- final validity and optimality;
- invalid-action rate by step index;
- branching factor over legal next actions from the same coarse state;
- error-correction rate after a bad or non-shortest move;
- time-to-first-success.

Maze definitions are direct: every emitted edge must exist, the path must reach
the target, and regret can be measured against shortest path length. Sudoku
should separate local legality from global consistency.

Potentially interesting pattern:

- a sudden invalid-action collapse or first-success transition under RL;
- smooth SFT trace-validity gain without support expansion;
- pass@1 rising while pass@k and branching factor collapse, which suggests
  sharpening rather than exploration;
- pass@1 rising while pass@k and distinct successful clusters are preserved,
  which is closer to support expansion.

Main artifacts:

- pass@k is sensitive to `k`, temperature, duplicate handling, max length, and
  verifier strictness;
- branching factor is meaningless unless computed on parsed puzzle states;
- long random walks can fake recovery unless normalized by length and state.

### Exploration and support dynamics

Operational objects:

- state coverage over graph nodes/edges or Sudoku board abstractions;
- action coverage over legal actions from coarse states;
- first-visit rate relative to pretrain rollouts and earlier checkpoints;
- occupancy entropy over states, actions, or trajectory clusters;
- trajectory-cluster birth and death;
- length-normalized base-pretrain log probability of successful rollouts;
- fraction of successful post-training rollouts already found under large-k
  pretrain sampling.

Potentially interesting pattern:

- early pretrain is broad but unstructured;
- early RL expands sampled state/action support before success appears;
- successful trajectory clusters are born during RL and later consolidate;
- SFT adopts a narrower solver-trace family with less exploratory support.

This family is the closest match to the project question. It directly tests
whether RL creates new reachable successful clusters or only amplifies clusters
already present under large-k pretrain sampling.

Main artifacts:

- coverage grows with sample count, so rollout budgets and confidence intervals
  must be fixed;
- base logprob must be length-normalized and computed under the same tokenizer;
- cluster birth/death can be created by arbitrary distance thresholds;
- raw-string clusters can confuse notation changes with strategy changes.

### Representation dynamics

These are conjectural but may become the most informative objects once
behavior-level logging is stable.

Operational objects:

- hidden-state covariance effective rank;
- participation ratio of activation manifolds;
- linear probes for current puzzle state, valid next action, distance to goal,
  future success, or contradiction risk;
- probe geometry: margins, calibration, and class separation;
- compression or apparent complexity of quantized activations;
- compression or apparent complexity of the policy field;
- light-cone-like prefix predictability.

A practical prefix-predictability proxy is:

```text
CE(future event) - CE(future event | prefix representation)
```

where the future event can be eventual success, next invalid action, return to a
shortest-path frontier, or Sudoku contradiction.

Potentially interesting pattern:

- representation rank or policy-field complexity rises while the model organizes
  puzzle structure, then falls as the policy consolidates;
- RL improves future-success predictability from prefixes before pass@1 rises;
- SFT and RL show different layerwise timing of state, action, and success
  probes.

Main artifacts:

- activation rank is sensitive to layernorm scale, batch composition, sequence
  position, and prompt identity;
- probes can learn shortcuts;
- float compression is meaningless unless quantization and serialization are
  fixed and stress-tested.

### Algorithmic and computational complexity proxies

These are closest in spirit to apparent complexity, but they are also the
easiest to overfit.

Operational objects:

- compressed description length of rollout sets conditioned on prompts;
- MDL of trajectory clusters: bits for prototypes plus bits for residuals;
- minimal automata or grammar over strategy traces;
- normalized compression distance between checkpoints' rollout distributions;
- MDL of the policy behavior over coarse puzzle states.

For mazes, strategy symbols can include shortest-frontier move, detour,
backtrack, revisit, invalid edge, terminate, and cycle. For Sudoku, symbols can
include forced single, hidden single, guess-like legal move, contradiction,
repair/backtrack if allowed, solved termination, and invalid termination.

Potentially interesting pattern:

- early behavior is simple random/local behavior;
- intermediate behavior needs a richer strategy grammar;
- late behavior compresses again into a smaller set of successful algorithms;
- SFT quickly adopts a low-MDL solver-trace automaton, while RL passes through a
  higher-MDL mixture of partial strategies before selection.

Main artifacts:

- the symbolization can smuggle in the conclusion;
- grammar induction can overfit;
- compression depends on serialization, duplicate handling, and trace ordering.

## First-Pass Discovery Protocol

Start with maze. Add Sudoku only after the logging surface is stable.

1. Instrument before adding more methods.
   Add multi-sample probe rollouts, parsed state/action traces, pass@k,
   trace-validity, invalid-action location, base-pretrain logprob, per-prompt
   rollout archives, checkpointed logits, and a small activation dump surface.

2. Create fixed probe sets.
   Use an ID probe, a hard probe where pretrain large-k success is low, and a
   structural OOD probe. Do not change these during the first discovery runs.

3. Train only the standard branches.
   Pretrain on random valid trajectories. Branch to SFT on solver traces and
   REINFORCE-style binary reward RL. Run AdamW and Muon versions with at least
   three seeds.

4. Compute four high-value statistics first.
   Track trace-validity phase curves, support occupancy entropy plus first-visit
   rate, trajectory-cluster birth/death plus base-logprob of successes, and
   policy-field apparent complexity.

5. Add one representation diagnostic after behavior logging is stable.
   Start with hidden-state effective rank and, optionally, a prefix probe for
   future success or next-action legality.

6. Run null and artifact controls.
   Use random legal walks, target-shuffled prompts, reward permutation, corrupted
   solver traces, rollout-order shuffle for compression, compression algorithm
   sweeps, quantization sweeps, temperature sweeps, and cluster-threshold sweeps.

The desired result is not "RL wins." A real discovery would look like:

```text
Policy-field apparent complexity rises and falls under RL but not SFT; the hump
coincides with cluster birth and first-success transitions; it disappears under
random-reward and local-validity nulls; and it is robust to compression and
quantization choices.
```

## Relationship To The Older Baseline Ladder

The earlier baseline ladder is still useful, but it should be demoted. Search,
best-of-N distillation, SePT-style self-training, on-policy supervision, dense
credit, and exploration-first algorithms are later perturbations and controls.
They should be introduced after the project has candidate observables worth
stress-testing.

The first milestone is now: find a statistic whose dynamics make the
pretrain->SFT and pretrain->RL branches look different for a mechanistic reason.
