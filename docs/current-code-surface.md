# Current Code Surface

Last updated: 2026-04-10 00:45 UTC

## What Exists

- `train.py`: a compact GPT training loop for maze-search data.
- `eval.py`: checkpoint evaluation on random mazes with path-validity and path-optimality metrics.
- `data/maze_gen.py`: connected random-graph generation, random walks, DFS traces, shortest-path utilities, and path validation helpers.
- `data/dataset.py` plus `data/tokenizer.py`: dataset/tokenization layer for the maze task.
- `optimizers.py`: optimizer builder with current `adam` / `muon` support.

## Current Training Stages

- `pretrain`: supervised training on maze-derived sequences.
- `midtrain`: supervised continuation from a prior checkpoint.

Despite the stage names, both are still supervised objectives. There is no verifier-driven rollout loop, no reward model, no policy-gradient update, and no search-plus-distill pipeline implemented yet.

## Why This Repo Is Still Useful

- The maze domain is fully verifiable.
- The generator and evaluator already expose answer correctness and path optimality.
- The code is small enough that later RL additions can be audited precisely instead of hidden inside a large post-training stack.

## Current Gaps

- No sampling/search harness that logs multiple trajectories per prompt.
- No explicit pass@k or diversity metrics.
- No trace-validity metric beyond final path validity/optimality.
- No distinction yet between solver traces, corrupted traces, and self-generated traces.
- No RL baseline of any kind.

## Implication For Current Claims

Any statement in this project about RL, exploration, or sampling must currently be literature-backed or design-level only. Repo-backed empirical claims are limited to the existing supervised maze pipeline until the later baselines are implemented.
