# Puzzle RL Theory Backlog

Last updated: 2026-04-12 02:57 UTC

## Immediate

- Define the first hard-slice evaluation where base-model large-k search still has near-zero success; use that slice as the only place where later support-expansion claims are allowed.
- Make matched large-k base search and positive-support distillation mandatory controls in every later RL experiment.
- Decide what trajectory/action clustering surface to log so later runs can test whether RL adds new successful modes or only amplifies early-found ones.
- Add multi-sample maze evaluation with `pass@k`, diversity logging, and base-logprob logging.
- Implement best-of-N filtering plus behavior cloning as the first non-RL baseline.
- Implement reward-free online self-training to isolate the value of on-policy refresh.
- Add a sparse binary-outcome RLVR baseline only after the above two baselines are in place.
- Define the step-wise action representation needed for the dense-credit SRL-style baseline.

## Core Mechanism Questions

- When does RL add information beyond fixed-trace imitation, rather than just reallocating probability mass toward already latent successful trajectories?
- What survives matched large-k search, search-plus-cloning, and reward-free online-refresh controls on the same hard slice?
- How much of the apparent RL gain is really on-policy data refresh or search, versus policy-gradient optimization?
- Which metrics best distinguish “better answers” from “better reasoning” in a fully verifiable puzzle setting?
- When does discrete RL collapse diversity, and can diversity-preserving variants keep pass@k while improving pass@1?
- How should train-time RL and test-time sampling/search be compared under a matched compute contract?
- When a method improves pass@1, how much of that gain compresses back into a plain supervised dataset?

## Experiment Design Tasks

- Specify the first maze task family, solver/verifier contract, and OOD splits.
- Specify the hard-slice contract explicitly:
  - near-zero base success even under aggressive search,
  - fixed across all later baselines,
  - logged separately from the easier bulk slice.
- Decide whether Sudoku should be the second domain or whether another graph/planning puzzle gives cleaner verifier control.
- Define a baseline suite:
  - supervised trace imitation,
  - best-of-N plus behavior cloning,
  - reward-free self-training,
  - RLVR,
  - later dense step-wise reward or diversity-preserving RL.
- Define metrics:
  - pass@1,
  - pass@k,
  - solution diversity,
  - trace validity,
  - path optimality,
  - base-policy log-probability of successful trajectories,
  - compression-back-to-SFT retention,
  - calibration / confidence concentration,
  - OOD transfer.

## Repo Work Needed Later

- Add a verifier-backed rollout/evaluation surface for self-generated trajectories.
- Add artifact logging that preserves both answer correctness and trace/process quality.
- Keep the current supervised stages reproducible while introducing RL baselines one at a time.
