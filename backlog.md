# Puzzle RL Theory Backlog

Last updated: 2026-04-24 23:06 UTC

## Immediate

- Treat the first experimental milestone as statistic discovery, not method comparison.
- Define the exact random-valid-trajectory pretrain distribution for maze.
- Define the exact solver-trace SFT branch and the binary-reward REINFORCE branch from the same pretrain checkpoint.
- Define AdamW and Muon as robustness branches for statistics rather than as optimization contestants.
- Add dense checkpoint logging for fixed probe prompts and fixed rollout budgets.
- Implement parsed state/action trace logging before any new method variants.
- Compute the first four statistics:
  - trace-validity phase curve;
  - support occupancy entropy and first-visit rate;
  - trajectory-cluster birth/death plus length-normalized base-pretrain logprob;
  - policy-field apparent complexity.
- Define nulls for the statistic surface:
  - random legal maze walk conditioned on current node but not target;
  - random-reward RL preserving reward rate;
  - target-shuffled prompts;
  - locally plausible but causally corrupted solver traces.
- Define artifact checks for compression, quantization, rollout order, temperature, rollout budget, and clustering thresholds.
- Define the first hard-slice evaluation where base-model large-k search still has near-zero success; use that slice as the only place where later support-expansion claims are allowed.
- Make matched large-k base search and positive-support distillation mandatory controls in every later RL experiment.
- Decide the concrete on-policy supervision baseline:
  - solver-labeled student rollouts,
  - search-conditioned teacher,
  - or both.
- Reserve a clean KL-versus-zero-KL ablation only after the sparse-RL baseline exists.
- Decide what trajectory/action clustering surface to log so later runs can test whether RL adds new successful modes or only amplifies early-found ones.
- Add multi-sample maze evaluation with `pass@k`, diversity logging, and base-logprob logging.
- Implement best-of-N filtering plus behavior cloning as the first non-RL baseline.
- Implement reward-free online self-training to isolate the value of on-policy refresh.
- Implement on-policy supervision or distillation on student rollouts before sparse RL.
- Add a sparse binary-outcome RLVR baseline only after the above three baselines are in place.
- Define the step-wise action representation needed for the dense-credit SRL-style baseline.

## Core Mechanism Questions

- Which statistic has a reproducible, interpretable trajectory over training, rather than only a final value?
- Which candidate statistic behaves like a monotone entropy variable, and which behaves like a nonmonotone apparent-complexity variable?
- Which coarse-graining is tied to puzzle causality rather than raw token form?
- Which statistic distinguishes pretrain -> SFT from pretrain -> RL across both AdamW and Muon?
- Which statistic disappears or flattens under random-reward, target-shuffled, or local-validity-only nulls?
- When does RL add information beyond fixed-trace imitation, rather than just reallocating probability mass toward already latent successful trajectories?
- What survives matched large-k search, search-plus-cloning, and reward-free online-refresh controls on the same hard slice?
- What survives matched on-policy supervision on the same rollout support?
- How much of the apparent RL gain is really on-policy data refresh or search, versus policy-gradient optimization?
- When zero-KL RL works, which stabilizers are actually carrying it: clipping, initialization, reward simplicity, or something else?
- Which metrics best distinguish “better answers” from “better reasoning” in a fully verifiable puzzle setting?
- When does discrete RL collapse diversity, and can diversity-preserving variants keep pass@k while improving pass@1?
- How should train-time RL and test-time sampling/search be compared under a matched compute contract?
- When a method improves pass@1, how much of that gain compresses back into a plain supervised dataset?

## Experiment Design Tasks

- Specify the fixed probe sets for statistic discovery:
  - ID probe;
  - hard probe;
  - structural OOD probe.
- Specify the policy-field grid for maze apparent complexity:
  - coarse puzzle states;
  - legal-action vocabulary;
  - quantization bins;
  - compression algorithms;
  - serialization and permutation controls.
- Specify the parsed trajectory clustering surface before using any cluster-birth result.
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
  - on-policy supervision or distillation,
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
  - time-to-first-success,
  - partial-state coverage,
  - calibration / confidence concentration,
  - OOD transfer,
  - policy-field apparent complexity,
  - hidden-state effective rank,
  - prefix predictability of future success or invalid action.

## Repo Work Needed Later

- Add a fixed-probe rollout archive surface keyed by checkpoint, branch, seed, prompt, and rollout index.
- Add parsers that convert token rollouts into maze state/action traces and error labels.
- Add statistic computation scripts before adding new training methods.
- Add a verifier-backed rollout/evaluation surface for self-generated trajectories.
- Add artifact logging that preserves both answer correctness and trace/process quality.
- Keep the current supervised stages reproducible while introducing RL baselines one at a time.
- If the project later adds interactive tools, define recovery-after-error metrics before adding RL code.
