# Research Scope

Last updated: 2026-04-12 05:22 UTC

## Core Question

What does reinforcement learning actually add over supervised fine-tuning in LLM-style reasoning systems, once we separate:

- fixed-trace imitation from self-generated trajectories,
- objective choice from data freshness,
- pass@1 gains from diversity collapse,
- and final-answer improvement from process-level reasoning quality?

## Immediate Focus

This project is about principled understanding first, not leaderboard chasing. The initial report should answer four things clearly:

1. What is the exact conceptual difference between SFT and RL?
2. Which recent results really support exploration or capability expansion, and which mostly support mode elicitation or search-plus-distillation?
3. How closely related are RL and sampling in current LLM reasoning pipelines?
4. What controlled puzzle experiments would most cleanly resolve the remaining disagreement?

The current follow-up note adds three sharper sub-questions:

5. What changes when the fixed-reference KL coefficient goes to zero, and what zero-KL evidence is actually source-backed?
6. When does on-policy supervision already reproduce gains that are often attributed to RL?
7. Which exploration algorithms should still be tested if search, self-training, and on-policy supervision are already strong?

## Working Hypotheses

- Much of current RLVR’s practical gain comes from search, filtering, and distribution sharpening over rare successful trajectories rather than from creating wholly new reasoning support.
- Zero-KL reasoning RL can work empirically, but once the fixed-reference KL term disappears the exact reverse-KL bridge no longer applies; the remaining stability story is local and algorithmic rather than theorem-level.
- The regimes where RL genuinely helps are the ones with a large generation-verification gap: correct trajectories are hard to write down but easy to verify.
- On-policy supervision is a distinct comparator, not just a softer version of RL. If it matches RL on the same rollout distribution, the extra gain is not obviously reward-specific.
- On hard small-model tasks, dense step-wise guidance may matter more than sparse final-answer rewards.
- Diversity is part of the scientific object, not just a side effect. A method that improves pass@1 by collapsing pass@k has to be interpreted differently from one that improves both.
- For agentic settings, the balance may shift further toward RL because the trajectory distribution drifts more strongly than in single-turn math/code tasks.

## What Counts As Evidence

### Strong evidence

- Exact objective identities or impossibility results with explicit assumptions.
- Controlled comparisons that hold prompts, verifiers, and data freshness fixed.
- Puzzle-domain experiments where both final correctness and reasoning-trace validity can be verified automatically.
- On a fixed hard slice, post-RL wins that survive larger pre-RL sampling/search and positive-support distillation controls while also improving trace-valid success.
- Matched on-policy supervision baselines that let us ask whether reward optimization adds anything beyond learning on the model's own rollout distribution.

### Weak evidence

- Benchmark gains without matched pass@k or diversity reporting.
- Comparisons that change both optimizer and sampled data distribution.
- Claims about “new reasoning” that only use final-answer accuracy.
- Higher pass@1 that large-k base search, search-plus-cloning, or reward-free online self-training can already recover.

## Target Experimental Object

Use small, fully verifiable puzzle domains to compare:

- SFT on solver traces,
- search plus distillation,
- reward-free self-training,
- on-policy supervision or distillation along student rollouts,
- RLVR with outcome rewards,
- and later dense or diversity-preserving RL variants.

The first environment should stay simple enough that we can audit:

- what trajectories were already in the base policy,
- what the verifier is rewarding,
- how diversity changes through training,
- whether any claimed gain survives compression back into supervised data,
- and whether zero-KL RL behaves differently from KL-regularized RL once the rollout distribution starts to drift.
