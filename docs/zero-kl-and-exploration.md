# Zero-KL, On-Policy Supervision, And Exploration

Last updated: 2026-04-12 05:22 UTC

## Why this note exists

The first theory pass established the main support-expansion criterion, but it still left four follow-up objects too compressed:

1. what the fixed-reference KL bridge does and does not imply at `beta = 0`;
2. whether SePT is actually doing off-policy correction;
3. how strong the current evidence is for diversity-preserving support expansion and long-horizon on-policy claims;
4. which exploration families are worth importing into the puzzle program if plain on-policy supervision already reproduces much of RL.

This note answers those four objects directly.

## 1. Zero-KL RL is not the same object as the fixed-reference bridge

For `beta > 0`, the clean bridge is:

- optimize `E[r(x, y)] - beta KL(pi(.|x) || pi_ref(.|x))`;
- equivalently project by reverse KL onto the Gibbs target
  `pi*_beta(y|x) propto pi_ref(y|x) exp(r(x, y) / beta)`.

That identity is exact, but only for the fixed-reference, positive-`beta` case.

### What happens as `beta -> 0`

The limit of the Gibbs targets is still well defined:

- all non-maximizing actions lose mass;
- among reward-maximizers, the limiting weights are the reference-policy weights restricted to the argmax set.

That does **not** mean literal `beta = 0` is the same optimization problem. At `beta = 0`, the objective becomes plain reward maximization, whose optima are generally set-valued. The fixed reference no longer picks a unique target distribution. So:

- `beta -> 0` preserves a particular ref-weighted limit;
- `beta = 0` removes the exact reverse-KL projection story.

The safe sentence is: the fixed-reference reverse-KL bridge explains KL-regularized RL, not generic zero-KL RL.

### What zero-KL LLM evidence actually exists

The cleanest source-backed zero-KL paper in this set is `Open-Reasoner-Zero` (`2503.24290`). The paper explicitly studies vanilla PPO-style reasoning RL without KL regularization and reports that removing KL loss / KL penalty gives its best stability and final performance. This is the strongest direct answer to "do recent zero-KL reasoning-RL results exist?"

`DeepSeek-R1` (`2501.12948`) is adjacent evidence, but it supports a slightly different point. Its `R1-Zero` stage shows that pure RL from a base model can induce strong reasoning behavior without a preliminary SFT warm start. At the same time, it also shows why "zero-KL can work" should not be over-read: the RL-only stage develops readability and language-mixing pathologies that the later pipeline repairs with cold-start data and multi-stage refinement.

### Why zero-KL can still work without contradicting the usual stability story

Removing the fixed-reference KL term removes one stabilizer, not every stabilizer. The plausible empirical stabilizers in the current LLM setting are:

1. strong initialization from a capable base model;
2. local old-policy / current-policy control from PPO ratios and clipping;
3. variance reduction from GAE, value models, or group-score baselines;
4. simple rule-based rewards rather than noisy learned reward models;
5. format templates, postprocessing, and cold-start data that keep behavior legible.

So the right conclusion is narrow: explicit fixed-reference KL is not necessary in every reasoning-RL setup, but zero-KL success is still an empirical result tied to strong initialization and local trust-region machinery. It is not a new theorem replacing the KL bridge.

## 2. SePT is not off-policy correction

The direct answer is no: `SePT` (`2510.18814`) does not use importance weighting or an off-policy correction term.

Its core loop is:

1. generate low-temperature self-samples;
2. train with ordinary negative log-likelihood on those samples;
3. refresh the self-generated dataset online.

The paper gives a sharper interpretation than "just an engineering trick." Within each round, the SePT objective is an occupancy-weighted forward-KL projection onto the current low-temperature self-teacher. Theorem 1 then says:

- when `tau_s < tau_t`, pairwise logit margins are amplified;
- when `tau_s = tau_t`, the expected first-order gradient vanishes;
- when `tau_s > tau_t`, the mismatch is destructive.

So SePT is best read as online self-sharpening with a better signal-to-noise ratio, not as corrected off-policy RL.

### What SePT does and does not say about on-policy versus off-policy

SePT is only a narrow counterexample to the slogan "on-policy is always better." It shows that, in single-turn reasoning, plain online self-training on sharpened self-samples can recover part of the gain people often attribute to RL. It does **not** show:

- that off-policy learning is generally better than on-policy learning;
- that reward optimization is unnecessary in interactive settings;
- or that low-temperature sampling is a substitute for exploration.

The clean interpretation is:

- low-temperature self-generation up-weights deterministic good responses;
- standard-temperature training copies that sharpened self-teacher;
- online refresh matters because the training distribution keeps moving with the model.

That makes SePT a strong baseline against sparse RL in static reasoning domains, not a general replacement for on-policy RL.

### Why on-policy supervision still matters as a separate comparator

Two later papers make this distinction sharper:

- `Self-Distilled Reasoner` (`2601.18734`) studies on-policy self-distillation and argues that dense teacher supervision along on-policy student rollouts can match or beat RL-style optimization in some reasoning settings.
- `Self-Distillation Enables Continual Learning` (`2601.19897`) shows that on-policy distillation can outperform offline distillation from the same teacher and can preserve long-form reasoning behavior better than plain SFT.

These papers matter because they isolate a question the maze project should test explicitly:

- if on-policy supervision already reproduces the gains, then reward optimization is not the whole story;
- if RL still wins after matched on-policy supervision, then the extra signal is more plausibly reward- or exploration-specific.

## 3. How strong is the evidence for support expansion beyond mode sharpening?

### Best current support

The strongest current support is still suggestive, not definitive.

`LaDi-RL` (`2602.01705`) is the sharpest direct paper in the current set. Its claim is not just higher `pass@1`; it is that latent-space exploration can improve `pass@1` and `pass@k` together while avoiding the diversity collapse of standard token-space RL.

Three newer exploration papers push the same direction:

1. `Reasoning with Exploration: An Entropy Perspective` (`2506.14758`) links high-entropy regions to rare behaviors and reports `pass@K` gains from entropy-shaped exploration.
2. `Rethinking Entropy Regularization in Large Reasoning Models` (`2509.25133`) argues that naive entropy bonuses are too blunt, but selective entropy around the policy nucleus and high-entropy decision points can improve diversity without destabilizing everything.
3. `Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR` (`2509.23808`) argues that token-space exploration may be the wrong surface and that hidden-state shaping can improve exploration without the usual token-level diversity collapse.

What these papers support is the **weaker** claim:

- some methods seem able to preserve or broaden successful-support coverage relative to standard token-space RL.

They do **not** yet prove the **strong** claim:

- training created successful reasoning modes that were absent from the base model's low-probability tail.

### The main cautionary sources still pull the other way

Three papers remain the strongest reasons not to overclaim:

1. `Reasoning with Sampling` (`2510.14901`) shows that better search over the base model can already recover much of the gain that people often assign to RL.
2. `On the Linearity of LLMs' RLVR Training` (`2601.04537`) suggests RLVR often amplifies early-found directions rather than discovering qualitatively new ones throughout training.
3. `Scalpel vs. Hammer` (`2507.10616`) explicitly argues that GRPO mainly amplifies existing capabilities, while SFT changes behavior in a more replacement-like way.

`Beyond Semantics` (`2505.13775`) adds the key process-level warning: better final answers do not automatically mean better reasoning traces.

### Current working rule

The current safe rule stays the same:

- only use "support expansion" if the post-trained policy succeeds on a fixed hard slice in a way that large-`k` base search, search-plus-distill, and reward-free online-refresh controls still cannot match, while trace-valid success and diversity do not collapse.

## 4. Where on-policy learning looks more important

The strongest source-backed claim is not "on-policy always wins." It is narrower:

- as tasks become more interactive, sequential, or hard-exploration dominated, on-policy or closed-loop learning looks more important than it does in single-turn math/code.

### Strongest support

`RobotxR1` (`2505.03238`) is the cleanest direct evidence in the current set. It studies closed-loop RL in an embodied robotics setting and reports large gains over SFT-style baselines.

`Supervised Reinforcement Learning` (`2510.25992`) supports the same mechanism from another angle: when correct multi-step trajectories are too rare to sample, sparse outcome RL and plain imitation both fail, while dense step-wise guidance works.

`Self-Distillation Enables Continual Learning` (`2601.19897`) is not a robotics paper, but it reinforces the sequential-learning part of the story by showing that on-policy distillation can retain and accumulate behavior better than offline SFT.

The broader general-RL backdrop is useful here:

- `Go-Explore` (`1901.10995`) shows why archive-return-explore structures matter in hard-exploration regimes;
- `Exploration by Random Network Distillation` (`1810.12894`) gives a concrete novelty-bonus template;
- `Soft Actor-Critic` (`1801.01290`) is the clean maximum-entropy reference for stabilizing exploration;
- `Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes` (`2408.03539`) is a useful reality check on why interactive control differs from static QA.

### What remains unproved

No paper in this set gives the ideal controlled comparison where horizon changes while architecture, task family, teacher quality, and compute are all held fixed. So the current claim is an inference across multiple papers, not a settled law.

## 5. Exploration families worth testing if plain supervision is already strong

If search-plus-distill, SePT-style self-training, and on-policy supervision already recover most sparse-RL gains, then the next algorithms should target settings where sharpening alone is structurally insufficient.

### Search-then-distill with archive support

Use `Go-Explore` as the template: remember rare successful branches, return to them, then robustify. In puzzles, the key measurement is not just accuracy; it is whether archived search discovers successful trajectory families that plain on-policy RL or plain best-of-`N` forgets.

### Intrinsic reward / novelty bonuses

Use `RND` and count-like bonuses when verifier rewards are too sparse to bootstrap anything. The key measurement is time-to-first-success and partial-state coverage on a fixed hard slice.

### Selective entropy control

Use the entropy papers and `SAC` as the design reference, but only on narrow decision surfaces. The target is not generic higher entropy; it is more successful strategy families without global entropy explosion.

### Latent-space or hidden-state exploration

Use `LaDi-RL` and the hidden-state paper as the template. The central question is whether exploration at a representation level preserves multiple successful modes better than discrete token-space RL.

### Skill discovery before sparse-RL finetuning

`Diversity is All You Need` (`1802.06070`) is the clean classical skill-discovery template, but the current intrinsic-reward literature is a warning that behavior diversity is not the same thing as useful exploration. In puzzles, the test is state coverage and downstream verifier success, not just different-looking traces.

### Closed-loop tool or environment interaction

If we later move beyond static maze prediction into a text-tool or compiler-in-the-loop environment, `RobotxR1` and the robotics literature suggest that closed-loop recovery after mistakes should become a first-class object. That is where plain offline imitation is least likely to be enough.

## 6. What this changes in the puzzle program

The maze plan should now treat the following baselines as mandatory, in this order:

1. large-`k` base-model sampling and search;
2. best-of-`N` plus behavior cloning;
3. SePT-style reward-free online self-training;
4. on-policy supervision / distillation along student rollouts;
5. sparse binary-outcome RL, with KL and zero-KL variants separated explicitly;
6. dense step-wise reward or action-level supervision;
7. only then exploration-first variants such as archive-return, novelty bonuses, selective entropy, or latent/hidden-state exploration.

The main measurement change is also now explicit:

- track distinct successful trajectory clusters, partial-state coverage, and time-to-first-success on the hard slice.

Without those three objects, the project will not be able to say whether a later gain was true support expansion or only a sharper way of visiting already-latent good modes.

## Source list

1. `2305.18290` - `Direct Preference Optimization: Your Language Model is Secretly a Reward Model`
2. `2501.12948` - `DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning`
3. `2502.06781` - `Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning`
4. `2503.24290` - `Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model`
5. `2505.03238` - `RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning`
6. `2505.13775` - `Beyond Semantics: The Unreasonable Effectiveness of Reasonless Intermediate Tokens`
7. `2506.14758` - `Reasoning with Exploration: An Entropy Perspective`
8. `2507.10616` - `Scalpel vs. Hammer: GRPO Amplifies Existing Capabilities, SFT Replaces Them`
9. `2510.14901` - `Reasoning with Sampling: Your Base Model is Smarter Than You Think`
10. `2510.18814` - `A Model Can Help Itself: Reward-Free Self-Training for LLM Reasoning`
11. `2510.25992` - `Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning`
12. `2601.04537` - `Not All Steps are Informative: On the Linearity of LLMs' RLVR Training`
13. `2601.18734` - `Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models`
14. `2601.19897` - `Self-Distillation Enables Continual Learning`
15. `2602.01705` - `Beyond Mode Elicitation: Diversity-Preserving Reinforcement Learning via Latent Diffusion Reasoner`
16. `1810.12894` - `Exploration by Random Network Distillation`
17. `1801.01290` - `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`
18. `1802.06070` - `Diversity is All You Need: Learning Skills without a Reward Function`
19. `1901.10995` - `Go-Explore: a New Approach for Hard-Exploration Problems`
20. `2408.03539` - `Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes`
21. `2501.11533` - `The impact of intrinsic rewards on exploration in Reinforcement Learning`
