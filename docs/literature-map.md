# RL vs SFT Literature Map

Last updated: 2026-04-12 05:22 UTC

## Research object

The question for this project is not "does RL help on benchmarks?" It is narrower:

1. What is the exact conceptual difference between SFT and RL?
2. When do observed RL gains mostly reflect sampling, search, or online data refresh?
3. When is there evidence that RL changes what the model can do, rather than only concentrating probability mass on behaviors already latent in the base model?
4. What should a controlled puzzle-domain experiment measure to distinguish these cases?

## Exact theory and safe claims

### 1. Fixed-data SFT and on-policy RL are different optimization objects

- SFT minimizes negative log-likelihood on a fixed dataset of prompt-response pairs.
- RL optimizes expected reward on trajectories sampled from the current policy, so the training distribution moves with the model.
- This difference matters even before discussing "reasoning": RL can change both the objective and the data source at the same time.

### 2. KL-regularized RL has one exact bridge to distillation

The clean exact identity is the fixed-reference KL-regularized case:

- `max_pi E[r(x,y)] - beta KL(pi(.|x) || pi_ref(.|x))`
- This is exactly equivalent to reverse-KL projection onto the Gibbs-reweighted target
  `pi_star(y|x) propto pi_ref(y|x) exp(r(x,y)/beta)`.

This is the strongest current "RL vs distillation" statement because it is an objective identity, not an analogy. A stable LLM-facing external reference for this bridge is the DPO derivation in `2305.18290`.

### 3. Most practical post-training algorithms are not exact fixed-reference KL-RL

Many modern post-training methods use moving anchors, self-teachers, pairwise surrogates, or changing online data. These may be related to the KL bridge, but they are not automatically objective-identical to it. The safe wording is:

- exact equivalence holds for fixed-reference KL-regularized RL;
- practical LLM post-training often mixes optimizer choice, data refresh, and target drift.

## Best empirical read from the current paper set

### A. A large fraction of current RL gains can be reproduced without policy-gradient magic

#### Pure sampling can recover much of the gain

`Reasoning with Sampling` (`2510.14901`) shows that a training-free sampler built from the base model's own likelihoods can nearly match or outperform GRPO on several reasoning tasks, while preserving pass@k and diversity. Their strongest mechanism claim is not "sampling is always enough", but that a lot of post-RL performance lives in high-likelihood regions already present in the base model.

#### Reward-free online self-training can also recover a large fraction of the gain

`SePT` (`2510.18814`) shows that self-generated supervision alone can improve reasoning on some models without rewards or verifiers. Two details matter:

- online refresh is important; freezing one self-generated dataset gives much weaker gains;
- temperature decoupling matters; low-temperature generation plus standard-temperature training is much stronger than coupling them.

This is strong evidence that "being on-policy" and "refreshing the training distribution" are major parts of the story.

#### In binary-reward settings, best-of-N positives plus cloning already go surprisingly far

`OREAL` (`2502.06781`) proves that, in binary-feedback environments, behavior cloning on positive trajectories from best-of-N sampling is sufficient for the KL-regularized optimum on the positive support. Their added RL machinery then focuses on two things:

- shaping negative-example learning so gradients are consistent; and
- adding token-level credit assignment to cope with sparse rewards.

This is important because it cleanly separates one part of RL that may be reducible to sampling-plus-distillation from another part that may still matter.

### B. Standard discrete RLVR often looks more like distribution sharpening than support expansion

#### RLVR trajectories look strongly linear

`On the Linearity of LLMs' RLVR Training` (`2601.04537`) finds that both weights and output log-probabilities evolve approximately linearly across training steps. Their interpretation is that RLVR often keeps amplifying directions discovered early, rather than continually finding qualitatively new behavior. This does not prove "no new capabilities", but it is evidence against the strongest exploration-heavy story.

#### Trace semantics are not a reliable explanation of RL gains

`Beyond Semantics` (`2505.13775`) is a particularly useful warning shot for this project. In a controlled puzzle-style setting, they find:

- correct-solution performance and trace validity can come apart;
- corrupted traces can train models almost as well as correct ones, and sometimes generalize better;
- GRPO can improve answer quality without improving trace validity.

So "the model learned better reasoning traces" is not a safe default interpretation of improved final-answer performance.

### C. There are still regimes where RL-style learning seems to add real signal

#### When pass@k is effectively zero, binary outcome RL is too weak

`Supervised Reinforcement Learning` (`2510.25992`) makes the strongest case in the supplied set for something that plain SFT and vanilla RLVR both miss. Their target regime is exactly the one we care about for small controlled models:

- the task is hard enough that the model almost never samples a correct full rollout;
- SFT overfits long demonstrations by rigid imitation;
- outcome-only RLVR has no usable signal because all rollouts are wrong.

Their fix is dense, step-wise similarity reward over expert actions. This suggests that the regimes where RL genuinely helps may be the ones where credit assignment, not just search, is the bottleneck.

#### Diversity-preserving exploration may be a route beyond mode elicitation

`LaDi-RL` (`2602.01705`) explicitly targets the diversity-collapse problem. Their claim is not merely higher pass@1, but simultaneous gains in pass@1 and pass@k by exploring in latent space instead of only in discrete token space. If that result survives broader testing, it would be one of the better current candidates for "RL is doing something more than mode elicitation."

## Current synthesis: what seems proved, plausible, and still open

### Best-supported statements

1. The exact theoretical bridge is fixed-reference KL-RL to Gibbs-reweighted distillation, not a blanket equivalence between all RL and all SFT.
2. Much of current LLM RLVR performance is consistent with distribution sharpening, search, and online data refresh over capabilities already latent in the base model.
3. Plain final-answer gains do not by themselves show better reasoning processes, better trace semantics, or support expansion.
4. Dense or structured credit assignment matters when correct full rollouts are too rare for outcome-only RL to learn from.
5. Diversity is part of the scientific object. Any method that raises pass@1 by collapsing pass@k should be interpreted differently from a method that improves both.

### Plausible but not yet proved

1. Some diversity-preserving RL variants may genuinely enlarge effective reasoning support beyond what token-space GRPO usually recovers.
2. Latent-space exploration may be a better optimizer family than discrete token RL for preserving multiple good reasoning modes.
3. In agentic settings with long horizons and stronger trajectory drift, on-policy RL may matter more than it does in single-turn math/code.

### Still open

1. How often does RL actually create successful trajectories that were not already reachable by better inference-time sampling from the base model?
2. When RL helps, is the key ingredient reward optimization, online data refresh, verifier filtering, or denser credit assignment?
3. Can any claimed RL gain be compressed back into a supervised dataset without losing most of the gain?
4. What is the right operational definition of "distribution expansion" in a reasoning setting: lower base-policy likelihood, new algorithm family, improved OOD transfer, or something else?

## Follow-up objects after the first Athena pass

The initial literature map was good enough to set the support-expansion criterion, but the collaborator then asked for deeper grounding on zero-KL RL, SePT's status, and the sourcing behind the diversity / on-policy claims. The detailed note is now `zero-kl-and-exploration.md`. The short version is:

### A. The exact reverse-KL bridge does not survive literal `beta = 0`

For `beta > 0`, the fixed-reference KL bridge is exact. As `beta -> 0`, the Gibbs targets converge to ref-weighted reward maximizers. But literal `beta = 0` removes the fixed reference from the objective and leaves a set-valued reward-maximization problem rather than a unique reverse-KL projection target.

That is why zero-KL reasoning RL should be described as an empirical algorithm family, not as a direct corollary of the fixed-reference bridge.

### B. Recent zero-KL evidence exists, but it is empirical rather than theorem-level

The strongest clean source here is `Open-Reasoner-Zero` (`2503.24290`), which explicitly studies PPO-style reasoning RL without KL regularization and reports strong stability/performance in that setting. `DeepSeek-R1` (`2501.12948`) is related but supports a slightly different point: pure RL from a base model can work, while also producing drift that later stages repair.

The safe interpretation is that explicit fixed-reference KL is not the only workable stabilizer. Strong initialization, clipping / old-policy ratios, simple rule-based rewards, and low-variance baselines can also stabilize training.

### C. SePT is self-sharpening, not importance-corrected off-policy RL

`SePT` (`2510.18814`) does not use an off-policy correction term. It samples at low temperature, trains with ordinary NLL, and refreshes the sampled data online. The paper's own theoretical framing is an occupancy-weighted forward-KL projection onto a low-temperature self-teacher.

So the real lesson from SePT is not "off-policy beats on-policy." It is that, in single-turn reasoning, online self-generated supervision plus temperature-decoupled sharpening can recover part of the gain often attributed to RL.

### D. The diversity-preserving claim is better sourced now, but still not fully proved

The best current supporting set is:

1. `LaDi-RL` (`2602.01705`)
2. `Reasoning with Exploration: An Entropy Perspective` (`2506.14758`)
3. `Rethinking Entropy Regularization in Large Reasoning Models` (`2509.25133`)
4. `Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR` (`2509.23808`)

These papers support the weaker claim that some methods preserve or broaden successful-support coverage better than standard token-space RL. They do not yet prove the stronger claim that training created successful reasoning modes that were absent from the base model's low-probability tail.

### E. The long-horizon on-policy claim is an inference across sources, not a law

The strongest supporting set is:

1. `RobotxR1` (`2505.03238`) for closed-loop embodied RL;
2. `Supervised Reinforcement Learning` (`2510.25992`) for dense guidance in hard multi-step regimes;
3. `Self-Distillation Enables Continual Learning` (`2601.19897`) for on-policy distillation in sequential learning.

This is enough to justify putting agentic / long-horizon on-policy learning on the research agenda. It is not enough to claim that on-policy RL always dominates static supervision once horizons grow.

## Athena consult update: the operational support-expansion rule

On 2026-04-12 02:57 UTC, Athena re-checked the same fixed evidence set with one narrow question: what should count as actual support expansion rather than mere probability-mass reallocation? The sharpened rule is:

- only call a result ``support expansion'' if, on a fixed hard slice, the post-RL policy emits trace-valid successful trajectories at a non-trivial rate that the pre-RL model still cannot recover under larger-k sampling/search and positive-support distillation controls.

This makes the control structure much more concrete than the first literature pass. The most useful consequences are:

### Strongest evidence, ordered from strongest to weaker

1. On held-out hard prompts, the base model still has near-zero success even after aggressive search, while the post-RL policy now succeeds directly.
2. That advantage survives best-of-N plus cloning and reward-free online self-training controls.
3. The new successful traces live in reasoning-pattern clusters that were absent or vanishingly rare in base-model samples and early RL checkpoints.
4. The gain shows up in step-level trace validity, not only in final-answer correctness.
5. pass@1 rises without pass@k or diversity collapsing.

### What does not count on its own

1. Higher final-answer accuracy alone.
2. Higher pass@1 when base-model search already nearly matches the gain.
3. The fact that RL found good trajectories at all, if best-of-N search plus cloning can already recover them.
4. Improvements that also appear under reward-free online self-training with refreshed data.
5. Longer, more natural-looking, or more semantically plausible traces.

## Implications for the puzzle project

The current maze repo is already useful because it lets us measure answer correctness, path validity, and path optimality in a fully controlled setting. The main experimental lesson from the literature is that the project should not compare "SFT" versus "RL" as a single axis. It should separately manipulate:

1. trajectory source: fixed expert traces vs online self-generated traces vs verifier-filtered best-of-N traces;
2. feedback type: none, binary outcome reward, dense step-wise reward;
3. optimizer family: pure SFT, search-plus-distill, reward-free online self-training, on-policy supervision, RLVR, later diversity-preserving RL.

That decomposition is what the first experiment contract now uses, and the new Athena rule makes three controls non-negotiable in any later ``support expansion'' claim:

1. matched large-k base-model search; and
2. matched search-plus-distill or reward-free online-refresh baselines; and
3. matched on-policy supervision on the same rollout support.

## Source list

Primary sources used for this note:

1. `2305.18290` - `Direct Preference Optimization: Your Language Model is Secretly a Reward Model`
2. `2502.06781` - `Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning`
3. `2505.13775` - `Beyond Semantics: The Unreasonable Effectiveness of Reasonless Intermediate Tokens`
4. `2510.14901` - `Reasoning with Sampling: Your Base Model is Smarter Than You Think`
5. `2510.18814` - `A Model Can Help Itself: Reward-Free Self-Training for LLM Reasoning`
6. `2510.25992` - `Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning`
7. `2601.04537` - `Not All Steps are Informative: On the Linearity of LLMs' RLVR Training`
8. `2602.01705` - `Beyond Mode Elicitation: Diversity-Preserving Reinforcement Learning via Latent Diffusion Reasoner`
9. `2507.10616` - `Scalpel vs. Hammer: GRPO Amplifies Existing Capabilities, SFT Replaces Them`
10. `2501.12948` - `DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning`
11. `2503.24290` - `Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model`
12. `2505.03238` - `RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning`
13. `2506.14758` - `Reasoning with Exploration: An Entropy Perspective`
14. `2509.25133` - `Rethinking Entropy Regularization in Large Reasoning Models`
15. `2509.23808` - `Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR`
16. `2601.18734` - `Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models`
17. `2601.19897` - `Self-Distillation Enables Continual Learning`
