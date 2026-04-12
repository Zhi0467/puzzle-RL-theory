# Puzzle RL Theory Roadmap

Last updated: 2026-04-12 05:22 UTC
Project slug: `puzzle-rl-theory`

Remote strategy status: `github remote active`.
Canonical repo: `git@github.com:Zhi0467/puzzle-RL-theory.git`

## Current Status

- Active milestone: Milestone 0 (consult-backed theory map and experiment contract).
- Current repo reality: the codebase supports supervised maze pretraining/midtraining plus evaluation; there is no RL path yet.
- Immediate objective: land the follow-up Athena-backed theory surface for collaborator review; code planning remains deliberately paused until the collaborator signs off on the literature-plus-consult pass.
- Later intended domains: maze first, then Sudoku or related verifiable puzzles once the mechanism questions are sharper.

## Milestone 0 - Theory Map And Decision Surface

Status: active

Deliverables:

- Durable literature map for the supplied papers plus attached review.
- Clear separation between exact theory, empirical regularities, and open questions.
- Initial experiment contract for the puzzle setting.
- Explicit treatment of zero-KL RL, on-policy supervision, and exploration-first alternatives.

Gate criteria:

- The project can answer, in plain language, what RL changes relative to SFT on the actual target object.
- The write-up distinguishes distribution sharpening / search from genuine support expansion.
- The write-up distinguishes reward optimization from on-policy supervision and from zero-KL stabilization tricks.
- The maze repo’s current limitations are documented explicitly.

## Milestone 1 - Controlled Puzzle Experiment Contract

Status: pending

Deliverables:

- Task definitions for maze and at least one follow-on puzzle family.
- Baseline grid covering SFT, search-plus-distill, reward-free self-training, and RLVR.
- Baseline grid explicitly separates on-policy supervision and zero-KL vs KL RL.
- Metrics covering pass@1, pass@k, diversity, trace validity, and OOD transfer.

Gate criteria:

- Every compared method is mapped to an exact data source, verifier, and compute budget.
- At least one hypothesis cleanly separates optimizer effects from fresh-data/search effects.
- At least one hypothesis cleanly separates reward optimization from matched on-policy supervision.

## Milestone 2 - Baseline Implementations

Status: pending

Deliverables:

- Repo-backed implementations for the chosen supervised and RL-like baselines.
- Reproducible scripts for generation, training, and evaluation.

Gate criteria:

- The repo can run the baseline matrix on a small local smoke slice.
- Verifier and logging surfaces are stable enough for ablation work.

## Milestone 3 - Mechanism Experiments

Status: pending

Deliverables:

- Controlled runs that test whether RL gains come from search, on-policy refresh, reward shaping, or true support expansion.
- Comparative plots and notes for in-distribution and OOD puzzle variants.

Gate criteria:

- The report includes both pass@1 and large-pass@k.
- Claims about “new reasoning” are backed by trace/process metrics, not only final-answer accuracy.

## Milestone 4 - Consolidated Research Deliverable

Status: pending

Deliverables:

- A polished report/PDF summarizing theory, empirical takeaways, and the puzzle experiment program.
- A repo doc surface that future sessions can resume from without thread history.

Gate criteria:

- The report names what is known, what is plausible, and what is still unproved.
- The experiment plan is specific enough to implement without re-deciding the scientific object.

## Activity Log

- 2026-04-10 00:45 UTC: Initialized the durable project documentation surface for the RL-theory program inside the existing `puzzle-RL-theory` GitHub repo instead of creating a parallel local-only scaffold. The repo already contained a compact maze-search codebase with supervised `pretrain` / `midtrain` stages and evaluation, which is useful because it gives a clean controlled environment while making the current limitation explicit: the later RL comparison work has not been implemented yet. The current milestone is therefore theory-first rather than code-first.
- 2026-04-10 23:35 UTC: Finished the first literature-grounded theory pass and translated it into a concrete puzzle experiment contract. The new read is sharper than the original "RL vs SFT" framing: the evidence in the supplied paper set suggests that ordinary token-space RL gains often decompose into sampling/search, online data refresh, and credit assignment rather than one monolithic "RL effect". The current durable docs now separate safe exact claims (fixed-reference KL-RL bridge), strong empirical regularities (distribution sharpening, diversity collapse, trace-semantics mismatch), and the regimes where RL-like learning may still add something real (zero-success settings needing dense step-wise signal, and possibly diversity-preserving exploration). Added `docs/literature-map.md`, `docs/experiment-contract.md`, and the collaborator-facing PDF bundle under `outputs/literature_review_2026-04-10/`.
- 2026-04-12 02:57 UTC: Athena consult succeeded from this worker after the earlier MCP failures. Mode: `standard`. Turns: `1`. Question: what is the sharpest operational criterion for saying that RL expanded capability/support rather than merely reallocating probability mass over already-latent successful trajectories? Key insight: do not call support expansion unless the post-RL policy emits trace-valid successful trajectories on a fixed hard slice that the pre-RL model still cannot recover under larger-k search and positive-support distillation controls. The consult also sharpened the evidence ordering: matched search controls, matched search-plus-distill or reward-free online-refresh baselines, trace-cluster novelty, and pass@k/diversity preservation all matter; pass@1 alone does not. Updated `docs/research-scope.md`, `docs/literature-map.md`, `docs/experiment-contract.md`, `backlog.md`, and the collaborator-facing PDF to make that rule durable. Full consult record: `.agent/runtime/consult_history/1775775928.265109.jsonl`.
- 2026-04-12 05:17 UTC: Athena follow-up succeeded in `deep` mode with one turn on the collaborator's four follow-up questions: zero-KL RL, SePT's on/off-policy status, sources behind the diversity-preserving and long-horizon on-policy claims, and exploration-first alternatives if on-policy supervision already reproduces most of RL. The key durable changes were: the fixed-reference bridge is exact only for `beta > 0`; zero-KL success is empirical rather than theorem-level; SePT is occupancy-weighted forward-KL self-sharpening rather than importance-corrected off-policy RL; and the maze contract should explicitly compare search, reward-free self-training, on-policy supervision, KL-RL, zero-KL RL, and later exploration-first variants. Added `docs/zero-kl-and-exploration.md` and refreshed the experiment contract and collaborator-facing PDF. Full consult record: `.agent/runtime/consult_history/1775775928.265109.jsonl`.
