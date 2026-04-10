# Puzzle RL Theory Roadmap

Last updated: 2026-04-10 23:35 UTC
Project slug: `puzzle-rl-theory`

Remote strategy status: `github remote active`.
Canonical repo: `git@github.com:Zhi0467/puzzle-RL-theory.git`

## Current Status

- Active milestone: Milestone 0 (theory map and experiment contract).
- Current repo reality: the codebase supports supervised maze pretraining/midtraining plus evaluation; there is no RL path yet.
- Immediate objective: convert the first paper pass into an executable maze experiment order that separates sampling, online refresh, sparse RL, and dense credit assignment.
- Later intended domains: maze first, then Sudoku or related verifiable puzzles once the mechanism questions are sharper.

## Milestone 0 - Theory Map And Decision Surface

Status: active

Deliverables:

- Durable literature map for the supplied papers plus attached review.
- Clear separation between exact theory, empirical regularities, and open questions.
- Initial experiment contract for the puzzle setting.

Gate criteria:

- The project can answer, in plain language, what RL changes relative to SFT on the actual target object.
- The write-up distinguishes distribution sharpening / search from genuine support expansion.
- The maze repo’s current limitations are documented explicitly.

## Milestone 1 - Controlled Puzzle Experiment Contract

Status: pending

Deliverables:

- Task definitions for maze and at least one follow-on puzzle family.
- Baseline grid covering SFT, search-plus-distill, reward-free self-training, and RLVR.
- Metrics covering pass@1, pass@k, diversity, trace validity, and OOD transfer.

Gate criteria:

- Every compared method is mapped to an exact data source, verifier, and compute budget.
- At least one hypothesis cleanly separates optimizer effects from fresh-data/search effects.

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
