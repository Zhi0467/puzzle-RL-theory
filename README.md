# Puzzle RL Theory

This repository studies what reinforcement learning is actually buying over supervised fine-tuning in small, fully verifiable puzzle domains. The immediate goal is doc-first: pin down the current theory and empirical claims around RL vs SFT, exploration, capability expansion vs mode elicitation, and the relation between RL and sampling before extending the code.

## Current State

- The codebase already contains a compact maze-search training/evaluation scaffold.
- Current training stages are `pretrain` and `midtrain`; both are supervised.
- There is no RL implementation in the repo yet.
- The active work is a theory-and-experiment-design pass that will define the later maze/Sudoku study.

## Where To Look

- `roadmap.md`: milestone status and chronological activity log.
- `backlog.md`: open questions and concrete next tasks.
- `docs/README.md`: durable research notes and experiment-planning docs.
- `train.py` / `eval.py`: current maze training and evaluation surface.

## Near-Term Goal

Use this repo as the controlled testbed for comparing:

- pure SFT / distillation,
- search plus distillation,
- reward-free self-training,
- RL with verifiable rewards,
- and later diversity-preserving or latent-space RL variants.

The intended scientific object is not just benchmark accuracy. It is the mechanism question: when does on-policy search plus verification create information that fixed demonstration data cannot efficiently exploit?
