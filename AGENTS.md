# Project: Puzzle RL Theory

This repo studies what RL contributes beyond SFT in controllable puzzle domains. The current focus is a document-first pass: map the theory, define the exact experiment objects, and only then extend the existing maze code.

## Key Docs

- `roadmap.md` - milestones, current status, activity log
- `backlog.md` - open tasks, unresolved questions
- `docs/README.md` - index of durable notes
- `docs/research-scope.md` - core questions, hypotheses, and decision criteria
- `docs/statistical-dynamics.md` - current statistics-first reframe and candidate observables
- `docs/current-code-surface.md` - what the repo currently implements

## Sub-Session Instructions

- Install: `python3 -m pip install -r requirements.txt`
- Smoke train: `python3 train.py --stage pretrain --max_iters 200 --n_train 500 --n_val 100`
- Eval: `python3 eval.py --checkpoint checkpoints/<stage>_<optimizer>/ckpt_best.pt --n_eval 200`
- Current code surface is supervised only. Do not describe any RL result as repo-backed unless the code and artifacts exist in this repo.
- Before changing training code, read `roadmap.md`, `docs/research-scope.md`, and `docs/current-code-surface.md`.

## Context Loading

- New to the project: read `roadmap.md`, then `docs/README.md`.
- Working on theory/reporting: read `docs/research-scope.md`.
- Working on experiments: read `docs/statistical-dynamics.md`, `docs/current-code-surface.md`, and the latest roadmap activity entries.
