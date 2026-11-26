# Agent instructions

## Setup and validation
- Install: `pip install .`
- Test: `python -m unittest discover -s tests`
- CI runs offline tests only. Tests never call RunPod or any other service, and never spend money.

## Branching and pull requests
- Start from the latest `main`. Branch as `<tool>/<topic>`. Use Conventional Commits.
- Agent-authored PR bodies begin with `Agent: <role>[<tool>#<host>] | Thread: <thread> @ heocr-compute`.
- Do not merge your own PR. Every PR waits for green CI and maintainer review.

## Hard boundaries
- This repository is public. No credentials, no local machine paths, no names of private datasets or
  of private evaluation material, in code, tests, docs, commit messages or PR text. Grep the diff before
  the first push.
- What may be uploaded is decided by `policy.py`: open roots (an allowlist of categories) and a local
  denylist. The machine-specific files (`~/.config/heocr/remote_open_roots.toml`,
  `~/.config/heocr/remote_denylist.txt`) are never committed. Tests use invented paths only.
- Run state (the spend ledger, results) lives outside the repository.
- Changing the account-balance reserve in `budget.py` needs maintainer approval.
