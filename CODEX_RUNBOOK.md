# CODEX_RUNBOOK.md — How to use Codex in this repo

This file assumes:
- Codex is running inside the dedicated Docker container
- only the FoodAI repo is mounted into the container
- dependencies come from the container image, not a repo-local virtualenv

## 1. Runtime files
Use these repo-local runtime files:
- `Dockerfile.codex`
- `.dockerignore`
- `requirements-codex-temporal.txt`
- `scripts/start_codex.sh`

Build the image from the repo root:
- `docker build -f Dockerfile.codex -t foodai-codex-temporal:latest .`

Run the container from the repo root:
- `docker run --rm -it -v "$(pwd):/workspace/foodai" -w /workspace/foodai foodai-codex-temporal:latest`

This runtime is intentionally conservative.
It is meant for:
- documentation work
- dataset inspection
- multires dataset builds
- simple temporal baselines
- temporal smoke tests

It is not meant to imply that the current temporal branch has already earned long-run escalation.

## 2. Session start workflow
At the start of a Codex session:
1. read:
   - `AGENTS.md`
   - `README.md`
   - `NEXT_DEVELOPER_HANDOFF.md`
   - `DEVELOPER_GUIDE.md`
   - `SCRIPT_CATALOG.md`
   - `PROJECT_HISTORY.md`
2. inspect:
   - `docs/repo_inventory/repo_tree.txt`
   - `docs/repo_inventory/repo_manifest.json`
3. inspect current branch and git status
4. inspect relevant reports/artifacts for the current stage before writing code
5. summarize current understanding before changing anything
6. plan the required git closeout for the run before staging anything

## 3. First useful Codex prompt
Use a prompt like this:

> You are working inside the FoodAI repo. Read AGENTS.md, README.md, NEXT_DEVELOPER_HANDOFF.md, DEVELOPER_GUIDE.md, SCRIPT_CATALOG.md, and PROJECT_HISTORY.md first. Then inspect the existing temporal multires v4.1 reports and summarize: (1) what is already trusted, (2) what the current blocker is, and (3) the best next script or experiment to implement. Do not modify files yet.

After that, use a second prompt like:

> Implement the next recommended step conservatively. Prefer additive changes. Update documentation if project state changes.

Required closeout follow-up prompt:

> Add a Git Closeout section before finishing. Show the exact staging plan, avoid `git add .`, exclude `models/temporal_multires/` by default, propose the commit message, and report the expected post-commit status.

## 4. Good Codex task types for this repo
Codex is a good fit for:
- creating new script versions
- fixing reporting/manifests
- generating summaries of existing experiment folders
- running the next diagnostic stage
- updating developer docs
- preparing automation helpers

Codex is a worse fit for:
- unconstrained architecture rewrites
- deleting historical files
- interpreting weak temporal runs as wins

## 5. Recommended default operating mode
Use Codex in:
- workspace write
- approval on request
- network disabled for model-generated shell actions unless a task really needs it

## 6. Best next local tasks for Codex
1. inspect and summarize current temporal ablation results
2. propose a `v4_2` or simpler temporal baseline path
3. implement that path
4. update docs
5. run the required git closeout step for the files changed in the run
6. optionally create repo automation helpers for experiment loops

## 7. Required git closeout step
Every experiment or documentation workflow must end with a git closeout step before commit or handoff.

That closeout must:
- run `git status --short`
- avoid `git add .`
- stage files in logical groups
- exclude `models/temporal_multires/` local artifacts by default
- state the proposed commit message
- state the expected post-commit status

## 8. Verification commands
After the container starts, verify the runtime with:
- `python --version`
- `python -c "import joblib, numpy, pandas, sklearn, torch; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available()})"`
- `python build_multires_sequence_dataset_v2.py --help`
- `python train_temporal_multires_simple_baselines_v1.py --help`
- `python train_temporal_multires_models_v4_1.py --help`
