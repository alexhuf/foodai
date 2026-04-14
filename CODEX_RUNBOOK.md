# CODEX_RUNBOOK.md — How to use Codex in this repo

This file assumes:
- Codex is running inside the dedicated Docker container
- only the FoodAI repo is mounted into the container
- project-level `.codex/config.toml` is present

## 1. Session start workflow
At the start of a Codex session:
1. read:
   - `AGENTS.md`
   - `CODEX_HANDOFF.md`
   - `README.md`
   - `NEXT_DEVELOPER_HANDOFF.md`
2. inspect current branch and git status
3. inspect relevant reports/artifacts for the current stage before writing code
4. summarize current understanding before changing anything

## 2. First useful Codex prompt
Use a prompt like this:

> You are working inside the FoodAI repo. Read AGENTS.md, CODEX_HANDOFF.md, README.md, NEXT_DEVELOPER_HANDOFF.md, SCRIPT_CATALOG.md, and PROJECT_HISTORY.md first. Then inspect the existing temporal multires v4.1 reports and summarize: (1) what is already trusted, (2) what the current blocker is, and (3) the best next script or experiment to implement. Do not modify files yet.

After that, use a second prompt like:

> Implement the next recommended step conservatively. Prefer additive changes. Update documentation if project state changes.

## 3. Good Codex task types for this repo
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

## 4. Recommended default operating mode
Use Codex in:
- workspace write
- approval on request
- network disabled for model-generated shell actions unless a task really needs it

## 5. Best next local tasks for Codex
1. inspect and summarize current temporal ablation results
2. propose a `v4_2` or simpler temporal baseline path
3. implement that path
4. update docs
5. optionally create repo automation helpers for experiment loops
