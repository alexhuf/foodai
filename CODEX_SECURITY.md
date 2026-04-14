# CODEX_SECURITY.md — Local containment model for FoodAI

This repository is intended to be used with Codex under local containment.

## Recommended containment stack
Windows host
→ dedicated WSL2 distro
→ Docker Desktop (WSL2 backend)
→ one Codex container
→ only the FoodAI repo mounted

## Rules
- Do not run Codex directly against the full Windows host filesystem.
- Do not mount unrelated folders into the Codex container.
- Do not copy private SSH keys into the container image.
- Use SSH agent forwarding instead.
- Keep Codex confined to the repo workspace.
- Keep approvals on request by default.
- Keep network off for model-generated shell actions unless the task needs it.

## Why
This repo is artifact-heavy and contains large generated outputs, models, and archives.
Containment should reduce blast radius while preserving Git/GPU workflow.

## When to use something stronger
If you want to let Codex run broad autonomous tasks with wider permissions, use a dedicated VM instead of widening permissions on the normal workstation environment.
