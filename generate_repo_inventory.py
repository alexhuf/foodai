#!/usr/bin/env python3
"""
generate_repo_inventory.py

Create a full directory inventory of the project repo.

Outputs:
- repo_tree.txt
- repo_manifest.json
- file_extensions.csv
- largest_files.csv

Usage:
    python generate_repo_inventory.py --root . --output-dir docs/repo_inventory
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, List

DEFAULT_IGNORES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "node_modules",
}


def should_ignore(path: Path, ignore_names: set[str]) -> bool:
    return any(part in ignore_names for part in path.parts)


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def walk_repo(root: Path, ignore_names: set[str]) -> tuple[list[Path], list[Path]]:
    dirs: list[Path] = []
    files: list[Path] = []
    for path in root.rglob("*"):
        if should_ignore(path.relative_to(root), ignore_names):
            continue
        if path.is_dir():
            dirs.append(path)
        elif path.is_file():
            files.append(path)
    return sorted(dirs), sorted(files)


def build_tree_lines(root: Path, ignore_names: set[str], max_depth: int | None = None) -> list[str]:
    lines: list[str] = [root.name + "/"]

    def rec(current: Path, prefix: str = "", depth: int = 0) -> None:
        if max_depth is not None and depth >= max_depth:
            return
        children = []
        for child in sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            rel = child.relative_to(root)
            if should_ignore(rel, ignore_names):
                continue
            children.append(child)

        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            branch = "└── " if is_last else "├── "
            label = child.name + ("/" if child.is_dir() else "")
            lines.append(prefix + branch + label)
            if child.is_dir():
                extension = "    " if is_last else "│   "
                rec(child, prefix + extension, depth + 1)

    rec(root)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a full project repo inventory.")
    parser.add_argument("--root", default=".", help="Repo root directory.")
    parser.add_argument("--output-dir", default="repo_inventory", help="Output directory.")
    parser.add_argument("--max-depth", type=int, default=None, help="Optional tree max depth.")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden directories/files instead of ignoring common tooling dirs.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ignore_names = set() if args.include_hidden else set(DEFAULT_IGNORES)

    dirs, files = walk_repo(root, ignore_names)
    tree_lines = build_tree_lines(root, ignore_names, max_depth=args.max_depth)

    ext_counter: Counter[str] = Counter()
    largest_files: List[dict] = []
    manifest_files: List[dict] = []

    for f in files:
        stat = f.stat()
        ext = f.suffix.lower() or "<no_ext>"
        rel = f.relative_to(root)
        ext_counter[ext] += 1
        largest_files.append({
            "path": str(rel),
            "size_bytes": stat.st_size,
            "size_human": human_size(stat.st_size),
        })
        manifest_files.append({
            "path": str(rel),
            "name": f.name,
            "suffix": ext,
            "size_bytes": stat.st_size,
        })

    largest_files = sorted(largest_files, key=lambda x: x["size_bytes"], reverse=True)[:100]

    manifest = {
        "root": str(root),
        "output_dir": str(output_dir),
        "directory_count": len(dirs),
        "file_count": len(files),
        "ignored_names": sorted(ignore_names),
        "tree_max_depth": args.max_depth,
        "extensions": dict(sorted(ext_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "largest_files_top_100": largest_files,
        "files": manifest_files,
    }

    (output_dir / "repo_tree.txt").write_text("\n".join(tree_lines) + "\n", encoding="utf-8")
    (output_dir / "repo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with (output_dir / "file_extensions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["extension", "count"])
        for ext, count in sorted(ext_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([ext, count])

    with (output_dir / "largest_files.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "size_bytes", "size_human"])
        for row in largest_files:
            writer.writerow([row["path"], row["size_bytes"], row["size_human"]])

    print(f"Wrote inventory to: {output_dir}")


if __name__ == "__main__":
    main()
