"""Sweep a tree for strings that would identify the authors; exit 1 when any survives.

    uv run python util/release/check_anonymous.py [TREE] [--deny util/release/deny.txt]

The patterns are not in this file on purpose — a list of identifying strings is itself
identifying. They live one regular expression per line in `util/release/deny.txt`, which is
gitignored: user names, real names, e-mail domains, cluster, partition and node names, internal
codenames, reservation names. `export_anonymous.sh` runs this on the exported tree and refuses to
finish when it reports anything. Binary files are skipped; matching is case-insensitive.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP_DIRS = {".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache"}


def load_patterns(path: Path) -> re.Pattern[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise SystemExit(f"{path} holds no pattern: nothing would be checked")
    return re.compile("|".join(f"(?:{ln})" for ln in lines), re.IGNORECASE)


def files_of(tree: Path) -> list[Path]:
    """Every file to sweep. A file argument sweeps that one file — never silently nothing."""
    if tree.is_file():
        return [tree]
    if not tree.is_dir():
        raise SystemExit(f"{tree} is neither a file nor a directory")
    return sorted(p for p in tree.rglob("*") if p.is_file() and not SKIP_DIRS & set(p.relative_to(tree).parts))


def sweep(tree: Path, deny: re.Pattern[str]) -> list[tuple[Path, int, str]]:
    found = []
    for path in files_of(tree):
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for m in deny.finditer(line):
                found.append((path if path == tree else path.relative_to(tree), n, m.group(0)))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", nargs="?", type=Path, default=Path("."))
    ap.add_argument("--deny", type=Path, default=HERE / "deny.txt", help="one regular expression per line")
    args = ap.parse_args(argv)
    if not args.deny.is_file():
        raise SystemExit(f"{args.deny} is missing: write the identifying strings there, one regex per line")
    found = sweep(args.tree, load_patterns(args.deny))
    for path, n, hit in found:
        print(f"{path}:{n}: {hit}")
    n_files = len(files_of(args.tree))
    print(f"{len(found)} identifying string(s) in {n_files} file(s) under {args.tree}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
