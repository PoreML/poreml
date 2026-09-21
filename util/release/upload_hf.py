"""Push `data/case` to a HuggingFace dataset repository, structure preserved.

The repository **is** the dataset directory: its root holds `drainage/`, `GDL/`, `trapping/`,
`underfill/`, the card and `assets/`, so

    hf download <repo> --repo-type dataset --local-dir data/case

reconstructs exactly the tree the benchmark's default `data.root` expects, with nothing to
move afterwards. That is the whole reason the upload is a folder push rather than a curated
export.

`upload_large_folder` is the tool for this size: it hashes and uploads in parallel, commits in
batches, retries a failed file, and keeps its own state under `<folder>/.cache/huggingface`, so
**rerunning is the resume path** — already-uploaded files are skipped after a hash check. It is
also the only uploader that copes with a 3.3 TB, 3 953-file push without holding it all in one
commit. Nothing here deletes anything on the Hub.

Run it from the repository root. The token comes from a file so it never lands in a process
listing or a log:

    uv run python util/release/upload_hf.py --repo PoreML/PoreML_data
    uv run python util/release/upload_hf.py --repo PoreML/PoreML_data --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FOLDER = REPO / "data" / "case"
DEFAULT_TOKEN_FILE = REPO.parent / "cred" / "hugginface.md"

# `upload_large_folder`'s own bookkeeping, and anything a half-finished conversion left behind.
IGNORE = ["**/.cache/**", "**/*.tmp", "**/*.repack.tmp", "**/*.h5.claim"]


def token_from(path: Path) -> str:
    """The HF token, read from a file and stripped. Never echo this."""
    token = path.read_text().strip().split()[0]
    if not token.startswith("hf_"):
        raise SystemExit(f"{path} does not look like a HuggingFace token")
    return token


def survey(folder: Path) -> tuple[int, int]:
    """(files, bytes) that the push will consider — what the ignore patterns leave."""
    n = total = 0
    for p in folder.rglob("*"):
        if p.is_file() and ".cache" not in p.parts and not p.name.endswith((".tmp", ".claim")):
            n += 1
            total += p.stat().st_size
    return n, total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="e.g. PoreML/PoreML_data")
    ap.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    ap.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    ap.add_argument("--workers", type=int, default=8, help="parallel hash/upload workers")
    ap.add_argument("--private", action="store_true", help="create the repo private if it does not exist")
    ap.add_argument("--dry-run", action="store_true", help="report what would be pushed; upload nothing")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a directory")
    files, size = survey(folder)
    print(f"folder   {folder}")
    print(f"contents {files} files, {size / 1e9:.0f} GB")
    print(f"target   https://huggingface.co/datasets/{args.repo}  (root = this folder's contents)")

    token = token_from(args.token_file)
    api = HfApi(token=token)
    who = api.whoami()
    print(f"account  {who.get('name')}  orgs={[o.get('name') for o in who.get('orgs', [])]}")

    if args.dry_run:
        print("dry run — nothing uploaded")
        return 0

    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    # Fail loudly and early on a token that can only open a pull request, rather than after
    # hours of hashing: create_commit is the same permission the folder push needs.
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    probe = ".upload_probe"
    try:
        api.create_commit(
            args.repo,
            repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=probe, path_or_fileobj=b"probe\n")],
            commit_message="probe: verify write access",
        )
        api.create_commit(
            args.repo,
            repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=probe)],
            commit_message="probe: remove",
        )
    except Exception as exc:  # noqa: BLE001 — the message is the diagnosis
        print(f"\nNO WRITE ACCESS to {args.repo}:\n  {str(exc)[:300]}", file=sys.stderr)
        print(
            "\nA fine-grained token must be scoped to the *owner* of the repo. A token scoped only "
            "to your user cannot write to an org-owned repo; grant it write on the organisation, "
            "or push to a repo in your own namespace.",
            file=sys.stderr,
        )
        return 2

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # xet handles the parallelism
    started = time.time()
    # The bound method, not the module-level function: the latter takes no `token` and would
    # fall back to whatever is cached on the machine.
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(folder),
        ignore_patterns=IGNORE,
        num_workers=args.workers,
        print_report=True,
        print_report_every=120,
    )
    dt = time.time() - started
    print(f"\ndone in {dt / 3600:.2f} h ({size / 1e6 / max(dt, 1):.0f} MB/s average)")
    print(f"https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
