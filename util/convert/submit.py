"""Submit the pending zstd conversions of the solver's case as one CPU-only SLURM array.

`poreml convert scan` decides what is pending (finished, in scope, no current copy under
data/case); this writes a manifest (array index -> source run dir) and submits
util/convert/convert_task.slurm over it. Resubmitting is safe: a converted run is skipped
by the scan, a torn copy never gets renamed into place, and every task re-verifies. Run
`uv run poreml convert ledger` afterwards to refresh the solver's case/h5_todo.md.

Usage:
  uv run python util/convert/submit.py [--dry-run] [--only SUBSTR] [--throttle 32] [--src ..] [--dst ..]
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "util/convert/_manifests"
LOGS = REPO / "util/convert/_logs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on the source run path (e.g. 'trapping' or 'blob/128')")
    ap.add_argument("--throttle", type=int, default=32, help="concurrent array tasks (each ~35 MB/s in + out on the share)")
    ap.add_argument("--src", type=Path, default=REPO / "../solver/case")
    ap.add_argument("--dst", type=Path, default=REPO / "data/case")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from poreml.convert import scan

    src, dst = args.src.resolve(), args.dst.resolve()
    entries = scan(src, dst)
    pending = [e for e in entries if e["state"] == "pending" and args.only in e["src"]]
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["state"]] = counts.get(e["state"], 0) + 1
    print(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"{len(pending)} to convert" + (f" (filtered by {args.only!r})" if args.only else ""))
    if not pending or args.dry_run:
        for e in pending[:20]:
            print(f"  {e['rel']}")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    MANIFESTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    manifest = MANIFESTS / f"convert_{stamp}.tsv"
    manifest.write_text("".join(f"{i}\t{e['src']}\t{dst / e['rel']}\n" for i, e in enumerate(pending)))
    cmd = [
        "sbatch",
        "--job-name=poreml_convert",
        f"--array=0-{len(pending) - 1}%{args.throttle}",
        f"--output={LOGS}/%A_%a.log",
        f"--export=ALL,POREML_REPO={REPO},MANIFEST={manifest}",
        str(REPO / "util/convert/convert_task.slurm"),
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode == 0 and (m := re.search(r"Submitted batch job (\d+)", result.stdout)):
        (MANIFESTS / f"job_{m.group(1)}.tsv").symlink_to(manifest.name)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
