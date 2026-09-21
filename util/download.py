"""Download the PoreML dataset from HuggingFace: `poreml download` as a script.

    uv run python util/download.py --dry-run                      # the whole dataset: what it holds, how big it is
    uv run python util/download.py --campaign drainage --size 128 # one campaign at one resolution
    uv run python util/download.py --split splits/gdl/gen.yaml    # exactly the runs one split names
    uv run python util/download.py                                # everything (3.3 TB), after you agree

The size of the selection is shown first and nothing is fetched until you agree (`--yes` skips
the question); files land under `data/case` in the layout every config's `data.root` expects, and
rerunning resumes. The logic lives in `poreml.download`; every option is `poreml download --help`.
"""

import sys

from poreml.cli import app

if __name__ == "__main__":
    app(args=["download", *sys.argv[1:]])
