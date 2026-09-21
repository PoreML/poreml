"""Download the benchmark's trained checkpoints from HuggingFace: `poreml checkpoints` as a script.

    uv run python util/download_checkpoints.py --dry-run                                  # what there is, how big (7 GB)
    uv run python util/download_checkpoints.py --phase train_push --which best_rollout    # the weights the paper reports
    uv run python util/download_checkpoints.py --campaign drainage --model unet --kind gen
    uv run python util/download_checkpoints.py                                            # everything, after you agree

The size of the selection is shown first and nothing is fetched until you agree (`--yes` skips
the question); runs land under `case/train` and `case/train_push`, where the push configs, the
studies and `util/inference` look a finished run up, and rerunning resumes. The logic lives in
`poreml.checkpoints`; every option is `poreml checkpoints --help`.
"""

import sys

from poreml.cli import app

if __name__ == "__main__":
    app(args=["checkpoints", *sys.argv[1:]])
