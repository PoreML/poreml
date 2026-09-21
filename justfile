# The dataset: `just download` shows what it holds and asks before fetching anything.
download *args:
    uv run poreml download {{args}}

demo:
    uv sync --group demo && uv run jupyter lab demo/demo.ipynb

# A tiny synthetic dataset, and the smoke path over it — no download needed.
fixture dir="tests/_data/tiny":
    uv run python -m tests.fixtures {{dir}}

train config="configs/smoke.yaml":
    uv run poreml train -c {{config}}

eval ckpt config="configs/smoke.yaml" split="test":
    uv run poreml eval -c {{config}} --ckpt {{ckpt}} --split {{split}}

test:
    uv run pytest

fmt:
    uv run ruff check --fix . && uv run ruff format .

lint:
    uv run ruff check . && uv run ruff format --check .
