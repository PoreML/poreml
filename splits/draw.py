"""The frozen draw behind splits/<campaign>/{gen,uCT,all}.yaml (Train1-9)
and splits/underfill/all.yaml (Train10). CSV inventories land beside each
split in <campaign>/csv/.

One training split per (campaign, geometry family) over the solver's small-domain
pool (domain dir 128 / 128x128x64; the 256-class runs are test-only). Each of the six
(campaign, gen|uCT) sets holds exactly 80 finished runs — 64 train / 16 val, and the
val set is **geometry-disjoint from train**: the campaign reuses a rock at a different
(M, theta) where a source pool ran dry, so val runs are drawn only from rocks that
occur once in the pool. The 16 val runs cover every (M, theta) cell exactly once
(the pool is 5 per cell) and are source-stratified by largest remainder
(28/26/26 -> 6/5/5). `<campaign>/all.yaml` is the union of the gen and uCT
splits (128 train / 32 val), so Train3/6/9 share their val runs with Train1/2 etc.
and transfer gaps stay comparable. Campaign folders are lowercase (`splits/gdl/`).
Underfill is a single 25 / 7 split over its 32-run flipchip campaign (M x theta
grid, 2-3 runs per cell, every sample used once): val cells are distinct and
capped per M and per theta level, so both sweeps stay covered.

Deterministic: re-running reproduces the published files bit-for-bit (SEED below).
The splits are frozen once published — this script is their provenance, not a tool
for redrawing them.

Usage: uv run python splits/draw.py [--check]  (--check verifies without writing)
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from poreml.data import RunRef, discover, load_split, resolve

SEED = 20260901
ROOT = Path(__file__).resolve().parent.parent / "data" / "case"  # the released dataset (`poreml download`)
OUT = Path(__file__).resolve().parent
N_TRAIN, N_VAL = 64, 16
N_TRAIN_UF, N_VAL_UF = 25, 7
SMALL_DOMAINS = {"128", "128x128x64"}
SOURCES = {
    "gen": ("blob", "poly", "sphere", "fiber"),
    "uCT": ("bentheimer", "buffberea", "castlegate", "gdl_ct", "gdl_ct_20", "gdl_ct_40"),
}
CAMPAIGNS = ("drainage", "trapping", "GDL")
FOLDERS = {"drainage": "drainage", "trapping": "trapping", "GDL": "gdl"}


def small_pool(runs: dict[str, RunRef], kind: str) -> list[RunRef]:
    """Finished small-domain runs of one geometry kind, in stable run-id order."""
    pool = [
        r
        for r in sorted(runs.values(), key=lambda r: r.run_id)
        if r.h5_path.parent.parent.name in SMALL_DOMAINS
        and r.h5_path.parent.parent.parent.name in SOURCES[kind]
        and r.status == "finished"
    ]
    if len(pool) != N_TRAIN + N_VAL:
        raise ValueError(f"expected {N_TRAIN + N_VAL} finished small runs, found {len(pool)}")
    return pool


def _quotas(pool: list[RunRef], n: int) -> dict[str, int]:
    """Largest-remainder apportionment of n val slots across sources by pool share."""
    counts = Counter(r.geometry["family"] for r in pool)
    exact = {f: n * c / len(pool) for f, c in counts.items()}
    quota = {f: int(q) for f, q in exact.items()}
    for f in sorted(exact, key=lambda f: (quota[f] - exact[f], f))[: n - sum(quota.values())]:
        quota[f] += 1
    return quota


def draw_val(pool: list[RunRef], rng: np.random.Generator) -> list[RunRef]:
    """16 val runs: singleton geometries only, one per (M, theta) cell, source quotas exact."""
    geo_uses = Counter((r.geometry["family"], r.geometry["id"]) for r in pool)
    cells: dict[tuple[float, float], list[RunRef]] = {}
    for r in pool:
        if geo_uses[(r.geometry["family"], r.geometry["id"])] == 1:
            cells.setdefault((r.params["M"], r.params["theta"]), []).append(r)
    if len(cells) != N_VAL:
        raise ValueError(f"expected {N_VAL} (M, theta) cells with single-use rocks, found {len(cells)}")
    for cands in cells.values():  # seeded shuffle once per cell, then plain DFS
        order = rng.permutation(len(cands))
        cands[:] = [cands[i] for i in order]
    remaining = _quotas(pool, N_VAL)
    chosen: list[RunRef] = []

    def dfs(todo: list[tuple[float, float]]) -> bool:
        if not todo:
            return True
        cell, *rest = sorted(todo, key=lambda c: len(cells[c]))  # most constrained cell first
        rest = [c for c in todo if c != cell]
        for r in cells[cell]:
            family = r.geometry["family"]
            if remaining[family] == 0:
                continue
            remaining[family] -= 1
            chosen.append(r)
            if dfs(rest):
                return True
            chosen.pop()
            remaining[family] += 1
        return False

    if not dfs(list(cells)):
        raise ValueError("no val draw satisfies singleton-geometry + cell cover + source quotas")
    return chosen


def draw_val_underfill(pool: list[RunRef], rng: np.random.Generator) -> list[RunRef]:
    """7 val runs: distinct (M, theta) cells, capped per M and per theta level.

    The 32-run pool uses every flipchip sample exactly once, so any draw is
    geometry-disjoint by construction. The grid is M {5, 10, 20, 30} x theta
    {30, 40, 50} with 2-3 runs per cell; caps of ceil(7 / n_levels) per level
    (2 per M, 3 per theta) force every level of both sweeps into val, and a cell
    contributes at most one val run so train keeps every cell too.
    """
    cells: dict[tuple[float, float], list[RunRef]] = {}
    for r in pool:
        cells.setdefault((r.params["M"], r.params["theta"]), []).append(r)
    if any(len(v) < 2 for v in cells.values()):
        raise ValueError(f"a cell with a single run would vanish from train: {sorted(cells)}")
    m_cap = -(-N_VAL_UF // len({m for m, _ in cells}))
    t_cap = -(-N_VAL_UF // len({t for _, t in cells}))
    order = [sorted(cells)[i] for i in rng.permutation(len(cells))]
    caps: Counter[tuple[str, float]] = Counter()
    chosen_cells: list[tuple[float, float]] = []

    def dfs(i: int) -> bool:
        if len(chosen_cells) == N_VAL_UF:
            return True
        if i == len(order):
            return False
        m, theta = order[i]
        if caps[("M", m)] < m_cap and caps[("theta", theta)] < t_cap:
            caps[("M", m)] += 1
            caps[("theta", theta)] += 1
            chosen_cells.append((m, theta))
            if dfs(i + 1):
                return True
            chosen_cells.pop()
            caps[("M", m)] -= 1
            caps[("theta", theta)] -= 1
        return dfs(i + 1)

    if not dfs(0):
        raise ValueError("no underfill val draw satisfies the level caps")
    return [cells[c][rng.integers(len(cells[c]))] for c in chosen_cells]


def n_frames(ref: RunRef) -> int:
    with h5py.File(ref.h5_path, "r") as f:
        return len(f["steps"])


def write_split(out_dir: Path, name: str, header: str, train: list[RunRef], val: list[RunRef], check: bool) -> None:
    train_geos = {(r.geometry["family"], r.geometry["id"]) for r in train}
    val_geos = {(r.geometry["family"], r.geometry["id"]) for r in val}
    if train_geos & val_geos:
        raise ValueError(f"{name}: val shares geometries with train: {sorted(train_geos & val_geos)}")

    lines = [header]
    for section, refs in (("train", train), ("val", val)):
        lines.append(f"{section}:")
        lines.extend(f"- {r.run_id}" for r in sorted(refs, key=lambda r: r.run_id))
    payload = "\n".join(lines) + "\n"

    rows = [
        {
            "run_id": r.run_id,
            "section": section,
            "campaign": r.campaign,
            "source": r.geometry["family"],
            "geometry": r.geometry["id"],
            "M": r.params["M"],
            "theta": r.params["theta"],
            "ca": "" if r.params["ca"] != r.params["ca"] else r.params["ca"],
            "n_frames": n_frames(r),
            "status": r.status,
            "finish_type": r.finish_type,
        }
        for section, refs in (("train", train), ("val", val))
        for r in sorted(refs, key=lambda r: r.run_id)
    ]

    yaml_path, csv_path = out_dir / f"{name}.yaml", out_dir / "csv" / f"{name}.csv"
    if check:
        if yaml_path.read_text() != payload:
            raise ValueError(f"{yaml_path} does not match the frozen draw")
        print(f"OK    {yaml_path.relative_to(OUT)}: matches the frozen draw")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(payload)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frames = sum(row["n_frames"] for row in rows)
    print(f"wrote {yaml_path.relative_to(OUT)}: {len(train)} train / {len(val)} val, {frames} frames total")


def main() -> None:
    check = "--check" in sys.argv
    rng = np.random.default_rng(SEED)
    for campaign in CAMPAIGNS:
        runs = discover(ROOT, campaign)
        out_dir = OUT / FOLDERS[campaign]
        parts: dict[str, tuple[list[RunRef], list[RunRef]]] = {}
        for kind in ("gen", "uCT"):
            pool = small_pool(runs, kind)
            val = draw_val(pool, rng)
            val_ids = {r.run_id for r in val}
            train = [r for r in pool if r.run_id not in val_ids]
            parts[kind] = (train, val)
        parts["all"] = (parts["gen"][0] + parts["uCT"][0], parts["gen"][1] + parts["uCT"][1])
        for kind, (train, val) in parts.items():
            sources = ", ".join(sorted({r.geometry["family"] for r in train + val}))
            rule = (
                [
                    "# unseen in train. Union of this folder's gen and uCT splits, so",
                    "# Train-on-all shares its val runs with the per-family tasks.",
                ]
                if kind == "all"
                else [
                    "# unseen in train (single-use geometries; the campaign reuses some rocks at another",
                    "# (M, theta)), one val run per (M, theta) cell, source-stratified.",
                ]
            )
            header = "\n".join(
                [
                    f"# Frozen split for the {campaign} / {kind} training task (beta-0.99 campaign).",
                    f"# Drawn by splits/draw.py (seed {SEED}) from finished small-domain runs of",
                    f"# the dataset ({sources}). {len(train)} train / {len(val)} val; val rocks are",
                    *rule,
                ]
            )
            write_split(out_dir, kind, header, train, val, check)
        # Resolve every file through the package's own loader as a final gate.
        for kind in parts:
            resolve(load_split(out_dir / f"{kind}.yaml"), runs)

    # Underfill (Train10): one 25/7 split over the 32-run flipchip campaign. Drawn
    # last so its rng consumption cannot shift the nine campaign draws above.
    runs = discover(ROOT, "underfill")
    pool = [r for r in sorted(runs.values(), key=lambda r: r.run_id) if r.status == "finished"]
    if len(pool) != N_TRAIN_UF + N_VAL_UF:
        raise ValueError(f"expected {N_TRAIN_UF + N_VAL_UF} finished underfill runs, found {len(pool)}")
    val = draw_val_underfill(pool, rng)
    val_ids = {r.run_id for r in val}
    train = [r for r in pool if r.run_id not in val_ids]
    header = "\n".join(
        [
            "# Frozen split for the underfill training task (Train10; beta-0.99 campaign).",
            f"# Drawn by splits/draw.py (seed {SEED}) from the 32 finished flipchip runs of",
            "# the dataset, underfill — M {5,10,20,30} x theta {30,40,50}, 2-3 runs per cell,",
            "# every sample used once, so val geometries are unseen in train by construction.",
            "# 25 train / 7 val; val cells are distinct, capped at 2 per M and 3 per theta",
            "# level (so every level appears). Run names carry no _b99 tag; beta is 0.99.",
        ]
    )
    write_split(OUT / "underfill", "all", header, train, val, check)
    resolve(load_split(OUT / "underfill" / "all.yaml"), runs)


if __name__ == "__main__":
    main()
