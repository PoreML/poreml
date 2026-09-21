"""Synthetic data in the solver's exact on-disk layout.

Tests must never depend on the multi-terabyte archive being mounted, and a developer
should be able to exercise the CLI without it either — `just fixture` writes one of
these trees.
"""

import json
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np

RUN_IDS = ["tiny_0000_M1_theta120", "tiny_0001_M1_theta130", "tiny_0002_M10_theta140"]


def write_fake_run(
    root: Path,
    campaign: str,
    run_id: str,
    shape: tuple[int, int, int] = (8, 8, 10),
    n_steps: int = 6,
    block: int = 100,
    m: float = 1.0,
    theta: float = 130.0,
    status: str = "finished",
) -> Path:
    """Write one run directory: <root>/<campaign>/runs/<run_id>/.

    The phase field starts fully wetting (-1) and invades along the last axis one slab
    per step, so a persistence baseline is wrong by exactly one slab. phi is NaN inside
    solid, matching the solver.
    """
    run_dir = root / campaign / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # zlib.crc32, not hash(): str hashing is salted per process, so hash() would
    # regenerate different geometry on every fresh pytest / `just fixture` run.
    rng = np.random.default_rng(zlib.crc32(run_id.encode()) % (2**32))
    rock = (rng.random(shape) < 0.3).astype(np.uint8)  # 1 = solid
    rock[:, :, 0] = 0  # keep the inlet slab open
    pore = rock == 0

    np.save(run_dir / f"{run_id}.npy", rock.astype(bool))

    with h5py.File(run_dir / f"drain_{run_id}.h5", "w") as f:
        f.create_dataset("rock", data=rock)
        steps = f.create_group("steps")
        for i in range(n_steps):
            phi = np.full(shape, -1.0, dtype=np.float32)
            phi[:, :, : i + 1] = 1.0  # invaded slabs
            phi[~pore] = np.nan  # the solver writes NaN inside solid
            phi_sign = (np.nan_to_num(phi, nan=-1.0) > 0).astype(np.float32)
            grp = steps.create_group(f"{i * block:09d}")
            p = np.full(shape, 0.33, dtype=np.float32) + 0.001 * i  # rises one mPa-ish per frame
            p[~pore] = np.nan
            u = np.zeros((*shape, 3), dtype=np.float32)
            u[..., 2] = 0.005 * phi_sign  # flow along the last axis, in the invaded region only
            u[~pore] = np.nan
            grp.create_dataset("phi", data=phi)
            grp.create_dataset("p", data=p)
            grp.create_dataset("rho", data=np.ones(shape, dtype=np.float32))
            grp.create_dataset("umag", data=np.abs(u[..., 2]))
            grp.create_dataset("u", data=u)

    meta = {
        "run": {"status": status, "notes": None},
        "solver": {"theta": theta, "lattice": "d3q19", "precision": "float32", "solver_commit": "fixture"},
        # the solver records the staged geometry's source path, `.../data/<family>/<size>/<stem>.npy`.
        "geometry": {
            "shape": list(shape),
            "porosity": float(pore.mean()),
            "sha256": f"{zlib.crc32(rock.tobytes()):08x}",
            "source": f"/synthetic/data/tiny/{shape[0]}/{run_id.rsplit('_', 2)[0]}.npy",
        },
        "environment": {"backend": "cpu"},
        "progress": {"step": (n_steps - 1) * block},
        "extra": {
            "M": m,
            "ca": 1e-5,
            "protocol": "synthetic fixture",
            "block": block,
            "regions": {"inbuf": [0, 1], "rock": [1, shape[2]]},
            **({"finish_type": "pv_cap"} if status == "finished" else {}),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / f"{run_id}_geometry.json").write_text(
        json.dumps({"geometry_type": "synthetic", "convention": "True = solid", "shape": list(shape)}, indent=2)
    )
    return run_dir


def write_fake_dataset(root: Path, campaign: str = "drainage") -> list[str]:
    """Write the three-run fixture dataset used by the smoke config and split."""
    params = [(1.0, 120.0), (1.0, 130.0), (10.0, 140.0)]
    for run_id, (m, theta) in zip(RUN_IDS, params, strict=True):
        write_fake_run(root, campaign, run_id, m=m, theta=theta)
    return list(RUN_IDS)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/_data/tiny")
    write_fake_dataset(target)
    print(f"wrote fixture dataset to {target}")
