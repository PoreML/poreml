import numpy as np
import pytest
import torch

from poreml.points import Points, as_grid, collate, gather, structure


def _solid():
    # 3 x 3 x 4 grid: a solid pillar at (1, 1, :) and a solid slab at z = 3.
    solid = np.zeros((3, 3, 4), dtype=bool)
    solid[1, 1, :] = True
    solid[:, :, 3] = True
    return solid


def test_structure_indexes_every_pore_voxel_in_ravel_order():
    solid = _solid()
    st = structure(solid, radii=(1,))
    assert st.shape == (3, 3, 4)
    assert st.index.tolist() == np.flatnonzero(~solid).tolist()
    assert st.pos.shape == (len(st.index), 3) and st.pos.dtype == np.float32
    # (ijk + 0.5) / max(shape): first pore voxel is (0, 0, 0)
    assert st.pos[0].tolist() == pytest.approx([0.5 / 4, 0.5 / 4, 0.5 / 4])
    assert st.pos.min() > 0 and st.pos.max() < 1


def test_structure_geometry_is_the_local_solid_fraction_and_wall_flags_solid_neighbours():
    solid = _solid()
    st = structure(solid, radii=(1, 2))
    assert st.geo.shape == (len(st.index), 2)
    ijk = np.stack(np.unravel_index(st.index, solid.shape), axis=1)
    # voxel (0, 0, 0): 27-neighbourhood (zero-padded) holds solid (1, 1, 0) and (1, 1, 1) -> 2 / 27
    at = np.flatnonzero((ijk == [0, 0, 0]).all(1))[0]
    assert st.geo[at, 0] == pytest.approx(2 / 27)
    assert st.wall[at]
    # every pore voxel here touches the pillar or the slab within one voxel: all wall
    assert st.wall.all()
    # a wide open box has interior points that are not wall
    open_box = np.zeros((7, 7, 7), dtype=bool)
    open_box[0], open_box[-1] = True, True
    st2 = structure(open_box, radii=(1,))
    assert st2.wall.sum() == 2 * 49  # the two pore layers next to the solid faces
    assert (~st2.wall).sum() == 3 * 49


def test_gather_and_dense_round_trip_with_per_channel_fill():
    solid = _solid()
    st = structure(solid, radii=(1,))
    frame = np.random.default_rng(0).standard_normal((2, 3, 3, 4)).astype(np.float32)
    values = gather(frame, st.index)
    assert values.shape == (len(st.index), 2)
    pts = Points(
        pos=torch.from_numpy(st.pos),
        feats=torch.zeros(len(st.index), 1),
        index=torch.from_numpy(st.index),
        wall=torch.from_numpy(st.wall),
        ptr=torch.tensor([0, len(st.index)]),
        shape=st.shape,
        fill=torch.tensor([-1.0, 0.0]),
    )
    grid = pts.dense(torch.from_numpy(values), pts.fill)
    assert grid.shape == (1, 2, 3, 3, 4)
    pore = torch.from_numpy(~solid)
    assert torch.equal(grid[0][:, pore], torch.from_numpy(frame)[:, pore])
    assert (grid[0, 0][~pore] == -1.0).all() and (grid[0, 1][~pore] == 0.0).all()
    mask = pts.dense(torch.ones(len(st.index), 1, dtype=torch.bool))
    assert mask.dtype == torch.bool and torch.equal(mask[0, 0], pore)


def _sample(n: int, shape=(2, 2, 2), c: int = 3):
    idx = torch.arange(n)
    pts = Points(
        pos=torch.rand(n, 3),
        feats=torch.arange(n * c, dtype=torch.float32).view(n, c),
        index=idx,
        wall=torch.zeros(n, dtype=torch.bool),
        ptr=torch.tensor([0, n]),
        shape=shape,
        fill=torch.tensor([-1.0]),
    )
    return pts, torch.full((n, 1), float(n)), torch.ones(n, 1, dtype=torch.bool), {"run": 0, "t": n}


def test_collate_concatenates_and_builds_ptr_and_meta():
    batch = collate([_sample(3), _sample(5)])
    pts, target, mask, meta = batch
    assert pts.ptr.tolist() == [0, 3, 8]
    assert pts.feats.shape == (8, 3) and pts.n_samples == 2
    assert target.shape == (8, 1) and mask.shape == (8, 1)
    assert meta["run"].tolist() == [0, 0] and meta["t"].tolist() == [3, 5]
    second = pts.sample(1)
    assert second.ptr.tolist() == [0, 5] and torch.equal(second.feats, pts.feats[3:])


def test_collate_rejects_mixed_grid_shapes():
    with pytest.raises(ValueError, match="shape"):
        collate([_sample(3, shape=(2, 2, 2)), _sample(3, shape=(2, 2, 3))])


def test_as_grid_is_identity_for_tensor_inputs_and_scatters_for_points():
    x = torch.zeros(1, 2, 2, 2, 2)
    a, b = as_grid(x, torch.ones(1), torch.zeros(1))
    assert a.item() == 1 and b.item() == 0
    pts, target, mask, _ = collate([_sample(8, shape=(2, 2, 2), c=3)])
    pred = torch.full((8, 1), 2.0)
    g_pred, g_target, g_mask = as_grid(pts, pred, target, mask)
    assert g_pred.shape == (1, 1, 2, 2, 2) and (g_pred == 2.0).all()
    assert (g_target == 8.0).all()
    assert g_mask.dtype == torch.bool and g_mask.all()


from poreml.data import SOLID_FILL, WindowDataset, discover  # noqa: E402
from poreml.points import PointWindowDataset  # noqa: E402


@pytest.fixture
def runs(fake_root):
    return list(discover(fake_root, "drainage").values())


def test_wall_is_the_first_feature_channel(runs):
    ds = PointWindowDataset(runs[:1], history=1, geometry_radii=(1, 2), wall_channel=True)
    pts, _, _, _ = ds[0]
    st = ds.structure(0)
    assert ds.n_geometry_channels == 1 + 2  # binary wall flag, then one solid fraction per radius
    assert pts.feats.shape[1] == 3 + 1  # wall + 2 fractions + phi
    assert torch.equal(pts.feats[:, 0], torch.from_numpy(st.wall.astype(np.float32)))
    assert np.allclose(pts.feats[:, 1].numpy(), st.geo[:, 0])  # fractions follow, shifted right by one


def test_point_dataset_yields_points_on_every_pore_voxel_of_the_roi(runs):
    ds = PointWindowDataset(runs[:1], history=1, geometry_radii=(1, 2))
    pts, target, mask, meta = ds[0]
    n_pore = int((~ds.solid(0)).sum())
    assert isinstance(pts, Points)
    assert pts.feats.shape == (n_pore, 2 + 1)  # 2 geometry channels + phi
    assert target.shape == (n_pore, 1) and mask.shape == (n_pore, 1) and mask.all()
    assert meta == {"run": 0, "t": 0}
    assert ds.n_geometry_channels == 2 and pts.shape == ds.solid(0).shape
    # the phi channel is the gathered ROI frame
    frame = ds.frame(0, 0)
    assert np.allclose(pts.feats[:, 2].numpy(), gather(frame, pts.index.numpy())[:, 0])
    assert pts.fill.tolist() == [SOLID_FILL]


def test_point_and_voxel_datasets_agree_through_encode_and_decode(runs):
    vox = WindowDataset(runs[:1], history=1)
    pts = PointWindowDataset(runs[:1], history=1)
    frames = [vox.frame(0, 0)]
    v_in = vox.encode(0, frames)
    p_in = pts.encode(0, frames)
    assert v_in.shape == (1, 2, *vox.solid(0).shape)
    assert p_in.n_samples == 1 and len(p_in.index) == int((~vox.solid(0)).sum())
    # decode: a point prediction scattered back equals the voxel frame with solid = fill
    truth = torch.from_numpy(vox.frame(0, 1))[None]
    p_pred = torch.from_numpy(gather(vox.frame(0, 1), p_in.index.numpy()))
    assert torch.equal(pts.decode(p_in, p_pred), truth)
    assert torch.equal(vox.decode(v_in, truth.clone()), truth)
    # voxel decode forces solid voxels to the fill even if the model wrote garbage there
    garbage = truth.clone()
    garbage[:, :, vox.solid(0)] = 7.0
    assert torch.equal(vox.decode(v_in, garbage), truth)


def test_train_points_subsamples_only_getitem(runs):
    ds = PointWindowDataset(runs[:1], history=1, train_points=5)
    pts, target, mask, _ = ds[0]
    assert len(pts.index) == 5 and target.shape == (5, 1)
    assert torch.equal(pts.index, pts.index.sort().values)  # kept in ravel order
    full = ds.encode(0, [ds.frame(0, 0)])
    assert len(full.index) == int((~ds.solid(0)).sum())


def test_point_dataset_collates_through_a_dataloader(runs):
    from torch.utils.data import DataLoader

    ds = PointWindowDataset(runs[:1], history=2)
    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate)
    pts, target, mask, meta = next(iter(loader))
    assert pts.n_samples == 2 and pts.ptr[-1] == len(pts.index) == target.shape[0]
    assert pts.feats.shape[1] == 3 + 2  # default radii (1, 2, 4) + two phi frames
    assert meta["t"].tolist() == [1, 2]


def test_point_conditions_sit_between_geometry_and_frames(runs):
    from poreml.data import ConditionSpec
    from poreml.points import PointWindowDataset

    plain = PointWindowDataset(runs, geometry_radii=(1,))
    ds = PointWindowDataset(runs, geometry_radii=(1,), conditions=(ConditionSpec(name="M", transform="log10"),))

    pts, target, mask, meta = ds[0]
    f = len(ds.channels)
    assert pts.feats.shape[1] == 1 + 1 + ds.history * f
    cond = pts.feats[:, 1]
    assert torch.unique(cond).numel() == 1
    assert cond[0].item() == pytest.approx(np.log10(ds.runs[meta["run"]].params["M"]))

    plain_pts, _, _, _ = plain[0]
    assert torch.equal(pts.feats[:, 0], plain_pts.feats[:, 0])  # geometry first
    assert torch.equal(pts.feats[:, -f:], plain_pts.feats[:, -f:])  # most recent frame last
