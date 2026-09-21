import math

import pytest
import torch

from poreml.metrics.descriptors import (
    area,
    area_contact,
    area_interface,
    euler,
    phase_mask,
    reach_from_ends,
    saturation,
    trapped_volume,
    volume,
)
from poreml.registry import DESCRIPTORS


def field_and_mask_2d():
    # One sample, one channel, 2 rows x 4 cols. S = solid. Layout (phi):
    #   [ +1  +1  -1   S ]
    #   [ +1  -1  -1  -1 ]
    phi = torch.tensor([[[[1.0, 1.0, -1.0, 0.0], [1.0, -1.0, -1.0, -1.0]]]])
    mask = torch.tensor([[[[True, True, True, False], [True, True, True, True]]]])
    return phi, mask


def test_registered_scalar_descriptors():
    for name in ("volume", "saturation", "area_interface", "area_contact", "voxels"):
        assert name in DESCRIPTORS
    assert DESCRIPTORS.get("volume").kind == "scalar"
    assert DESCRIPTORS.get("volume").params == {"phase"}
    assert DESCRIPTORS.get("area_interface").params == frozenset()
    assert DESCRIPTORS.get("voxels").kind == "voxels"


def test_phase_mask_excludes_solid_from_both_phases():
    phi, mask = field_and_mask_2d()
    nw = phase_mask(phi, mask, "nw")
    w = phase_mask(phi, mask, "w")
    assert nw.sum().item() == 3
    assert w.sum().item() == 4
    assert not (nw | w)[..., 0, 3].item()  # the solid voxel is neither


def test_volume_counts_voxels_of_the_requested_phase():
    phi, mask = field_and_mask_2d()
    assert volume(phi, mask, phase="nw").tolist() == [3.0]
    assert volume(phi, mask, phase="w").tolist() == [4.0]
    assert volume(phi, mask).tolist() == [3.0]  # default is non-wetting


def test_saturation_is_phase_volume_over_pore_volume():
    phi, mask = field_and_mask_2d()
    assert saturation(phi, mask).tolist() == pytest.approx([3.0 / 7.0])
    assert saturation(phi, mask, phase="w").tolist() == pytest.approx([4.0 / 7.0])


def test_saturation_is_nan_for_an_all_solid_sample():
    phi = torch.zeros(1, 1, 2, 2)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    assert math.isnan(saturation(phi, mask).item())


def test_area_interface_counts_nw_to_w_faces():
    phi, mask = field_and_mask_2d()
    # NW-W faces: (0,1)-(0,2), (0,1)-(1,1), (1,0)-(1,1) -> 3
    assert area_interface(phi, mask).tolist() == [3.0]


def test_area_contact_counts_phase_to_solid_faces():
    phi, mask = field_and_mask_2d()
    # Solid at (0,3): neighbours (0,2) W and (1,3) W. No NW contact, 2 W contacts.
    assert area_contact(phi, mask, phase="nw").tolist() == [0.0]
    assert area_contact(phi, mask, phase="w").tolist() == [2.0]


def test_descriptors_are_per_sample_and_work_in_3d():
    # Batch of 2, spatial 2x2x2. Sample 0 all NW, sample 1 half NW along the last axis.
    phi = torch.ones(2, 1, 2, 2, 2)
    phi[1, ..., 1] = -1.0
    mask = torch.ones(2, 1, 2, 2, 2, dtype=torch.bool)
    assert volume(phi, mask).tolist() == [8.0, 4.0]
    assert area_interface(phi, mask).tolist() == [0.0, 4.0]


def test_scalar_descriptors_reject_multichannel_and_bad_mask():
    phi = torch.ones(1, 3, 2, 2)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="channel"):
        volume(phi, mask)
    with pytest.raises(ValueError, match="mask"):
        volume(torch.ones(1, 1, 2, 2), torch.ones(1, 2, 2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask"):
        volume(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))  # not bool
    with pytest.raises(ValueError, match="batch"):
        volume(torch.ones(2, 2), torch.ones(2, 2, dtype=torch.bool))  # no batch/channel dims


def test_reach_from_ends_follows_only_face_neighbours():
    # 3 rows x 5 cols; flow axis is the last (columns). Inlet = col 0, outlet = col 4.
    #   row 0: [ N  N  .  .  . ]   attached to inlet
    #   row 1: [ .  .  N  .  . ]   touches (0,1) only diagonally -> NOT reached
    #   row 2: [ .  .  .  N  N ]   attached to outlet
    nw = torch.tensor(
        [[[[True, True, False, False, False], [False, False, True, False, False], [False, False, False, True, True]]]]
    )
    reached = reach_from_ends(nw)
    assert reached[0, 0, 0].tolist() == [True, True, False, False, False]
    assert reached[0, 0, 1].tolist() == [False, False, False, False, False]
    assert reached[0, 0, 2].tolist() == [False, False, False, True, True]


def test_trapped_volume_is_the_phase_not_reached_from_inlet_or_outlet():
    #   row 0: [ +  +  -  -  - ]
    #   row 1: [ -  -  +  -  - ]     <- isolated NW blob of 1 voxel: trapped
    #   row 2: [ -  +  -  +  + ]     <- (2,1) isolated: trapped; (2,3),(2,4) reach outlet
    phi = torch.tensor([[[[1.0, 1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, -1.0, -1.0], [-1.0, 1.0, -1.0, 1.0, 1.0]]]])
    mask = torch.ones_like(phi, dtype=torch.bool)
    assert trapped_volume(phi, mask).tolist() == [2.0]
    # Wetting phase: (2,2) is a W voxel boxed in by NW on three sides and the wall -> 1 trapped
    assert trapped_volume(phi, mask, phase="w").tolist() == [1.0]


def test_trapped_volume_does_not_walk_through_solid():
    # NW at (0,0) and (0,2) with solid between them; (0,2) is not connected to any end.
    phi = torch.tensor([[[[1.0, 0.0, 1.0, -1.0]]]])
    mask = torch.tensor([[[[True, False, True, True]]]])
    assert trapped_volume(phi, mask).tolist() == [1.0]


def test_trapped_volume_is_per_sample_in_3d():
    phi = -torch.ones(2, 1, 3, 3, 3)
    phi[0, 0, 1, 1, 1] = 1.0  # centre voxel, isolated -> trapped
    phi[1, 0, 1, 1, :] = 1.0  # a rod spanning inlet to outlet -> connected
    mask = torch.ones_like(phi, dtype=torch.bool)
    assert trapped_volume(phi, mask).tolist() == [1.0, 0.0]


def test_area_is_registered_as_a_scalar_with_phase():
    assert "area" in DESCRIPTORS
    assert DESCRIPTORS.get("area").kind == "scalar"
    assert DESCRIPTORS.get("area").params == {"phase"}


def test_area_is_the_phase_boundary_to_other_fluid_and_solid():
    phi, mask = field_and_mask_2d()
    # NW: 3 faces to W, 0 to solid. W: 3 faces to NW, 2 to solid.
    assert area(phi, mask, phase="nw").tolist() == [3.0]
    assert area(phi, mask, phase="w").tolist() == [5.0]
    assert area(phi, mask).tolist() == [3.0]  # default is non-wetting


def test_area_does_not_count_domain_boundary_faces():
    # A lone NW voxel filling the whole domain has no neighbour on any side.
    phi = torch.ones(1, 1, 1, 1)
    mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    assert area(phi, mask).tolist() == [0.0]


def test_area_is_per_sample_in_3d():
    # Sample 0 all NW (area 0); sample 1 half NW along the last axis (one 2x2 interface).
    phi = torch.ones(2, 1, 2, 2, 2)
    phi[1, ..., 1] = -1.0
    mask = torch.ones(2, 1, 2, 2, 2, dtype=torch.bool)
    assert area(phi, mask).tolist() == [0.0, 4.0]


def test_euler_is_registered_as_a_scalar_with_phase():
    assert "euler" in DESCRIPTORS
    assert DESCRIPTORS.get("euler").kind == "scalar"
    assert DESCRIPTORS.get("euler").params == {"phase"}


def test_euler_counts_components():
    # A single blob is a ball: chi = 1. A solid cube likewise.
    phi = -torch.ones(2, 1, 3, 3, 3)
    phi[0, 0, 1, 1, 1] = 1.0
    phi[1] = 1.0
    mask = torch.ones_like(phi, dtype=torch.bool)
    assert euler(phi, mask).tolist() == [1.0, 1.0]


def test_euler_uses_6_connectivity_like_trapped_volume():
    # Two voxels touching only diagonally are two components, not one neck.
    phi = -torch.ones(1, 1, 1, 2, 2)
    phi[0, 0, 0, 0, 0] = 1.0
    phi[0, 0, 0, 1, 1] = 1.0
    mask = torch.ones_like(phi, dtype=torch.bool)
    assert euler(phi, mask).tolist() == [2.0]


def test_euler_counts_a_loop_as_zero():
    # The 8-voxel outline of a 3x3 square is a circle: chi = 0.
    phi = -torch.ones(1, 1, 1, 3, 3)
    phi[0, 0, 0] = 1.0
    phi[0, 0, 0, 1, 1] = -1.0
    mask = torch.ones_like(phi, dtype=torch.bool)
    assert euler(phi, mask).tolist() == [0.0]


def test_euler_counts_a_cavity():
    # A 3x3x3 cube with its centre removed is a spherical shell: chi = 2.
    phi = torch.ones(1, 1, 3, 3, 3)
    phi[0, 0, 1, 1, 1] = -1.0
    mask = torch.ones_like(phi, dtype=torch.bool)
    assert euler(phi, mask).tolist() == [2.0]


def test_euler_measures_the_requested_phase_and_ignores_solid():
    phi, mask = field_and_mask_2d()
    # NW: (0,0), (0,1), (1,0) form one L-shaped blob -> 1.
    # W: (0,2), (1,1), (1,2), (1,3) form one blob -> 1; the solid voxel joins neither.
    assert euler(phi, mask, phase="nw").tolist() == [1.0]
    assert euler(phi, mask, phase="w").tolist() == [1.0]


def test_euler_is_zero_for_an_empty_phase():
    phi = -torch.ones(1, 1, 2, 2, 2)
    mask = torch.ones_like(phi, dtype=torch.bool)
    assert euler(phi, mask, phase="nw").tolist() == [0.0]


def test_euler_matches_skimage_on_random_fields():
    from skimage.measure import euler_number as sk_euler

    torch.manual_seed(0)
    for shape in ((1, 4, 6, 5), (1, 7, 6), (1, 12, 12, 12)):
        phi = torch.rand(3, *shape) * 2.0 - 1.0
        mask = torch.rand(3, 1, *shape[1:]) > 0.2
        got = euler(phi, mask).tolist()
        for b in range(3):
            blob = phase_mask(phi, mask, "nw")[b, 0].numpy()
            assert got[b] == float(sk_euler(blob, connectivity=1))
