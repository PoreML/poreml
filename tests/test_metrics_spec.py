import pydantic
import pytest

from poreml.metrics.spec import MetricSpec


def test_error_only_names_the_error():
    assert MetricSpec(error="mae").key == "mae"


def test_channel_is_appended_with_at():
    assert MetricSpec(error="mae", channel="ux").key == "mae@ux"
    assert MetricSpec(error="mae", channel=2).key == "mae@2"


def test_descriptor_and_error_are_joined_by_slash():
    assert MetricSpec(descriptor="saturation", error="abs_err").key == "saturation/abs_err"


def test_wetting_phase_is_appended_but_default_nw_is_not():
    assert MetricSpec(descriptor="area_contact", error="rel_err", phase="w").key == "area_contact/rel_err@w"
    assert MetricSpec(descriptor="area_contact", error="rel_err", phase="nw").key == "area_contact/rel_err"


def test_channel_comes_before_phase():
    assert MetricSpec(descriptor="volume", error="pred", channel="phi", phase="w").key == "volume/pred@phi@w"


def test_explicit_name_overrides_the_canonical_key():
    assert MetricSpec(error="mae", name="my_mae").key == "my_mae"


def test_error_is_required_and_unknown_keys_are_rejected():
    with pytest.raises(pydantic.ValidationError):
        MetricSpec(descriptor="volume")
    with pytest.raises(pydantic.ValidationError):
        MetricSpec(error="mae", foo=1)


def test_phase_and_bins_are_validated():
    with pytest.raises(pydantic.ValidationError):
        MetricSpec(error="mae", phase="oil")
    with pytest.raises(pydantic.ValidationError):
        MetricSpec(descriptor="hist", error="w1", bins=0, range=(0.0, 1.0))


def test_round_trips_through_dict():
    spec = MetricSpec(descriptor="hist", error="w1", channel="umag", bins=8, range=(0.0, 0.5))
    assert MetricSpec.model_validate(spec.model_dump(mode="json")) == spec
