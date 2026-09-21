import pytest

from poreml.registry import DESCRIPTORS, ERRORS, MODELS, TASKS, Registry


def test_register_and_get_round_trip():
    reg = Registry("widget")

    @reg.register("gizmo")
    class Gizmo:
        pass

    assert reg.get("gizmo") is Gizmo
    assert reg.names() == ["gizmo"]
    assert "gizmo" in reg


def test_register_returns_the_object_unchanged():
    reg = Registry("widget")

    def fn():
        return 1

    decorated = reg.register("fn")(fn)
    assert decorated is fn


def test_get_unknown_name_lists_registered_options():
    reg = Registry("widget")
    reg.register("alpha")(object)
    reg.register("beta")(object)

    with pytest.raises(KeyError) as excinfo:
        reg.get("gamma")

    message = str(excinfo.value)
    assert "gamma" in message
    assert "alpha" in message
    assert "beta" in message


def test_duplicate_registration_raises():
    reg = Registry("widget")
    reg.register("alpha")(object)

    with pytest.raises(ValueError, match="already registered"):
        reg.register("alpha")(object)


def test_names_are_sorted():
    reg = Registry("widget")
    reg.register("zeta")(object)
    reg.register("alpha")(object)

    assert reg.names() == ["alpha", "zeta"]


def test_module_level_registries_exist_and_are_distinct():
    assert {TASKS.kind, MODELS.kind, DESCRIPTORS.kind, ERRORS.kind} == {"task", "model", "descriptor", "error"}
    assert TASKS is not MODELS
