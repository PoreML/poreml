import json

from poreml.metrics import json_safe


def test_json_safe_turns_non_finite_floats_into_null():
    payload = {"metrics": {"mae": float("nan"), "iou": 0.5}, "bounds": [float("inf"), float("-inf"), 3]}

    assert json_safe(payload) == {"metrics": {"mae": None, "iou": 0.5}, "bounds": [None, None, 3]}


def test_json_safe_output_has_no_bare_nan_token():
    def reject(token: str) -> float:
        raise AssertionError(f"non-JSON token {token!r} was written")

    text = json.dumps(json_safe({"mae": float("nan"), "iou": float("inf")}))

    assert json.loads(text, parse_constant=reject) == {"mae": None, "iou": None}
