import tempfile
from pathlib import Path

from dlm.cache import PassCache, make_key


def test_key_ignores_field_order():
    assert make_key({"a": 1, "b": [1, 2]}) == make_key({"b": [1, 2], "a": 1})


def test_key_changes_with_config():
    base = {"revision": "abc", "input_ids": [1, 2, 3]}
    assert make_key(base) != make_key({**base, "revision": "abd"})
    assert make_key(base) != make_key({**base, "input_ids": [1, 2]})


def test_roundtrip_and_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "passes.sqlite"
        value = {"logprob": -0.123456789012345, "per_slot": [-0.1, -0.023456789012345]}
        cache = PassCache(path)
        key = make_key({"x": 1})
        assert cache.get(key) is None
        cache.put(key, {"question": 0}, value)
        assert cache.get(key) == value
        cache.close()
        reopened = PassCache(path)
        assert reopened.get(key) == value
        assert len(reopened) == 1
        reopened.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
