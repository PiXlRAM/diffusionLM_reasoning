from dlm.models import Item


def test_hide_reasoning_offsets_by_prompt_and_keeps_tokens():
    item = Item(prompt=[1, 2], reasoning=[10, 11, 12, 13], prefix=[5], slots=[7, 8])
    hidden = item.hide_reasoning(1, 3)
    assert hidden.hidden == [3, 4]
    assert hidden.context() == item.context()
    assert item.hidden == []


def test_hide_reasoning_accumulates_spans():
    item = Item(prompt=[1], reasoning=[10, 11, 12, 13], prefix=[], slots=[7])
    assert item.hide_reasoning(0, 1).hide_reasoning(3, 4).hidden == [1, 4]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
