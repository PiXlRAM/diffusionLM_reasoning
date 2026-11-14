from dlm.edits import apply_final_result_plant, last_result_match, plant_value


def test_last_result_match_takes_rhs():
    quote = "she earns 15 * $5 = $75."
    m = last_result_match(quote, "75")
    assert m is not None and m.group() == "75"
    assert last_result_match(quote, "15").group() == "15"


def test_plant_value_same_length_and_not_gold():
    assert plant_value("75", "75") != "75"
    assert len(plant_value("75", "80")) == 2
    assert plant_value("75", "76") != "76"


def test_apply_final_result_plant():
    reasoning = "In total she earns 15 * 5 = 75 dollars."
    span = (0, len(reasoning))
    out = apply_final_result_plant(reasoning, span, "75", "76")
    assert out is not None
    new, a, b = out
    assert new == "In total she earns 15 * 5 = 76 dollars."
    assert new[a:b] == "76"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
