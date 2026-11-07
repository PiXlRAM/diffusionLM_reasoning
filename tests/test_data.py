from dlm.data import extract_number, normalize_number, split_generation, split_reference


def test_normalize_number():
    assert normalize_number(" 1,250 ") == "1250"
    assert normalize_number("$18.00") == "18"
    assert normalize_number("2.5") == "2.5"


def test_extract_number_from_boxed_content():
    assert extract_number("72") == "72"
    assert extract_number("\\$1{,}250") == "1250"
    assert extract_number("18 \\text{ dollars}") == "18"
    assert extract_number("no number") is None


def test_split_generation_cuts_answer_sentence():
    text = "She sold 48/2 = 24 in May.\nIn total 48 + 24 = 72.\nThe answer is \\boxed{72}."
    reasoning, pred, removed = split_generation(text)
    assert reasoning == "She sold 48/2 = 24 in May.\nIn total 48 + 24 = 72."
    assert pred == "72"
    assert removed == "The answer is \\boxed{72}."


def test_split_generation_same_line():
    reasoning, pred, _ = split_generation("So 3 + 5 = 8. Therefore the answer is \\boxed{8}.")
    assert reasoning == "So 3 + 5 = 8."
    assert pred == "8"


def test_split_generation_uses_last_boxed():
    _, pred, _ = split_generation("Try \\boxed{5}. Wait, recheck.\nFinal: \\boxed{6}.")
    assert pred == "6"


def test_split_generation_without_boxed():
    reasoning, pred, removed = split_generation("I am not sure.")
    assert (reasoning, pred, removed) == ("I am not sure.", None, "")


def test_split_reference_strips_calculator_notes():
    reasoning, answer = split_reference("48/2 = <<48/2=24>>24\n#### 72")
    assert reasoning == "48/2 = 24" and answer == "72"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
