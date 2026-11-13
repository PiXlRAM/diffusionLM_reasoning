from dlm.steps import (
    apply_rounding,
    licensed_constants,
    matches_result,
    question_values,
    safe_eval,
    step_token_spans,
    token_steps,
    verify,
)

QUESTION = "Shoes cost $85. Officers get a 20% discount, then an additional 25% off."
TEXT = (
    "The price after the first discount is $85 * 0.80 = $68. "
    "The price after the second discount is $68 * 0.75 = $51.00. So they pay $51."
)


def step(quote, role, expression=None, result=None, inputs=(), explicit=None, rounding=None):
    if explicit is None:
        explicit = role == "computation" and expression is not None
    return {
        "quote": quote,
        "role": role,
        "equation_explicit": explicit,
        "expression": expression,
        "rounding": rounding,
        "result": result,
        "inputs": [{"value": v, "source": s, "step": k} for v, s, k in inputs],
    }


def good_steps():
    return [
        step(
            "The price after the first discount is $85 * 0.80 = $68.",
            "computation",
            "85 * 0.80",
            "68",
            [("85", "question", None), ("0.80", "constant", None)],
        ),
        step(
            "The price after the second discount is $68 * 0.75 = $51.00.",
            "computation",
            "68 * 0.75",
            "51.00",
            [("68", "step", 0), ("0.75", "constant", None)],
        ),
        step("So they pay $51.", "restatement", None, "51", [("51", "step", 1)], explicit=False),
    ]


def test_safe_eval_and_rounding():
    assert safe_eval("85 * (1 - 0.20)") == 68.0
    assert safe_eval("1,200 / 4") == 300.0
    assert apply_rounding(8.4, "floor") == 8
    assert matches_result(8.4, "8", "floor")
    assert not matches_result(8.4, "8", None)
    try:
        safe_eval("__import__('os')")
        assert False
    except ValueError:
        pass


def test_question_values_include_number_words():
    q = "Sandy bought 1 million tokens and took a two-hour break."
    vals = question_values(q)
    assert any(abs(v - 1_000_000) < 1 for v in vals)
    assert any(abs(v - 2) < 1e-9 for v in vals)


def test_licensed_constants():
    assert 2.0 in licensed_constants("she keeps half of them")
    assert 3.0 in licensed_constants("spent thrice as much")
    assert 100.0 in licensed_constants("a 20% tip")
    assert 7.0 not in licensed_constants("no temporal words here")
    assert 0.8 in licensed_constants("discount", qvals=[0.2])


def test_verify_accepts_a_correct_graph():
    check = verify(QUESTION, TEXT, "51", good_steps())
    assert check["ok"], check["errors"]
    assert check["final_step"] == 1 and check["chain"] == [0, 1]
    assert check["spans"][0] == (0, TEXT.index("$68.") + 4)
    assert len(check["answer_mentions"]) == 2


def test_verify_rejects_inexact_division_without_rounding():
    question = "How many 5-foot boats fit in 42 feet?"
    text = "42 feet / 5 feet per boat = 8 boats."
    steps = [
        step(
            "42 feet / 5 feet per boat = 8 boats.",
            "computation",
            "42/5",
            "8",
            [("42", "question", None), ("5", "question", None)],
        )
    ]
    check = verify(question, text, "8", steps)
    assert not check["ok"]
    assert any("not 8" in e for e in check["errors"])

    steps[0]["rounding"] = "floor"
    check = verify(question, text, "8", steps)
    assert check["ok"], check["errors"]


def test_verify_result_only_and_producer_rewrite():
    question = "Uber costs $3 more than Lyft. Uber is $22. A tip is $3."
    text = "The Lyft ride costs $19. The total cost of the ride including tip is $22."
    steps = [
        step(
            "The Lyft ride costs $19.",
            "computation",
            None,
            "19",
            [("22", "question", None), ("3", "question", None)],
            explicit=False,
        ),
        step(
            "The total cost of the ride including tip is $22.",
            "computation",
            None,
            "22",
            [("19", "step", 0), ("3", "question", None)],
            explicit=False,
        ),
    ]
    check = verify(question, text, "22", steps)
    assert check["ok"], check["errors"]
    assert check["chain"] == [0, 1]

    # Restatement link is rewritten to the computation producer.
    text2 = "Lyft is $19. Lyft is $19. Then $19 + $3 = $22."
    steps2 = [
        step("Lyft is $19.", "computation", None, "19", (), explicit=False),
        step("Lyft is $19.", "restatement", None, "19", [("19", "step", 0)], explicit=False),
        step(
            "Then $19 + $3 = $22.",
            "computation",
            "19 + 3",
            "22",
            [("19", "step", 1), ("3", "question", None)],  # linked via restatement
        ),
    ]
    check = verify(question, text2, "22", steps2)
    assert check["ok"], check["errors"]
    assert steps2[2]["inputs"][0]["step"] == 0
    assert check["chain"] == [0, 2]


def test_verify_catches_bad_arithmetic_links_and_quotes():
    steps = good_steps()
    steps[1]["expression"] = "68 * 0.70"
    steps[1]["inputs"][1]["value"] = "0.70"
    assert any("not 51.00" in e for e in verify(QUESTION, TEXT, "51", steps)["errors"])

    steps = good_steps()
    steps[1]["inputs"][0]["step"] = 2  # restatement index; rewritten to producer 0
    check = verify(QUESTION, TEXT, "51", steps)
    assert check["ok"], check["errors"]
    assert steps[1]["inputs"][0]["step"] == 0

    steps = good_steps()
    steps[1]["inputs"][0] = {"value": "99", "source": "step", "step": 0}
    assert any("not the result" in e for e in verify(QUESTION, TEXT, "51", steps)["errors"])

    steps = good_steps()
    steps[0]["inputs"][0]["value"] = "86"
    steps[0]["expression"] = "86 * 0.80"
    errors = verify(QUESTION, TEXT, "51", steps)["errors"]
    assert any("not a number in the problem" in e for e in errors)

    steps = good_steps()
    steps[2]["quote"] = "So they pay $52."
    assert any("quote not found" in e for e in verify(QUESTION, TEXT, "51", steps)["errors"])


def test_constant_must_be_licensed_and_not_from_question():
    question = "Lily spent thrice $7."
    text = "3 * 7 = 21."
    steps = [
        step(
            "3 * 7 = 21.",
            "computation",
            "3 * 7",
            "21",
            [("3", "constant", None), ("7", "question", None)],
        )
    ]
    assert verify(question, text, "21", steps)["ok"]

    steps = [
        step(
            "3 * 7 = 21.",
            "computation",
            "3 * 7",
            "21",
            [("3", "constant", None), ("7", "constant", None)],
        )
    ]
    errors = verify(question, text, "21", steps)["errors"]
    assert any("appears in the problem" in e or "not licensed" in e for e in errors)


def test_ambiguous_only_for_two_computations():
    question = "Tom has 2 bags of 6 and also 10 plus 2."
    text = "2 * 6 = 12. 10 + 2 = 12. Then 12 + 1 = 13. So 13."
    steps = [
        step(
            "2 * 6 = 12.",
            "computation",
            "2 * 6",
            "12",
            [("2", "question", None), ("6", "question", None)],
        ),
        step(
            "10 + 2 = 12.",
            "computation",
            "10 + 2",
            "12",
            [("10", "question", None), ("2", "question", None)],
        ),
        step(
            "Then 12 + 1 = 13.",
            "computation",
            "12 + 1",
            "13",
            [("12", "step", 0), ("1", "constant", None)],
        ),
        step("So 13.", "restatement", None, "13", [("13", "step", 2)], explicit=False),
    ]
    check = verify(question, text, "13", steps)
    assert check["ok"], check["errors"]
    # Two different computations produce 12; restating 13 does not add a flag.
    assert any("matches computations" in f and "step 2:" in f for f in check["flags"])
    assert not any("step 3:" in f for f in check["flags"])


def test_verify_percent_and_whitespace_and_missing_answer():
    text = "The discount is $85 * 0.20 = $17."
    steps = [
        step(
            "The discount  is\n$85 * 0.20 = $17.",
            "computation",
            "85 * 0.20",
            "17",
            [("85", "question", None), ("0.20", "question", None)],
        )
    ]
    check = verify(QUESTION, text, "17", steps)
    assert check["ok"], check["errors"]
    assert check["spans"] == [(0, len(text))]

    text = "The price after the first discount is $85 * 0.80 = $68."
    check = verify(QUESTION, text, "51", good_steps()[:1])
    assert check["ok"] and check["final_step"] is None and "answer not in reasoning" in check["flags"]


def test_token_steps_majority_overlap():
    spans = [(0, 10), (12, 20)]
    offsets = [(0, 4), (4, 11), (10, 13), (13, 20), (20, 22)]
    labels = token_steps(offsets, spans)
    assert labels == [0, 0, -1, 1, -1]
    assert step_token_spans(labels, 2) == [[0, 1], [3]]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
