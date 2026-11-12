from dlm.steps import safe_eval, step_token_spans, token_steps, verify

QUESTION = "Shoes cost $85. Officers get a 20% discount, then an additional 25% off."
TEXT = ("The price after the first discount is $85 * 0.80 = $68. "
        "The price after the second discount is $68 * 0.75 = $51.00. So they pay $51.")


def step(quote, role, expression=None, result=None, inputs=()):
    return {"quote": quote, "role": role, "expression": expression, "result": result,
            "inputs": [{"value": v, "source": s, "step": k} for v, s, k in inputs]}


def good_steps():
    return [
        step("The price after the first discount is $85 * 0.80 = $68.", "computation", "85 * 0.80", "68",
             [("85", "question", None), ("0.80", "constant", None)]),
        step("The price after the second discount is $68 * 0.75 = $51.00.", "computation", "68 * 0.75",
             "51.00", [("68", "step", 0), ("0.75", "constant", None)]),
        step("So they pay $51.", "restatement", None, "51", [("51", "step", 1)]),
    ]


def test_safe_eval():
    assert safe_eval("85 * (1 - 0.20)") == 68.0
    assert safe_eval("1,200 / 4") == 300.0
    try:
        safe_eval("__import__('os')")
        assert False
    except ValueError:
        pass


def test_verify_accepts_a_correct_graph():
    check = verify(QUESTION, TEXT, "51", good_steps())
    assert check["ok"], check["errors"]
    assert check["final_step"] == 1 and check["chain"] == [0, 1]
    assert check["spans"][0] == (0, TEXT.index("$68.") + 4)
    assert len(check["answer_mentions"]) == 2


def test_verify_catches_bad_arithmetic_links_and_quotes():
    steps = good_steps()
    steps[1]["expression"] = "68 * 0.70"
    steps[1]["inputs"][1]["value"] = "0.70"
    assert any("not 51.00" in e for e in verify(QUESTION, TEXT, "51", steps)["errors"])

    steps = good_steps()
    steps[1]["inputs"][0]["step"] = 2
    assert any("not the result of step 2" in e for e in verify(QUESTION, TEXT, "51", steps)["errors"])

    steps = good_steps()
    steps[0]["inputs"][0]["value"] = "86"
    steps[0]["expression"] = "86 * 0.80"
    errors = verify(QUESTION, TEXT, "51", steps)["errors"]
    assert any("not a number in the problem" in e for e in errors)

    steps = good_steps()
    steps[2]["quote"] = "So they pay $52."
    assert any("quote not found" in e for e in verify(QUESTION, TEXT, "51", steps)["errors"])


def test_verify_percent_and_whitespace_and_missing_answer():
    text = "The discount is $85 * 0.20 = $17."
    steps = [step("The discount  is\n$85 * 0.20 = $17.", "computation", "85 * 0.20", "17",
                  [("85", "question", None), ("0.20", "question", None)])]
    check = verify(QUESTION, text, "17", steps)
    assert check["ok"], check["errors"]
    assert check["spans"] == [(0, len(text))]

    text = "The price after the first discount is $85 * 0.80 = $68."
    check = verify(QUESTION, text, "51", good_steps()[:1])
    assert check["ok"] and check["final_step"] is None and "answer not in reasoning" in check["flags"]


def test_ambiguous_link_is_flagged():
    question = "Tom has 12 apples and buys 2 bags of 6."
    text = "2 * 6 = 12. 12 + 12 = 24."
    steps = [
        step("2 * 6 = 12.", "computation", "2 * 6", "12", [("2", "question", None), ("6", "question", None)]),
        step("12 + 12 = 24.", "computation", "12 + 12", "24", [("12", "question", None), ("12", "step", 0)]),
    ]
    check = verify(question, text, "24", steps)
    assert check["ok"], check["errors"]
    assert sum(l["ambiguous"] for l in check["links"]) == 2


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
