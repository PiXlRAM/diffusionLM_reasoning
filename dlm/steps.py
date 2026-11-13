"""Computation graphs of reasoning traces: LLM extraction, checks in code, and token mapping.

The LLM splits a trace into steps (verbatim quotes) with a role each, and for every computation
the arithmetic expression, the stated result and where each input comes from. Nothing it returns
is trusted unchecked: quotes must be found in the trace, expressions must evaluate to the stated
result, every link must match by value, and the gold answer must be produced by some step.
"""

import ast
import math
import operator
import re

from dlm.data import normalize_number

ROLES = ("setup", "plan", "computation", "restatement", "other")
SOURCES = ("question", "step", "constant")
REMOVABLE_ROLES = ("computation", "restatement")
ROUNDINGS = ("floor", "ceil", "round")

INSTRUCTIONS = """You annotate a model's step-by-step solution to a grade-school math problem.

Split the SOLUTION into steps, in order. Each step's "quote" is a contiguous excerpt copied
character for character from the solution (keep $ signs, commas, spaces and punctuation). Steps
must not overlap and must follow the order of the text. Short connectives that belong to no step
may be left out.

Each step has one role:
- setup: restates facts given in the problem, with no new number produced by the model.
- plan: says what will be computed, without computing it.
- computation: states a new number the model has figured out. One step per stated result.
- restatement: repeats a number computed earlier, with no new arithmetic.
- other: anything else.

For every computation step set equation_explicit:
- true if the quote itself shows the arithmetic (operators, "=", or a clear inline calculation).
- false if the quote only states the result (e.g. "The Lyft ride costs $19." or
  "Betty works for 3 hours") with no operation written. Do NOT invent an expression for these.

When equation_explicit is true:
- expression: a Python arithmetic expression with only numbers, + - * / and parentheses, copying
  the arithmetic as written. Write percents as decimals or /100, as the text does.
- rounding: null if the stated result equals the expression exactly; otherwise "floor", "ceil",
  or "round" when the text truncates or rounds (e.g. how many whole boats fit).
- result: the number the step states, digits only (no $, %, commas or units).
- inputs: one entry per number in the expression, in order.
When equation_explicit is false:
- expression null, rounding null, result the stated number.
- inputs may still list the numbers that produce this result (so the dependency
  chain is recoverable), even though the equation is not written. Use [] only when
  the quote merely asserts a number with no clear precursors.

Input sources:
- "question": a number that appears in the PROBLEM (digits or number words such as "two",
  "million", "20%").
- "step": an earlier step's result. Prefer the earliest computation that produced that value,
  not a later restatement of it.
- "constant": only when a word in the PROBLEM or this step's quote licenses it: 2 for half/twice,
  3 for third/thrice, 4 for quarter, 12 for dozen, 7 for week, 60 for minutes/hours, 24 for
  hours/day, 100 for percent, or the structural 1 in complements such as 1 - 0.20. Do not mark a
  number as constant if it appears in the problem.

For restatement steps: equation_explicit false, expression null, rounding null, result the
repeated number, and one input with source "step" pointing at the earliest computation that
produced it.
For setup, plan and other steps: equation_explicit false, expression null, rounding null,
result null, inputs [].
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["quote", "role", "equation_explicit", "expression", "rounding", "result", "inputs"],
                "properties": {
                    "quote": {"type": "string"},
                    "role": {"type": "string", "enum": list(ROLES)},
                    "equation_explicit": {"type": "boolean"},
                    "expression": {"type": ["string", "null"]},
                    "rounding": {"type": ["string", "null"], "enum": list(ROUNDINGS) + [None]},
                    "result": {"type": ["string", "null"]},
                    "inputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["value", "source", "step"],
                            "properties": {
                                "value": {"type": "string"},
                                "source": {"type": "string", "enum": list(SOURCES)},
                                "step": {"type": ["integer", "null"]},
                            },
                        },
                    },
                },
            },
        }
    },
}

NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent)", re.I)
WORD_NUMBER = re.compile(
    r"\b(?:"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million"
    r")\b",
    re.I,
)
WORD_VALUES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "million": 1_000_000,
}


def user_message(question, solution):
    return f"PROBLEM:\n{question}\n\nSOLUTION:\n{solution}"


def to_float(text):
    try:
        return float(normalize_number(str(text)))
    except ValueError:
        return None


def same(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def numbers_in(text):
    """(value, start, end) for every digit number in text."""
    return [(to_float(m.group()), m.start(), m.end()) for m in NUMBER.finditer(text)]


_COMPOUND = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\s-]+"
    r"(one|two|three|four|five|six|seven|eight|nine)\b",
    re.I,
)
_CENTS = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*cents?\b", re.I)


def word_numbers_in(text):
    """Values for number words in text (two, twenty-eight, million, ...)."""
    values = [float(WORD_VALUES[m.group().lower()]) for m in WORD_NUMBER.finditer(text)]
    values += [
        float(WORD_VALUES[m.group(1).lower()] + WORD_VALUES[m.group(2).lower()])
        for m in _COMPOUND.finditer(text)
    ]
    return values


def values_stated_in(text):
    """Numeric values written in text as digits or number words."""
    return [v for v, _, _ in numbers_in(text) if v is not None] + word_numbers_in(text)


def question_values(question):
    values = [v for v, _, _ in numbers_in(question) if v is not None]
    values += word_numbers_in(question)
    values += [to_float(m.group(1)) / 100 for m in PERCENT.finditer(question)]
    values += [to_float(m.group(1)) / 100 for m in _CENTS.finditer(question)]
    # "1 million" / "1,000,000" style: digit times a scale word nearby is already covered
    # separately; also accept digit*million when both appear.
    if re.search(r"\bmillion\b", question, re.I):
        for v, _, _ in numbers_in(question):
            if v is not None and v < 1000:
                values.append(v * 1_000_000)
    return values


def licensed_constants(text, qvals=()):
    """Constants a word in text may introduce (half -> 2, thrice -> 3, ...)."""
    t = text.lower()
    out = {1.0}  # structural complement, e.g. 1 - 0.20
    if re.search(r"\bhalf\b|\bhalves\b|\btwice\b|\bdouble\b|\baverage\b|\bboth\b", t):
        out.add(2.0)
    if re.search(r"\bthrice\b|\bthird\b|\bthirds\b", t):
        out.add(3.0)
    if re.search(r"\bfourth\b|\bfourths\b|three-quarters|three quarters|a quarter of|one-quarter", t):
        out.add(4.0)
    if re.search(r"\bdozen\b", t):
        out.add(12.0)
    if re.search(r"\bweek\b|\bweeks\b", t):
        out.add(7.0)
    if re.search(r"\bminute\b|\bminutes\b|\bhour\b|\bhours\b", t):
        out.add(60.0)
    if re.search(r"\bday\b|\bdays\b|\bhour\b|\bhours\b", t):
        out.add(24.0)
    if re.search(r"%|\bpercent\b|\bcents?\b", t):
        out.add(100.0)
    # US coin face values when those coins are named.
    if re.search(r"\bquarters?\b", t):
        out.add(0.25)
    if re.search(r"\bdimes?\b", t):
        out.add(0.10)
    if re.search(r"\bnickels?\b", t):
        out.add(0.05)
    if re.search(r"\bpenn(?:y|ies)\b", t):
        out.add(0.01)
    # Complements of rates from the problem: 20% -> 0.20 and 0.80.
    for v in qvals:
        if v is not None and 0 < abs(v) < 1:
            out.add(1.0 - v)
            out.add(abs(v))
    return out


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def safe_eval(expression):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"unsupported expression: {expression!r}")

    return ev(ast.parse(expression.replace(",", ""), mode="eval"))


def apply_rounding(value, rounding):
    if rounding is None:
        return value
    if rounding == "floor":
        return math.floor(value)
    if rounding == "ceil":
        return math.ceil(value)
    if rounding == "round":
        return float(round(value))
    raise ValueError(f"unknown rounding {rounding!r}")


def matches_result(value, stated, rounding):
    target = to_float(stated)
    if target is None:
        return False
    return same(apply_rounding(value, rounding), target)


def locate(text, quote, cursor):
    """(start, end) of quote in text at or after cursor, tolerating whitespace differences."""
    i = text.find(quote, cursor)
    if i >= 0:
        return i, i + len(quote)
    words = quote.split()
    if not words:
        return None
    m = re.compile(r"\s+".join(map(re.escape, words))).search(text, cursor)
    return (m.start(), m.end()) if m else None


def computation_producers(steps, results, before, value):
    """Earlier computation steps whose result equals value."""
    return [
        k for k in range(before)
        if steps[k]["role"] == "computation" and same(results[k], value)
    ]


def earliest_producer(steps, results, before, value):
    found = computation_producers(steps, results, before, value)
    return found[0] if found else None


def verify(question, text, gold, steps):
    """Check an extraction against the trace. Returns spans, links and the answer chain.

    `errors` are failures the LLM should fix on a retry; `flags` are properties of the trace
    (ambiguous links, answer missing from the reasoning) that a retry cannot change.
    Step links that point at a restatement are rewritten to the earliest computation producer.
    """
    errors, flags = [], []
    spans, cursor = [], 0
    for i, s in enumerate(steps):
        span = locate(text, s["quote"], cursor)
        if span is None:
            errors.append(f"step {i}: quote not found in order in the solution: {s['quote']!r}")
            spans.append(None)
            continue
        spans.append(span)
        cursor = span[1]

    qvals = question_values(question)
    results = [to_float(s["result"]) if s["result"] is not None else None for s in steps]
    links = []

    for i, s in enumerate(steps):
        role = s["role"]
        explicit = bool(s.get("equation_explicit"))
        rounding = s.get("rounding")
        quote = s["quote"]
        allowed_const = licensed_constants(question + " " + quote, qvals)

        if role == "computation":
            if s["result"] is None:
                errors.append(f"step {i}: computation needs a result")
            elif explicit:
                if s["expression"] is None:
                    errors.append(f"step {i}: explicit computation needs an expression")
                elif rounding not in (None, *ROUNDINGS):
                    errors.append(f"step {i}: rounding must be null, floor, ceil, or round")
                else:
                    try:
                        value = safe_eval(s["expression"])
                        if not matches_result(value, s["result"], rounding):
                            shown = apply_rounding(value, rounding)
                            errors.append(
                                f"step {i}: {s['expression']} = {value:g}"
                                + (f" ({rounding} -> {shown:g})" if rounding else "")
                                + f", not {s['result']}"
                            )
                    except (ValueError, SyntaxError, ZeroDivisionError) as e:
                        errors.append(f"step {i}: cannot evaluate expression ({e})")
                    expr_vals = [v for v, _, _ in numbers_in(s["expression"]) if v is not None]
                    in_vals = [v for v in (to_float(x["value"]) for x in s["inputs"]) if v is not None]
                    # Inputs may omit numbers that are already in the problem or licensed
                    # constants (coin values, 100 for cents, ...). Every listed input must
                    # still appear in the expression.
                    remaining = list(expr_vals)
                    for v in in_vals:
                        for k, ev in enumerate(remaining):
                            if same(v, ev):
                                remaining.pop(k)
                                break
                        else:
                            errors.append(f"step {i}: input {v:g} is not in the expression")
                            break
                    else:
                        for ev in remaining:
                            ok_omit = any(same(ev, q) for q in qvals) or any(
                                same(ev, c) for c in allowed_const
                            )
                            if not ok_omit:
                                errors.append(
                                    f"step {i}: inputs must cover expression number {ev:g}"
                                )
                                break
            else:
                if s["expression"] is not None:
                    errors.append(f"step {i}: result-only computation must not invent an expression")
                if rounding is not None:
                    errors.append(f"step {i}: result-only computation must have null rounding")
        elif role == "restatement":
            if explicit or s["expression"] is not None or rounding is not None:
                errors.append(f"step {i}: a restatement is result-only with no expression")
            if s["result"] is None or len(s["inputs"]) != 1 or s["inputs"][0]["source"] != "step":
                errors.append(f"step {i}: a restatement needs a result and one input from a step")
        else:
            if explicit or s["inputs"] or s["result"] is not None or s["expression"] is not None:
                errors.append(f"step {i}: {role} steps take no equation, inputs or result")

        if role in REMOVABLE_ROLES and spans[i] is not None and results[i] is not None:
            lo, hi = spans[i]
            if not any(same(v, results[i]) for v in values_stated_in(text[lo:hi])):
                errors.append(f"step {i}: its result {s['result']} does not appear in its quote")

        for j, inp in enumerate(s["inputs"]):
            v = to_float(inp["value"])
            from_q = any(same(v, q) for q in qvals)
            producers = computation_producers(steps, results, i, v)
            producer = producers[0] if producers else None
            # Ambiguity only when two different computations produce this value.
            ambiguous = len(producers) > 1
            chosen = inp["step"]
            resolved = chosen
            if inp["source"] == "step" and producer is not None:
                if chosen in producers:
                    resolved = chosen  # keep LLM's choice among true producers
                else:
                    # Restatement or wrong index: rewrite to earliest producer.
                    resolved = producer
                    inp["step"] = producer
            link = {
                "step": i, "input": j, "value": v, "source": inp["source"], "from": resolved,
                "ambiguous": ambiguous,
            }
            if inp["source"] == "question" and not from_q:
                errors.append(f"step {i}: input {inp['value']} is not a number in the problem")
            elif inp["source"] == "step":
                k = resolved
                if k is None or not 0 <= k < i or not same(v, results[k]):
                    errors.append(f"step {i}: input {inp['value']} is not the result of an earlier computation")
                elif steps[k]["role"] != "computation":
                    errors.append(f"step {i}: input {inp['value']} must link to a computation, not step {k}")
            elif inp["source"] == "constant":
                if from_q:
                    errors.append(f"step {i}: input {inp['value']} appears in the problem; use source question")
                elif not any(same(v, c) for c in allowed_const):
                    errors.append(
                        f"step {i}: constant {inp['value']} is not licensed by the problem or quote wording"
                    )
                elif producers:
                    flags.append(
                        f"step {i}: constant {inp['value']} also equals computation step {producers[0]}"
                    )
            if ambiguous:
                flags.append(
                    f"step {i}: input {inp['value']} matches computations {producers}"
                )
            links.append(link)

    g = to_float(gold)
    producers = [i for i, s in enumerate(steps) if s["role"] == "computation" and same(results[i], g)]
    mentions = [(lo, hi) for v, lo, hi in numbers_in(text) if same(v, g)]
    final = producers[-1] if producers else None
    if final is None:
        if mentions:
            errors.append(f"the solution states {gold} but no computation step produces it")
        else:
            flags.append("answer not in reasoning")

    chain = set()
    if final is not None and not errors:
        stack = [final]
        while stack:
            i = stack.pop()
            if i in chain:
                continue
            chain.add(i)
            stack += [x["step"] for x in steps[i]["inputs"] if x["source"] == "step"]

    return {
        "ok": not errors,
        "errors": errors,
        "flags": flags,
        "spans": spans,
        "links": links,
        "final_step": final,
        "chain": sorted(chain),
        "answer_mentions": mentions,
    }


def token_steps(offsets, spans):
    """Step index per token (-1 for text in no step), by majority character overlap."""
    labels = []
    for a, b in offsets:
        best, best_overlap = -1, 0
        for i, span in enumerate(spans):
            if span is None:
                continue
            overlap = min(b, span[1]) - max(a, span[0])
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        labels.append(best if 2 * best_overlap >= b - a and b > a else -1)
    return labels


def step_token_spans(labels, n_steps):
    """Token positions belonging to each step."""
    out = [[] for _ in range(n_steps)]
    for t, s in enumerate(labels):
        if s >= 0:
            out[s].append(t)
    return out
