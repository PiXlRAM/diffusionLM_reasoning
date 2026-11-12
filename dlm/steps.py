"""Computation graphs of reasoning traces: LLM extraction, checks in code, and token mapping.

The LLM splits a trace into steps (verbatim quotes) with a role each, and for every computation
the arithmetic expression, the stated result and where each input comes from. Nothing it returns
is trusted unchecked: quotes must be found in the trace, expressions must evaluate to the stated
result, every link must match by value, and the gold answer must be produced by some step.
"""

import ast
import operator
import re

from dlm.data import normalize_number

ROLES = ("setup", "plan", "computation", "restatement", "other")
SOURCES = ("question", "step", "constant")
REMOVABLE_ROLES = ("computation", "restatement")

INSTRUCTIONS = """You annotate a model's step-by-step solution to a grade-school math problem.

Split the SOLUTION into steps, in order. Each step's "quote" is a contiguous excerpt copied
character for character from the solution (keep $ signs, commas, spaces and punctuation). Steps
must not overlap and must follow the order of the text. Short connectives that belong to no step
may be left out.

Each step has one role:
- setup: restates facts given in the problem, with no new arithmetic.
- plan: says what will be computed, without computing it.
- computation: does arithmetic and states a new number. Use one step per stated result; if a
  sentence states two results, split it into two quotes.
- restatement: repeats a number computed earlier, with no new arithmetic.
- other: anything else.

For computation steps:
- expression: a Python arithmetic expression with only numbers, + - * / and parentheses, that
  reproduces the step's arithmetic as written. Write percents as decimals or /100, as the text does.
- result: the number the step states as its result, digits only (no $, %, commas or units).
- inputs: one entry per number in the expression, in order. source is "question" for a number
  given in the problem (including a percent written as a decimal), "step" with "step" set to the
  0-based index of the earlier step whose result it is, or "constant" for a fixed fact or a number
  introduced on the spot (60 minutes per hour, 2 for "twice", the 1 in 1 - 0.20).
For restatement steps: expression null, result the repeated number, and one input with source
"step" pointing at the step it repeats.
For setup, plan and other steps: expression null, result null, inputs [].
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
                "required": ["quote", "role", "expression", "result", "inputs"],
                "properties": {
                    "quote": {"type": "string"},
                    "role": {"type": "string", "enum": list(ROLES)},
                    "expression": {"type": ["string", "null"]},
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
PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent)")


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
    """(value, start, end) for every number in text."""
    return [(to_float(m.group()), m.start(), m.end()) for m in NUMBER.finditer(text)]


def question_values(question):
    values = [v for v, _, _ in numbers_in(question) if v is not None]
    values += [to_float(m.group(1)) / 100 for m in PERCENT.finditer(question)]
    return values


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


def matches_rounded(value, stated):
    """True if `value` rounds to the stated result at the stated number of decimals."""
    target = to_float(stated)
    if target is None:
        return False
    decimals = len(stated.split(".")[1]) if "." in stated else 0
    return abs(value - target) <= 0.5 * 10 ** -decimals + 1e-9 or same(value, target)


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


def verify(question, text, gold, steps):
    """Check an extraction against the trace. Returns spans, links and the answer chain.

    `errors` are failures the LLM should fix on a retry; `flags` are properties of the trace
    (ambiguous links, answer missing from the reasoning) that a retry cannot change.
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
        if role == "computation":
            if s["expression"] is None or s["result"] is None:
                errors.append(f"step {i}: computation needs an expression and a result")
            else:
                try:
                    value = safe_eval(s["expression"])
                    if not matches_rounded(value, s["result"]):
                        errors.append(f"step {i}: {s['expression']} = {value:g}, not {s['result']}")
                except (ValueError, SyntaxError, ZeroDivisionError) as e:
                    errors.append(f"step {i}: cannot evaluate expression ({e})")
                    value = None
                expr_vals = sorted(v for v, _, _ in numbers_in(s["expression"]) if v is not None)
                in_vals = sorted(v for v in (to_float(x["value"]) for x in s["inputs"]) if v is not None)
                if len(expr_vals) != len(in_vals) or not all(map(same, expr_vals, in_vals)):
                    errors.append(f"step {i}: inputs must list every number in the expression once")
        elif role == "restatement":
            if s["result"] is None or len(s["inputs"]) != 1 or s["inputs"][0]["source"] != "step":
                errors.append(f"step {i}: a restatement needs a result and one input from a step")
        elif s["inputs"] or s["result"] is not None:
            errors.append(f"step {i}: {role} steps take no inputs or result")

        if role in REMOVABLE_ROLES and spans[i] is not None and results[i] is not None:
            lo, hi = spans[i]
            if not any(same(v, results[i]) for v, _, _ in numbers_in(text[lo:hi])):
                errors.append(f"step {i}: its result {s['result']} does not appear in its quote")

        for j, inp in enumerate(s["inputs"]):
            v = to_float(inp["value"])
            from_q = any(same(v, q) for q in qvals)
            from_steps = [k for k in range(i) if same(v, results[k])]
            link = {"step": i, "input": j, "value": v, "source": inp["source"], "from": inp["step"],
                    "ambiguous": from_q + len(from_steps) > 1}
            if inp["source"] == "question" and not from_q:
                errors.append(f"step {i}: input {inp['value']} is not a number in the problem")
            elif inp["source"] == "step":
                k = inp["step"]
                if k is None or not 0 <= k < i or not same(v, results[k]):
                    errors.append(f"step {i}: input {inp['value']} is not the result of step {k}")
            elif inp["source"] == "constant" and from_steps:
                flags.append(f"step {i}: constant {inp['value']} equals the result of step {from_steps[-1]}")
            if link["ambiguous"]:
                flags.append(f"step {i}: input {inp['value']} matches several sources")
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
