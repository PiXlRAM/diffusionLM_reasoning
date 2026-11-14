"""Planted edits on reasoning traces, driven by the step graph."""

import re

from dlm.data import NUMBER, normalize_number
from dlm.steps import same, to_float


def last_result_match(quote, result):
    """Last digit-span in quote whose value equals the step result."""
    target = to_float(result)
    if target is None:
        return None
    hits = [m for m in NUMBER.finditer(quote) if same(to_float(m.group()), target)]
    return hits[-1] if hits else None


def plant_value(result, gold):
    """A wrong number, preferring the same digit-string length as `result`."""
    v = to_float(result)
    g = to_float(gold)
    if v is None:
        raise ValueError(f"cannot plant from {result!r}")
    if v == int(v):
        iv, ig = int(v), int(g) if g is not None and g == int(g) else None
        src = str(iv)
        for delta in (1, -1, 2, -2, 10, -10, 5, -5):
            p = iv + delta
            if p < 0 or p == iv or p == ig:
                continue
            if len(str(p)) == len(src):
                return str(p)
        for delta in (1, -1, 2, -2, 10, -10):
            p = iv + delta
            if p >= 0 and p != iv and p != ig:
                return str(p)
        return str(iv + 11)
    # decimals: nudge the last digit
    text = normalize_number(str(result))
    if "." in text:
        whole, frac = text.split(".", 1)
        digits = list(frac)
        for i in range(len(digits) - 1, -1, -1):
            d = int(digits[i])
            for nd in ((d + 1) % 10, (d + 2) % 10, (d + 3) % 10):
                digits[i] = str(nd)
                cand = f"{whole}.{''.join(digits)}"
                if not same(to_float(cand), v) and not same(to_float(cand), g):
                    return cand
    return plant_value(str(int(v)), gold)


def apply_final_result_plant(reasoning, span, result, planted):
    """Replace the final step's stated result inside `span` with `planted`.

    Returns (new_reasoning, abs_start, abs_end) or None if the result cannot be located.
    """
    lo, hi = span
    quote = reasoning[lo:hi]
    m = last_result_match(quote, result)
    if m is None:
        return None
    a, b = lo + m.start(), lo + m.end()
    # Prefer writing the planted value with the same decoration (commas) only when lengths match.
    written = planted
    return reasoning[:a] + written + reasoning[b:], a, b
