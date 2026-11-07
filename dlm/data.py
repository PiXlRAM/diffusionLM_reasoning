import re

CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def load_gsm8k(split):
    from datasets import load_dataset

    return load_dataset("openai/gsm8k", "main", split=split)


def normalize_number(text):
    s = text.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        value = float(s)
    except ValueError:
        return s
    return str(int(value)) if value == int(value) else str(value)


BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_number(text):
    match = NUMBER.search(text.replace("\\!", "").replace("{,}", ","))
    return normalize_number(match.group()) if match else None


def split_generation(text):
    """Split a model's solution into (reasoning, predicted answer, removed answer sentence).

    The reasoning ends where the sentence containing the last \\boxed{} begins, so the scorer's
    own answer prefix can be appended in its place.
    """
    matches = list(BOXED.finditer(text))
    if not matches:
        return text.strip(), None, ""
    last = matches[-1]
    cut = max(text.rfind("\n", 0, last.start()), text.rfind(". ", 0, last.start()) + 1, 0)
    return text[:cut].strip(), extract_number(last.group(1)), text[cut:].strip()


def split_reference(answer_field):
    """Split a GSM8K reference solution into (reasoning, normalized final answer)."""
    reasoning, final = answer_field.split("####")
    reasoning = CALC_ANNOTATION.sub("", reasoning).strip()
    return reasoning, normalize_number(final)
