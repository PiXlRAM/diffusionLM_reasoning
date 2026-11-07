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


def split_reference(answer_field):
    """Split a GSM8K reference solution into (reasoning, normalized final answer)."""
    reasoning, final = answer_field.split("####")
    reasoning = CALC_ANNOTATION.sub("", reasoning).strip()
    return reasoning, normalize_number(final)
