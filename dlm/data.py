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
# Glue that only ties the sentence to the boxed answer (strip with the box under keep_operation).
_ANSWER_GLUE = re.compile(
    r",?\s*(?:resulting in|giving|which (?:gives|equals?|is)|equals?|"
    r"so(?:\s+the)?(?:\s+final)?(?:\s+answer)?\s+is|"
    r"(?:therefore\s+)?(?:the\s+)?(?:final\s+)?answer\s+is)\s*\.?\s*$",
    re.I,
)


def extract_number(text):
    match = NUMBER.search(text.replace("\\!", "").replace("{,}", ","))
    return normalize_number(match.group()) if match else None


def _sentence_cut_before_match(text, match):
    """Index where the sentence containing match begins (after prior newline or '. ')."""
    return max(text.rfind("\n", 0, match.start()), text.rfind(". ", 0, match.start()) + 1, 0)


def _strip_terminal_answer(sentence, pred):
    """Keep operation-bearing text; remove \\boxed{...} and terminal '= answer' / answer glue."""
    s = BOXED.sub("", sentence)
    # Drop leftover dots left behind by removing the box mid-phrase ("is .").
    s = re.sub(r"\s*\.\s*$", "", s)
    s = _ANSWER_GLUE.sub("", s)
    if pred is not None:
        pred_re = re.escape(pred)
        s = re.sub(rf"\s*=\s*(?:\\\$)?{pred_re}\.?\s*$", "", s)
        s = re.sub(rf",?\s+(?:is|are|was|were)\s+(?:\\\$)?{pred_re}\.?\s*$", "", s, flags=re.I)
    s = _ANSWER_GLUE.sub("", s)
    return s.strip(" \t\n,;:")


def split_generation(text, mode="cut_sentence"):
    """Split a model's solution into (reasoning, predicted answer, removed answer material).

    mode:
      - cut_sentence (original): drop the whole sentence that contains the last \\boxed{}.
      - keep_operation (revised): keep that sentence's operation text; strip only the boxed
        answer and terminal answer glue / '= gold', so final calcs are not deleted by phrasing.
    """
    matches = list(BOXED.finditer(text))
    if not matches:
        return text.strip(), None, ""
    last = matches[-1]
    pred = extract_number(last.group(1))
    cut = _sentence_cut_before_match(text, last)
    prefix, sentence = text[:cut], text[cut:]
    if mode == "cut_sentence":
        return prefix.strip(), pred, sentence.strip()
    if mode != "keep_operation":
        raise ValueError(f"unknown split mode {mode!r}")
    kept = _strip_terminal_answer(sentence, pred)
    removed = sentence.strip()
    if kept:
        reasoning = (prefix + kept).strip() if prefix.strip() else kept
    else:
        reasoning = prefix.strip()
    return reasoning, pred, removed


def split_at_first_answer(text):
    """Split at the first \\boxed{} into (reasoning, answer, answer sentence, filler).

    LLaDA's template canvas forces it to keep writing after it has answered; everything after the
    first answer sentence is filler and is kept apart from the reasoning.
    """
    first = BOXED.search(text)
    if first is None:
        return text.strip(), None, "", ""
    cut = max(text.rfind("\n", 0, first.start()), text.rfind(". ", 0, first.start()) + 1, 0)
    end = first.end() + (text[first.end() : first.end() + 1] == ".")
    return text[:cut].strip(), extract_number(first.group(1)), text[cut:end].strip(), text[end:].strip()


def split_reference(answer_field):
    """Split a GSM8K reference solution into (reasoning, normalized final answer)."""
    reasoning, final = answer_field.split("####")
    reasoning = CALC_ANNOTATION.sub("", reasoning).strip()
    return reasoning, normalize_number(final)
