import os
from pathlib import Path

PROJ = Path(os.environ.get("PROJ", Path(__file__).resolve().parents[1]))
CACHE_DIR = PROJ / "cache"
RESULTS_DIR = PROJ / "results"
TRACES_DIR = RESULTS_DIR / "traces"

MODELS = {
    "llada": {
        "repo": "GSAI-ML/LLaDA-8B-Instruct",
        "revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "kind": "diffusion",
        "end_marker": "<|eot_id|>",
    },
    "qwen": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "kind": "ar",
        "end_marker": "<|im_end|>",
    },
}

LLADA_MASK_ID = 126336

# Greedy decoding. Both models get the same 384-token budget; traces that hit it are dropped.
GENERATION = {
    "llada": {"gen_length": 384, "steps": 384, "block_length": 32, "remasking": "low_confidence"},
    "qwen": {"max_new_tokens": 384, "do_sample": False},
}

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. Write the reasoning in plain text, "
    "one short sentence per step, without headings, bold text or LaTeX. "
    "Then give the final answer as \\boxed{{number}}.\n\n{question}"
)
ANSWER_PREFIX = "\nThe final answer is \\boxed{"
ANSWER_SUFFIX = "}."

ESTIMATORS = ("joint", "sequential")
