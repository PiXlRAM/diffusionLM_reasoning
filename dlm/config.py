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
        "estimator": "sequential",
    },
    "qwen": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "kind": "ar",
        "end_marker": "<|im_end|>",
        "estimator": "exact",
    },
}
# estimator: LLaDA scores answers slot by slot; the train pilot found one pass ranks deletion
# drops differently (Spearman 0.83, threshold agreement 0.86, below the 0.95 bar).

# The answer may close with either token; the closing slot sums over both. Each model's
# preference between them varies by question (Qwen sat at p=0.5 on several), and a flip would
# otherwise count as a drop in the answer.
ANSWER_CLOSINGS = ("}", "}.")

LLADA_MASK_ID = 126336

# Greedy decoding. Both models get the same 384-token budget; traces that hit it are dropped.
# LLaDA decodes free-form: one token per step, 32-token blocks left to right, low-confidence
# remasking, stopping after the block that contains the end marker.
GENERATION = {
    "llada": {"gen_length": 384, "steps": 384, "block_length": 32, "remasking": "low_confidence"},
    "qwen": {"max_new_tokens": 384, "do_sample": False},
}

# Reasoning-as-infilling (Horvitz et al.), not used for the experiments: the canvas is
# [reasoning masks] + ANSWER_PREFIX + [answer masks], the answer decoded last. With it LLaDA pads
# its reasoning with restatements and repeated solutions to fill the reasoning region, leaving no
# clean end point. Swap it into GENERATION["llada"] to regenerate those traces.
LLADA_INFILLING = {
    "layout": "reasoning_as_infilling",
    "gen_length": 384,
    "answer_slots": 10,
    "answer_block": "after_reasoning",
    "reasoning_cut": "before_first_answer",
    "block_length": 32,
    "tokens_per_step": 1,
    "remasking": "entropy",
}

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. Write the reasoning in plain text, "
    "one short sentence per step, without headings or bold text, and without LaTeX except "
    "for the final answer. End with the final answer written exactly as \\boxed{{number}}."
    "\n\n{question}"
)
ANSWER_PREFIX = "\nThe final answer is \\boxed{"
ESTIMATORS = ("joint", "sequential")
