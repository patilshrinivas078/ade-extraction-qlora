"""
EXP-03: LLM-as-a-judge semantic evaluation with DeepEval / G-Eval.

Run this script twice:
    1. MODEL_TYPE = "base"
    2. MODEL_TYPE = "fine_tuned"

Each run evaluates the corresponding predictions from EXP-01 and appends/
updates only that model's aggregate results in:
    results/EX03 comparison.json

DeepEval prints its own evaluation output to the CLI. This script additionally
stores Average Score and Pass Rate locally and in W&B.

No individual test-case results are stored.
"""

import json
from pathlib import Path

import wandb
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.evaluate import CacheConfig, DisplayConfig, AsyncConfig


INPUT_FILE = Path("results/EX01 comparison.json")
OUTPUT_FILE = Path("results/EX03 comparison.json")

MODEL_TYPE = "fine_tuned"  # Change to "fine_tuned" for the second run.
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7
LIMIT = None  # Use None for all 500 test examples.

JUDGE_STEPS = [
    "Evaluate the predicted drug-ADE relations against the gold annotation and the input sentence.",
    "Judge whether each predicted drug-ADE relation is clinically correct, supported by the sentence, and attributed to the correct drug.",
    "Do not penalize the model for omitting some gold drug-ADE relations; completeness and recall are evaluated separately by the deterministic metrics.",
    "Allow harmless lexical differences, singular/plural differences, synonymous clinical wording, and reasonable adverse-effect span-boundary differences when the underlying clinical relation is the same.",
    "Do not penalize concise output when the predicted relations are correct and supported.",
    "If the gold contains one or more adverse events and the prediction is empty, treat this as a major extraction failure and assign a very low score.",
    "If the gold contains no adverse events and the prediction is empty, treat the prediction as correct.",
    "Penalize unsupported, clinically incorrect, or hallucinated drug-ADE relations.",
    "Penalize assigning an adverse effect to the wrong drug.",
    "Do not give credit merely because a drug or effect appears; the predicted drug-effect relationship must itself be clinically correct.",
    "Treat invalid or unusable output as incorrect."
]


def pairs_to_json(pairs):
    return json.dumps(
        {"adverse_events": [{"drug": d, "effect": e} for d, e in pairs]},
        ensure_ascii=False,
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}")

    if MODEL_TYPE not in {"base", "fine_tuned"}:
        raise ValueError("MODEL_TYPE must be 'base' or 'fine_tuned'.")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    examples = data[f"{MODEL_TYPE}_model"]["predictions"]

    if LIMIT is not None:
        examples = examples[:LIMIT]

    test_cases = [
        LLMTestCase(
            input=ex["input_sentence"],
            actual_output=pairs_to_json(ex["predicted"]),
            expected_output=pairs_to_json(ex["gold"]),
        )
        for ex in examples
    ]

    metric = GEval(
        name="ADE Semantic Correctness",
        model=JUDGE_MODEL,
        evaluation_steps=JUDGE_STEPS,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
    )

    print(
        f"Running EXP-03: {MODEL_TYPE} model | "
        f"{len(test_cases)} examples | judge={JUDGE_MODEL}"
    )

    result = evaluate(
        test_cases=test_cases,
        metrics=[metric],
        identifier=f"EXP-03-{MODEL_TYPE}",
        async_config=AsyncConfig(max_concurrent=5),
        display_config=DisplayConfig(results_folder="./results"),
        cache_config=CacheConfig(write_cache=False)
    )

    metric_results = [
        metric
        for test_result in result.test_results
        for metric in test_result.metrics_data
    ]

    scores = [m.score for m in metric_results if m.score is not None]
    passed = [m.success for m in metric_results if m.success is not None]

    average_score = sum(scores) / len(scores) if scores else 0.0
    pass_rate = 100 * sum(passed) / len(passed) if passed else 0.0

    current_results = {}
    if OUTPUT_FILE.exists():
        try:
            current_results = json.loads(
                OUTPUT_FILE.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            print("WARNING: Existing EX03 file is invalid JSON. Starting a new file.")

    current_results.setdefault("experiment", "EXP-03")
    current_results.setdefault("judge", "DeepEval GEval")
    current_results.setdefault("judge_model", JUDGE_MODEL)
    current_results.setdefault("threshold", THRESHOLD)

    current_results[f"{MODEL_TYPE}_model"] = {
        "n_examples": len(test_cases),
        "average_score": round(average_score, 4),
        "pass_rate_pct": round(pass_rate, 2),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(current_results, indent=2),
        encoding="utf-8",
    )

    print("EXP-03 aggregate results:")
    print(f"Model:         {MODEL_TYPE}")
    print(f"Average Score: {average_score:.4f}")
    print(f"Pass Rate:     {pass_rate:.2f}%")
    print(f"Saved to:      {OUTPUT_FILE}")

    run = wandb.init(
        project="ade-extraction-qlora",
        name=f"eval-qwen25-3b-ade-lora-exp3-{MODEL_TYPE}",
        job_type="evaluation",
        config={
            "experiment": "EXP-03",
            "judge": "DeepEval GEval",
            "judge_model": JUDGE_MODEL,
            "model_type": MODEL_TYPE,
            "n_examples": len(test_cases),
            "threshold": THRESHOLD,
        },
    )

    run.summary["average_score"] = round(average_score, 4)
    run.summary["pass_rate_pct"] = round(pass_rate, 2)
    run.summary["n_examples"] = len(test_cases)
    run.finish()


if __name__ == "__main__":
    main()
