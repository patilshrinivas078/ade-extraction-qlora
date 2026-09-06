"""
Evaluates base model vs LoRA-fine-tuned model on the held-out ADE
extraction test set.

Metrics, computed at THREE levels of strictness (not just one blended
number), because a single pair-level F1 can't tell you WHERE the model is
failing:
  - Pair-level: exact (drug, effect) pair match -- the strictest, "did you
    get the full relation right" metric.
  - Drug-level: ignoring which effect it was paired with, did the model
    find the right set of drug names in the sentence?
  - Effect-level: same idea, for adverse effect phrases.
  A model that's good at drug-level and effect-level but weak at pair-level
  is finding the right entities but mis-pairing them -- a genuinely
  different failure mode from missing entities entirely, and one that's
  invisible if you only look at pair-level F1.

Also reports:
  - JSON validity rate (does output even parse as the expected schema)
  - Exact match (the full extracted set identical to gold)
  - A per-example error-category breakdown (missed / hallucinated /
    mis-paired / correct) to guide error analysis, not just a leaderboard number.

All numbers come only from the actual run. Predictions are also logged to
Weights & Biases as a browsable table (if enabled) so error analysis can
happen in the W&B UI rather than only in a local JSON file.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "You are a clinical information extraction system. Given a sentence "
    "from a medical case report, extract every drug and the adverse effect "
    "it caused, if any. Respond with ONLY a JSON object of the exact form:\n"
    '{"adverse_events": [{"drug": "<drug name>", "effect": "<adverse effect>"}]}\n'
    "If the sentence describes no adverse drug event, respond with "
    '{"adverse_events": []}. Do not include any text other than the JSON object.'
)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def parse_model_json(raw_output: str):
    """Attempts to parse the model's output as the expected JSON schema.
    Strips common wrapper artifacts (markdown code fences) a non-fine-tuned
    model is prone to adding. Returns None if parsing fails entirely."""
    text = raw_output.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict) or "adverse_events" not in parsed:
            return None
        events = parsed["adverse_events"]
        if not isinstance(events, list):
            return None
        pairs = set()
        for e in events:
            if isinstance(e, dict) and "drug" in e and "effect" in e:
                pairs.add((normalize(str(e["drug"])), normalize(str(e["effect"]))))
        return pairs
    except (json.JSONDecodeError, TypeError):
        return None


def generate_batch(model, tokenizer, sentences: list, max_new_tokens=128) -> list:
    """Generates for a whole batch of sentences in one forward-pass sequence,
    instead of one generate() call per example. Left-padding is required here:
    with left-padding, every sequence in the batch ends at the same position,
    so the generated continuation always starts at the same index
    (inputs["input_ids"].shape[1]) for every example -- with right-padding,
    that start index would differ per example and slicing out just the
    generated portion would be far messier."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": s}],
            tokenize=False, add_generation_prompt=True,
        )
        for s in sentences
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[:, input_len:]
    return [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in generated]


def prf(tp: int, fp: int, fn: int) -> tuple:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def categorize_error(gold_pairs: set, pred_pairs: set, is_valid_json: bool) -> str:
    """Buckets one example's result into a diagnostic category. This is the
    thing to look at when deciding what to fix next -- e.g. lots of
    'mispaired_entities' means the model finds the right drugs/effects but
    doesn't associate them correctly, which calls for different fixes
    (e.g. more multi-relation training examples) than 'missed_all' would
    (which points more at recall / negative-example calibration)."""
    if not is_valid_json:
        return "invalid_json"
    if pred_pairs == gold_pairs:
        return "correct"
    if not gold_pairs and pred_pairs:
        return "hallucinated"  # model found an ADE in a sentence that has none
    if gold_pairs and not pred_pairs:
        return "missed_all"  # model found nothing where gold has >=1 ADE

    pred_drugs = {d for d, _ in pred_pairs}
    gold_drugs = {d for d, _ in gold_pairs}
    pred_effects = {e for _, e in pred_pairs}
    gold_effects = {e for _, e in gold_pairs}

    if pred_drugs == gold_drugs and pred_effects == gold_effects:
        return "mispaired_entities"  # right entities, wrong drug<->effect pairing
    return "partial_match"  # some overlap but entities themselves are also off


def evaluate_model(model, tokenizer, test_examples: list, batch_size: int = 16) -> dict:
    n_valid_json = 0
    n_exact_match = 0
    # Separate TP/FP/FN counters at each level of strictness
    pair_tp = pair_fp = pair_fn = 0
    drug_tp = drug_fp = drug_fn = 0
    effect_tp = effect_fp = effect_fn = 0
    error_categories = {"correct": 0, "missed_all": 0, "hallucinated": 0,
                         "mispaired_entities": 0, "partial_match": 0, "invalid_json": 0}
    predictions = []

    batches = [test_examples[i : i + batch_size] for i in range(0, len(test_examples), batch_size)]
    for batch in tqdm(batches, desc=f"Evaluating (batch_size={batch_size})"):
        sentences = [ex["input_sentence"] for ex in batch]
        raw_outputs = generate_batch(model, tokenizer, sentences)

        for ex, raw_output in zip(batch, raw_outputs):
            pred_pairs = parse_model_json(raw_output)
            is_valid = pred_pairs is not None
            n_valid_json += int(is_valid)
            if pred_pairs is None:
                pred_pairs = set()

            gold_pairs = {
                (normalize(e["drug"]), normalize(e["effect"])) for e in ex["gold_adverse_events"]
            }

            pred_drugs = {d for d, _ in pred_pairs}
            gold_drugs = {d for d, _ in gold_pairs}
            pred_effects = {e for _, e in pred_pairs}
            gold_effects = {e for _, e in gold_pairs}

            pair_tp += len(pred_pairs & gold_pairs)
            pair_fp += len(pred_pairs - gold_pairs)
            pair_fn += len(gold_pairs - pred_pairs)

            drug_tp += len(pred_drugs & gold_drugs)
            drug_fp += len(pred_drugs - gold_drugs)
            drug_fn += len(gold_drugs - pred_drugs)

            effect_tp += len(pred_effects & gold_effects)
            effect_fp += len(pred_effects - gold_effects)
            effect_fn += len(gold_effects - pred_effects)

            if pred_pairs == gold_pairs:
                n_exact_match += 1

            category = categorize_error(gold_pairs, pred_pairs, is_valid)
            error_categories[category] += 1

            predictions.append({
                "input_sentence": ex["input_sentence"],
                "gold": sorted(list(gold_pairs)),
                "predicted": sorted(list(pred_pairs)),
                "raw_output": raw_output,
                "valid_json": is_valid,
                "exact_match": pred_pairs == gold_pairs,
                "error_category": category,
            })

    n = len(test_examples)
    pair_p, pair_r, pair_f1 = prf(pair_tp, pair_fp, pair_fn)
    drug_p, drug_r, drug_f1 = prf(drug_tp, drug_fp, drug_fn)
    effect_p, effect_r, effect_f1 = prf(effect_tp, effect_fp, effect_fn)

    return {
        "n_examples": n,
        "json_validity_pct": round(100 * n_valid_json / n, 2),
        "exact_match_pct": round(100 * n_exact_match / n, 2),
        # kept for backward compatibility with the earlier flat comparison table
        "precision_pct": round(100 * pair_p, 2),
        "recall_pct": round(100 * pair_r, 2),
        "f1_pct": round(100 * pair_f1, 2),
        "pair_level": {"precision_pct": round(100 * pair_p, 2), "recall_pct": round(100 * pair_r, 2), "f1_pct": round(100 * pair_f1, 2)},
        "drug_level": {"precision_pct": round(100 * drug_p, 2), "recall_pct": round(100 * drug_r, 2), "f1_pct": round(100 * drug_f1, 2)},
        "effect_level": {"precision_pct": round(100 * effect_p, 2), "recall_pct": round(100 * effect_r, 2), "f1_pct": round(100 * effect_f1, 2)},
        "error_categories": error_categories,
        "predictions": predictions,
    }


def log_to_wandb(base_results: dict, ft_results: dict, run_name: str, project: str):
    import wandb

    run = wandb.init(project=project, name=run_name, job_type="evaluation")

    for tag, results in [("base", base_results), ("fine_tuned", ft_results)]:
        run.summary[f"{tag}/json_validity_pct"] = results["json_validity_pct"]
        run.summary[f"{tag}/exact_match_pct"] = results["exact_match_pct"]
        for level in ["pair_level", "drug_level", "effect_level"]:
            for metric, value in results[level].items():
                run.summary[f"{tag}/{level}/{metric}"] = value
        for category, count in results["error_categories"].items():
            run.summary[f"{tag}/error_categories/{category}"] = count

    # A compact side-by-side comparison table, mirroring the console printout
    comparison_rows = []
    for level in ["pair_level", "drug_level", "effect_level"]:
        for metric in ["precision_pct", "recall_pct", "f1_pct"]:
            comparison_rows.append([
                f"{level}/{metric}", base_results[level][metric], ft_results[level][metric]
            ])
    comparison_rows.append(["json_validity_pct", base_results["json_validity_pct"], ft_results["json_validity_pct"]])
    comparison_rows.append(["exact_match_pct", base_results["exact_match_pct"], ft_results["exact_match_pct"]])
    run.log({"comparison_table": wandb.Table(columns=["metric", "base", "fine_tuned"], data=comparison_rows)})

    # Full predictions as a browsable table -- this is the actual error-analysis tool:
    # filter/sort by error_category in the W&B UI to look at specific failure modes.
    pred_columns = ["input_sentence", "gold", "predicted", "raw_output", "valid_json", "exact_match", "error_category"]
    for tag, results in [("base", base_results), ("fine_tuned", ft_results)]:
        rows = [[p[c] if not isinstance(p[c], list) else json.dumps(p[c]) for c in pred_columns]
                for p in results["predictions"]]
        run.log({f"{tag}_predictions": wandb.Table(columns=pred_columns, data=rows)})

    run.finish()
    print(f"Logged evaluation results to W&B project '{project}', run '{run_name}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_dir", default="outputs/qwen25-3b-ade-lora")
    parser.add_argument("--test_file", default="data/test.jsonl")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--wandb_project", default="ade-extraction-qlora")
    parser.add_argument("--run_name", default="eval-qwen25-3b-ade-lora-exp1")
    parser.add_argument("--no_wandb", action="store_true", help="Skip W&B logging, just write the local JSON")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size -- raise if you have VRAM headroom, lower if you hit OOM")
    args = parser.parse_args()

    test_examples = [json.loads(l) for l in open(args.test_file)]
    print(f"Loaded {len(test_examples)} test examples")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation -- see generate_batch()'s docstring

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, device_map={"": 0}
    )
    base_model.eval()
    base_results = evaluate_model(base_model, tokenizer, test_examples, batch_size=args.batch_size)
    del base_model
    torch.cuda.empty_cache()

    print("Loading fine-tuned model (base + adapter)...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, device_map={"": 0}
    )
    ft_model = PeftModel.from_pretrained(ft_model, args.adapter_dir)
    ft_model.eval()
    ft_results = evaluate_model(ft_model, tokenizer, test_examples, batch_size=args.batch_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = {"base_model": base_results, "fine_tuned_model": ft_results}
    with open(out_dir / "EX01 comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 66)
    print(f"{'Metric':<30}{'Base Model':>16}{'Fine-tuned':>16}")
    print("=" * 66)
    for level_label, level in [("Pair-level", "pair_level"), ("Drug-level", "drug_level"), ("Effect-level", "effect_level")]:
        for metric, label in [("precision_pct", "Precision"), ("recall_pct", "Recall"), ("f1_pct", "F1")]:
            print(f"{level_label} {label:<18}{base_results[level][metric]:>16}{ft_results[level][metric]:>16}")
    print("-" * 66)
    print(f"{'JSON Validity (%)':<30}{base_results['json_validity_pct']:>16}{ft_results['json_validity_pct']:>16}")
    print(f"{'Exact Match (%)':<30}{base_results['exact_match_pct']:>16}{ft_results['exact_match_pct']:>16}")
    print("=" * 66)
    print("\nError categories (fine-tuned model):")
    for cat, count in ft_results["error_categories"].items():
        print(f"  {cat:<22}{count:>5} / {ft_results['n_examples']}")
    print(f"\nFull predictions + metrics written to {out_dir / 'comparison.json'}")

    if not args.no_wandb:
        log_to_wandb(base_results, ft_results, args.run_name, args.wandb_project)


if __name__ == "__main__":
    main()