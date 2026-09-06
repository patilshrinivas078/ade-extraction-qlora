"""
EXP-02: Relaxed evaluation of base model vs LoRA-fine-tuned model on the
held-out ADE extraction test set.

Purpose
-------
EXP-01 (evaluate.py) scores drug/effect pairs with exact string equality
after light normalization. Manual error analysis of EXP-01's "partial_match"
cases showed that a meaningful fraction of them are not genuine extraction
failures -- they're surface-form/span-boundary differences from the gold
annotation (e.g. gold says "severe cholestasis", the model correctly
identified the same adverse event but said "cholestasis"; gold says
"interferons", the model said "interferon").

EXP-02 does NOT replace EXP-01. It keeps every EXP-01 strict metric
unchanged and computed identically, and adds a SEPARATE, clearly-labeled
relaxed scoring pass on top of the exact same model, predictions, and
frozen 500-example test set. The purpose is narrow and diagnostic: quantify
how much of the EXP-01 error rate is span/surface-form disagreement versus
genuine extraction failure. It is not a claim that the relaxed number is
"the real" model performance -- strict remains the primary, most defensible
metric; relaxed is context for interpreting it.

Relaxations used
----------------
1. Normalization (applied to BOTH strict and relaxed comparisons, same as
   EXP-01's evaluate.py):
   - Case-insensitive comparison.
   - Collapsed whitespace.

2. Conservative containment matching (relaxed pass only):
   - Two normalized strings match if they're equal, OR if the shorter one
     appears in the longer one as a WHOLE WORD (regex word-boundary match,
     not a raw substring check) AND the shorter string is at least 4
     characters long.
   - The word-boundary + minimum-length requirements exist specifically to
     block false matches like "ph" inside "nephropathy" or "MS" inside
     "chronic MS-related fatigue" -- a naive `a in b` substring check (an
     earlier draft of this script) allows both of those, which is not a
     "conservative" relaxation, it's a scoring bug.
   - This does NOT catch negation ("pain" vs "no significant pain" still
     matches) -- that's a known, accepted limitation. Fixing it would
     require actual NLP (negation detection), which is explicitly out of
     scope for this conservative, inspectable matcher.
   - Applied independently to drug and effect strings; a pair-level relaxed
     match requires BOTH fields to independently pass.

3. One-to-one matching, made deterministic:
   - A predicted entity/pair can match at most one gold entity/pair, and
     vice versa -- this prevents one broad prediction from claiming credit
     for multiple gold items.
   - Matching iterates over predicted/gold items in SORTED order, not raw
     set order. Python randomizes string hashing per process by default, so
     iterating over a set of strings in insertion/hash order is NOT
     guaranteed reproducible across separate runs -- an earlier draft of
     this script did exactly that, meaning re-running the identical
     evaluation on the identical model/predictions could silently report a
     slightly different relaxed score depending on which gold candidate a
     multi-candidate prediction happened to greedily claim first. Sorting
     first removes that non-determinism.
   - When a prediction has multiple valid gold candidates, an exact match
     (after normalization) is preferred over a containment-only match, if
     one is available among the remaining unmatched gold items. This is a
     more principled greedy choice than "whichever candidate is encountered
     first," not just a determinism fix.

NOT relaxed
-----------
- No LLM/semantic judging.
- No synonym dictionary.
- No medical ontology.
- No automatic split/merge inference (a gold pair is never combined with or
  split from another to make matching easier).
- No new drug-ADE relations invented.
- No modification of the gold test set.
- Exact/strict metrics (EXP-01's definitions) remain the primary benchmark
  and are reported alongside the relaxed ones, not replaced by them.

Outputs
-------
    - Strict pair/drug/effect precision, recall, F1 (identical definition
      and normalization to EXP-01)
    - Relaxed pair/drug/effect precision, recall, F1
    - Strict JSON validity and exact match
    - Relaxed exact match
    - Per-example strict-vs-relaxed status (did relaxing the match criteria
      change this example's outcome?)
    - EXP-01's error-category breakdown (correct / missed_all / hallucinated
      / mispaired_entities / partial_match / invalid_json), computed on the
      strict result so it's directly comparable to EXP-01's numbers
    - Full predictions logged to Weights & Biases as browsable tables
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
    """Same light normalization used by EXP-01: lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower())


def parse_model_json(raw_output: str):
    """Parses the expected JSON schema and returns a set of (drug, effect)
    pairs, or None if the output doesn't parse as valid JSON matching the
    schema at all."""
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


def generate_batch(model, tokenizer, sentences: list, max_new_tokens: int = 128) -> list:
    """Generates for a whole batch of sentences in one forward-pass sequence.
    Requires left-padding: with left-padding every sequence in the batch ends
    at the same position, so the generated continuation always starts at the
    same index for every example, which is what makes the post-hoc slicing
    below correct."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": s}],
            tokenize=False,
            add_generation_prompt=True,
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


def relaxed_text_match(a: str, b: str) -> bool:
    """Conservative relaxed matching for a single drug or effect string.

    Exact equality is always a match. Otherwise, the shorter normalized string 
    must appear in the longer one as a WHOLE WORD. A plain substring check would also match
    "ph" inside "nephropathy" or "MS" inside "chronic MS-related fatigue",
    which is not a conservative relaxation, it's a false-positive risk. The
    4-character minimum on the shorter string is an extra guard against
    short strings/abbreviations matching too permissively even with word
    boundaries enforced.

    This intentionally does NOT catch negation (e.g. "pain" still matches
    inside "no significant pain")"""
    a, b = normalize(a), normalize(b)
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 4:
        return False
    return re.search(r"\b" + re.escape(shorter) + r"\b", longer) is not None


def relaxed_pair_match(pred_pair: tuple, gold_pair: tuple) -> bool:
    """Relaxed match for a complete (drug, effect) pair -- BOTH fields must
    independently pass relaxed_text_match. A pair is never credited just
    because the combined/concatenated text looks similar."""
    pred_drug, pred_effect = pred_pair
    gold_drug, gold_effect = gold_pair
    return relaxed_text_match(pred_drug, gold_drug) and relaxed_text_match(pred_effect, gold_effect)

def exact_normalized_match(a, b) -> bool:
    """Exact equality after the same string normalization used by EXP-01.
    Handles both individual strings and (drug, effect) tuples."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return (
            normalize(a[0]) == normalize(b[0])
            and normalize(a[1]) == normalize(b[1])
        )

    return normalize(a) == normalize(b)

def relaxed_set_counts(predicted_items, gold_items, item_match_fn) -> tuple:
    """Computes TP/FP/FN using one-to-one relaxed matching: each prediction
    can match at most one gold item and each gold item can be matched by at
    most one prediction.

    Iterates over SORTED predicted/gold items, not raw set order. Python
    randomizes string hashing per process by default, so a plain `for x in
    some_set` loop does not have a reproducible order across separate runs --
    when a prediction has multiple valid gold candidates, which one it
    greedily claims first could silently differ between two runs of the
    exact same evaluation, changing the reported TP count. Sorting first
    removes that non-determinism.

    When a prediction has multiple valid candidates among the remaining
    unmatched gold items, an EXACT match (post-normalization) is preferred
    over a containment-only match, if one is available -- a more principled
    greedy choice than "whichever candidate the loop reaches first."""
    unmatched_gold = sorted(gold_items)
    tp = 0
    for pred in sorted(predicted_items):
        exact = next((g for g in unmatched_gold if exact_normalized_match(pred, g)), None)
        match = exact if exact is not None else next((g for g in unmatched_gold if item_match_fn(pred, g)), None)
        if match is not None:
            tp += 1
            unmatched_gold.remove(match)
    fp = len(predicted_items) - tp
    fn = len(gold_items) - tp
    return tp, fp, fn


def categorize_error(gold_pairs: set, pred_pairs: set, is_valid_json: bool) -> str:
    """Same strict error categorization used by EXP-01, computed on the
    STRICT (not relaxed) pair sets so it's directly comparable to EXP-01's
    reported category counts."""
    if not is_valid_json:
        return "invalid_json"
    if pred_pairs == gold_pairs:
        return "correct"
    if not gold_pairs and pred_pairs:
        return "hallucinated"
    if gold_pairs and not pred_pairs:
        return "missed_all"

    pred_drugs = {d for d, _ in pred_pairs}
    gold_drugs = {d for d, _ in gold_pairs}
    pred_effects = {e for _, e in pred_pairs}
    gold_effects = {e for _, e in gold_pairs}

    if pred_drugs == gold_drugs and pred_effects == gold_effects:
        return "mispaired_entities"
    return "partial_match"


def evaluate_model(model, tokenizer, test_examples: list, batch_size: int = 16) -> dict:
    n_valid_json = 0
    n_exact_match = 0
    n_relaxed_exact_match = 0

    # Strict counters (identical definition to EXP-01)
    pair_tp = pair_fp = pair_fn = 0
    drug_tp = drug_fp = drug_fn = 0
    effect_tp = effect_fp = effect_fn = 0

    # Relaxed counters
    relaxed_pair_tp = relaxed_pair_fp = relaxed_pair_fn = 0
    relaxed_drug_tp = relaxed_drug_fp = relaxed_drug_fn = 0
    relaxed_effect_tp = relaxed_effect_fp = relaxed_effect_fn = 0

    error_categories = {
        "correct": 0, "missed_all": 0, "hallucinated": 0,
        "mispaired_entities": 0, "partial_match": 0, "invalid_json": 0,
    }
    relaxed_status_counts = {
        "strict_correct_relaxed_correct": 0,
        "strict_error_relaxed_correct": 0,
        "strict_error_relaxed_error": 0,
    }

    predictions = []
    batches = [test_examples[i : i + batch_size] for i in range(0, len(test_examples), batch_size)]

    for batch in tqdm(batches, desc=f"Evaluating relaxed (batch_size={batch_size})"):
        sentences = [ex["input_sentence"] for ex in batch]
        raw_outputs = generate_batch(model, tokenizer, sentences)

        for ex, raw_output in zip(batch, raw_outputs):
            pred_pairs = parse_model_json(raw_output)
            is_valid = pred_pairs is not None
            n_valid_json += int(is_valid)
            if pred_pairs is None:
                pred_pairs = set()

            gold_pairs = {(normalize(e["drug"]), normalize(e["effect"])) for e in ex["gold_adverse_events"]}

            # Strict pair-level
            pair_tp += len(pred_pairs & gold_pairs)
            pair_fp += len(pred_pairs - gold_pairs)
            pair_fn += len(gold_pairs - pred_pairs)

            # Strict drug/effect levels
            pred_drugs = {d for d, _ in pred_pairs}
            gold_drugs = {d for d, _ in gold_pairs}
            pred_effects = {e for _, e in pred_pairs}
            gold_effects = {e for _, e in gold_pairs}

            drug_tp += len(pred_drugs & gold_drugs)
            drug_fp += len(pred_drugs - gold_drugs)
            drug_fn += len(gold_drugs - pred_drugs)

            effect_tp += len(pred_effects & gold_effects)
            effect_fp += len(pred_effects - gold_effects)
            effect_fn += len(gold_effects - pred_effects)

            strict_exact = pred_pairs == gold_pairs
            if strict_exact:
                n_exact_match += 1

            # Relaxed pair-level
            r_pair_tp, r_pair_fp, r_pair_fn = relaxed_set_counts(pred_pairs, gold_pairs, relaxed_pair_match)
            relaxed_pair_tp += r_pair_tp
            relaxed_pair_fp += r_pair_fp
            relaxed_pair_fn += r_pair_fn

            # Relaxed drug/effect levels
            r_drug_tp, r_drug_fp, r_drug_fn = relaxed_set_counts(pred_drugs, gold_drugs, relaxed_text_match)
            relaxed_drug_tp += r_drug_tp
            relaxed_drug_fp += r_drug_fp
            relaxed_drug_fn += r_drug_fn

            r_effect_tp, r_effect_fp, r_effect_fn = relaxed_set_counts(pred_effects, gold_effects, relaxed_text_match)
            relaxed_effect_tp += r_effect_tp
            relaxed_effect_fp += r_effect_fp
            relaxed_effect_fn += r_effect_fn

            relaxed_exact = r_pair_fp == 0 and r_pair_fn == 0
            if relaxed_exact:
                n_relaxed_exact_match += 1

            if strict_exact and relaxed_exact:
                relaxed_status_counts["strict_correct_relaxed_correct"] += 1
            elif (not strict_exact) and relaxed_exact:
                relaxed_status_counts["strict_error_relaxed_correct"] += 1
            else:
                relaxed_status_counts["strict_error_relaxed_error"] += 1

            category = categorize_error(gold_pairs, pred_pairs, is_valid)
            error_categories[category] += 1

            predictions.append({
                "input_sentence": ex["input_sentence"],
                "gold": sorted(list(gold_pairs)),
                "predicted": sorted(list(pred_pairs)),
                "raw_output": raw_output,
                "valid_json": is_valid,
                "exact_match": strict_exact,
                "relaxed_pair_match": relaxed_exact,
                "relaxed_only_match": (not strict_exact) and relaxed_exact,
                "error_category": category,
            })

    n = len(test_examples)
    pair_p, pair_r, pair_f1 = prf(pair_tp, pair_fp, pair_fn)
    drug_p, drug_r, drug_f1 = prf(drug_tp, drug_fp, drug_fn)
    effect_p, effect_r, effect_f1 = prf(effect_tp, effect_fp, effect_fn)

    r_pair_p, r_pair_r, r_pair_f1 = prf(relaxed_pair_tp, relaxed_pair_fp, relaxed_pair_fn)
    r_drug_p, r_drug_r, r_drug_f1 = prf(relaxed_drug_tp, relaxed_drug_fp, relaxed_drug_fn)
    r_effect_p, r_effect_r, r_effect_f1 = prf(relaxed_effect_tp, relaxed_effect_fp, relaxed_effect_fn)

    return {
        "n_examples": n,
        "json_validity_pct": round(100 * n_valid_json / n, 2),
        "exact_match_pct": round(100 * n_exact_match / n, 2),
        "relaxed_exact_match_pct": round(100 * n_relaxed_exact_match / n, 2),
        "precision_pct": round(100 * pair_p, 2),
        "recall_pct": round(100 * pair_r, 2),
        "f1_pct": round(100 * pair_f1, 2),
        "pair_level": {"precision_pct": round(100 * pair_p, 2), "recall_pct": round(100 * pair_r, 2), "f1_pct": round(100 * pair_f1, 2)},
        "drug_level": {"precision_pct": round(100 * drug_p, 2), "recall_pct": round(100 * drug_r, 2), "f1_pct": round(100 * drug_f1, 2)},
        "effect_level": {"precision_pct": round(100 * effect_p, 2), "recall_pct": round(100 * effect_r, 2), "f1_pct": round(100 * effect_f1, 2)},
        "relaxed_pair_level": {"precision_pct": round(100 * r_pair_p, 2), "recall_pct": round(100 * r_pair_r, 2), "f1_pct": round(100 * r_pair_f1, 2)},
        "relaxed_drug_level": {"precision_pct": round(100 * r_drug_p, 2), "recall_pct": round(100 * r_drug_r, 2), "f1_pct": round(100 * r_drug_f1, 2)},
        "relaxed_effect_level": {"precision_pct": round(100 * r_effect_p, 2), "recall_pct": round(100 * r_effect_r, 2), "f1_pct": round(100 * r_effect_f1, 2)},
        "error_categories": error_categories,
        "relaxed_status_counts": relaxed_status_counts,
        "predictions": predictions,
    }


def log_to_wandb(base_results: dict, ft_results: dict, run_name: str, project: str):
    import wandb

    run = wandb.init(project=project, name=run_name, job_type="evaluation")

    metric_levels = ["pair_level", "drug_level", "effect_level", "relaxed_pair_level", "relaxed_drug_level", "relaxed_effect_level"]

    for tag, results in [("base", base_results), ("fine_tuned", ft_results)]:
        run.summary[f"{tag}/json_validity_pct"] = results["json_validity_pct"]
        run.summary[f"{tag}/exact_match_pct"] = results["exact_match_pct"]
        run.summary[f"{tag}/relaxed_exact_match_pct"] = results["relaxed_exact_match_pct"]
        for level in metric_levels:
            for metric, value in results[level].items():
                run.summary[f"{tag}/{level}/{metric}"] = value
        for category, count in results["error_categories"].items():
            run.summary[f"{tag}/error_categories/{category}"] = count
        for category, count in results["relaxed_status_counts"].items():
            run.summary[f"{tag}/relaxed_status/{category}"] = count

    comparison_rows = []
    for level in metric_levels:
        for metric in ["precision_pct", "recall_pct", "f1_pct"]:
            comparison_rows.append([f"{level}/{metric}", base_results[level][metric], ft_results[level][metric]])
    comparison_rows.extend([
        ["json_validity_pct", base_results["json_validity_pct"], ft_results["json_validity_pct"]],
        ["exact_match_pct", base_results["exact_match_pct"], ft_results["exact_match_pct"]],
        ["relaxed_exact_match_pct", base_results["relaxed_exact_match_pct"], ft_results["relaxed_exact_match_pct"]],
    ])
    run.log({"comparison_table": wandb.Table(columns=["metric", "base", "fine_tuned"], data=comparison_rows)})

    pred_columns = ["input_sentence", "gold", "predicted", "raw_output", "valid_json", "exact_match", "relaxed_pair_match", "relaxed_only_match", "error_category"]
    for tag, results in [("base", base_results), ("fine_tuned", ft_results)]:
        rows = [[p[c] if not isinstance(p[c], list) else json.dumps(p[c]) for c in pred_columns] for p in results["predictions"]]
        run.log({f"{tag}_predictions": wandb.Table(columns=pred_columns, data=rows)})

    run.finish()
    print(f"Logged relaxed evaluation results to W&B project '{project}', run '{run_name}'")


def print_results(base_results: dict, ft_results: dict):
    print("\n" + "=" * 82)
    print(f"{'Metric':<38}{'Base Model':>20}{'Fine-tuned':>20}")
    print("=" * 82)

    strict_levels = [("Pair-level", "pair_level"), ("Drug-level", "drug_level"), ("Effect-level", "effect_level")]
    relaxed_levels = [("Relaxed Pair-level", "relaxed_pair_level"), ("Relaxed Drug-level", "relaxed_drug_level"), ("Relaxed Effect-level", "relaxed_effect_level")]

    for level_label, level in strict_levels:
        for metric, label in [("precision_pct", "Precision"), ("recall_pct", "Recall"), ("f1_pct", "F1")]:
            print(f"{level_label} {label:<22}{base_results[level][metric]:>20.2f}{ft_results[level][metric]:>20.2f}")

    print("-" * 82)

    for level_label, level in relaxed_levels:
        for metric, label in [("precision_pct", "Precision"), ("recall_pct", "Recall"), ("f1_pct", "F1")]:
            print(f"{level_label} {label:<18}{base_results[level][metric]:>20.2f}{ft_results[level][metric]:>20.2f}")

    print("-" * 82)
    print(f"{'JSON Validity (%)':<38}{base_results['json_validity_pct']:>20.2f}{ft_results['json_validity_pct']:>20.2f}")
    print(f"{'Exact Match (%)':<38}{base_results['exact_match_pct']:>20.2f}{ft_results['exact_match_pct']:>20.2f}")
    print(f"{'Relaxed Exact Match (%)':<38}{base_results['relaxed_exact_match_pct']:>20.2f}{ft_results['relaxed_exact_match_pct']:>20.2f}")
    print("=" * 82)

    print("\nFine-tuned model error categories (strict):")
    for cat, count in ft_results["error_categories"].items():
        print(f"  {cat:<24}{count:>5} / {ft_results['n_examples']}")

    print("\nFine-tuned model relaxed status:")
    for cat, count in ft_results["relaxed_status_counts"].items():
        print(f"  {cat:<38}{count:>5} / {ft_results['n_examples']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_dir", default="outputs/qwen25-3b-ade-lora")
    parser.add_argument("--test_file", default="data/test.jsonl")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--wandb_project", default="ade-extraction-qlora")
    parser.add_argument("--run_name", default="eval-qwen25-3b-ade-lora-exp2")
    parser.add_argument("--no_wandb", action="store_true", help="Skip W&B logging, just write the local JSON")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size -- raise if you have VRAM headroom, lower if you hit OOM")
    args = parser.parse_args()

    with open(args.test_file, encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]
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
    tokenizer.padding_side = "left"

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id, quantization_config=bnb_config, device_map={"": 0})
    base_model.eval()
    base_results = evaluate_model(base_model, tokenizer, test_examples, batch_size=args.batch_size)
    del base_model
    torch.cuda.empty_cache()

    print("Loading fine-tuned model (base + adapter)...")
    ft_model = AutoModelForCausalLM.from_pretrained(args.base_model_id, quantization_config=bnb_config, device_map={"": 0})
    ft_model = PeftModel.from_pretrained(ft_model, args.adapter_dir)
    ft_model.eval()
    ft_results = evaluate_model(ft_model, tokenizer, test_examples, batch_size=args.batch_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "experiment": "EXP-02",
        "base_model": base_results,
        "fine_tuned_model": ft_results,
        "relaxed_matching": {
            "method": "case-insensitive, whitespace-normalized, word-boundary-aware containment matching",
            "one_to_one_matching": True,
            "deterministic": True,
            "automatic_split_merge": False,
            "llm_judge": False,
            "test_set_modified": False,
        },
    }
    out_path = out_dir / "EX02 comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print_results(base_results, ft_results)
    print(f"\nFull predictions + metrics written to {out_path}")

    if not args.no_wandb:
        log_to_wandb(base_results, ft_results, args.run_name, args.wandb_project)


if __name__ == "__main__":
    main()