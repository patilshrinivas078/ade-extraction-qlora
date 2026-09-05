"""
Evaluates base model vs LoRA-fine-tuned model on the held-out ADE
extraction test set.

Metrics:
  - JSON validity rate: does the output even parse as the expected JSON
    schema? (a model that isn't fine-tuned on this exact output format
    often wraps JSON in markdown fences, adds commentary, etc. — this
    metric alone can show a large fine-tuning gain even before you get
    to extraction correctness)
  - Set-based Precision / Recall / F1 over (drug, effect) pairs, using
    normalized string comparison. Set-based (not exact list-order match)
    because two extractions with the same pairs in different order are
    equally correct.
  - Exact match: the full extracted set is identical to gold (strict).

Numbers are computed only from the actual run — nothing here is fabricated
or hardcoded.
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
    # Take the first {...} block in case there's trailing commentary
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


def generate(model, tokenizer, sentence: str, max_new_tokens=200) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentence},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate_model(model, tokenizer, test_examples: list) -> dict:
    n_valid_json = 0
    n_exact_match = 0
    total_tp, total_fp, total_fn = 0, 0, 0
    predictions = []

    for ex in tqdm(test_examples, desc="Evaluating"):
        raw_output = generate(model, tokenizer, ex["input_sentence"])
        pred_pairs = parse_model_json(raw_output)

        gold_pairs = {
            (normalize(e["drug"]), normalize(e["effect"])) for e in ex["gold_adverse_events"]
        }

        is_valid = pred_pairs is not None
        n_valid_json += int(is_valid)

        if pred_pairs is None:
            pred_pairs = set()  # treat unparseable output as "extracted nothing" for scoring purposes

        tp = len(pred_pairs & gold_pairs)
        fp = len(pred_pairs - gold_pairs)
        fn = len(gold_pairs - pred_pairs)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if pred_pairs == gold_pairs:
            n_exact_match += 1

        predictions.append({
            "input_sentence": ex["input_sentence"],
            "gold": sorted(list(gold_pairs)),
            "predicted": sorted(list(pred_pairs)),
            "raw_output": raw_output,
            "valid_json": is_valid,
            "exact_match": pred_pairs == gold_pairs,
        })

    n = len(test_examples)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_examples": n,
        "json_validity_pct": round(100 * n_valid_json / n, 2),
        "exact_match_pct": round(100 * n_exact_match / n, 2),
        "precision_pct": round(100 * precision, 2),
        "recall_pct": round(100 * recall, 2),
        "f1_pct": round(100 * f1, 2),
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_dir", default="outputs/qwen25-3b-ade-lora")
    parser.add_argument("--test_file", default="data/test.jsonl")
    parser.add_argument("--out_dir", default="results")
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

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, device_map="auto"
    )
    base_model.eval()
    base_results = evaluate_model(base_model, tokenizer, test_examples)
    del base_model
    torch.cuda.empty_cache()

    print("Loading fine-tuned model (base + adapter)...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(ft_model, args.adapter_dir)
    ft_model.eval()
    ft_results = evaluate_model(ft_model, tokenizer, test_examples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = {"base_model": base_results, "fine_tuned_model": ft_results}
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'Metric':<24}{'Base Model':>16}{'Fine-tuned':>16}")
    print("=" * 60)
    for metric, label in [
        ("json_validity_pct", "JSON Validity (%)"),
        ("exact_match_pct", "Exact Match (%)"),
        ("precision_pct", "Precision (%)"),
        ("recall_pct", "Recall (%)"),
        ("f1_pct", "F1 (%)"),
    ]:
        print(f"{label:<24}{base_results[metric]:>16}{ft_results[metric]:>16}")
    print("=" * 60)
    print(f"\nFull predictions + metrics written to {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
