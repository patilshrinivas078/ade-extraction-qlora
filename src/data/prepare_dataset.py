"""
Builds a text -> structured JSON extraction dataset from ADE Corpus V2.

ADE Corpus V2 ships as several HF "configs" of the same underlying corpus:
  - Ade_corpus_v2_drug_ade_relation: (text, drug, effect) triples for
    sentences that DO contain a drug/adverse-effect relation. A sentence
    with multiple relations appears as multiple rows with the same text.
  - Ade_corpus_v2_classification: (text, label) where label indicates
    whether the sentence is ADE-related at all — this is where we source
    NEGATIVE examples (sentences with no adverse event to extract).

IMPORTANT: verify these exact config names and field names against the
current dataset card on Hugging Face before running — dataset schemas can
change, and this was written without live access to confirm them.

We combine both configs into a single extraction task:
    input:  a clinical sentence
    output: {"adverse_events": [{"drug": ..., "effect": ...}, ...]}
            (empty list if the sentence has no adverse event)

Training on positives only would teach the model to always "find" an ADE
even when there isn't one — real extraction pipelines need to handle the
negative case correctly, so we deliberately include a controlled fraction
of negative examples (see `negative_ratio` in the config).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a clinical information extraction system. Given a sentence "
    "from a medical case report, extract every drug and the adverse effect "
    "it caused, if any. Respond with ONLY a JSON object of the exact form:\n"
    '{"adverse_events": [{"drug": "<drug name>", "effect": "<adverse effect>"}]}\n'
    "If the sentence describes no adverse drug event, respond with "
    '{"adverse_events": []}. Do not include any text other than the JSON object.'
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_positive_examples(data_cfg: dict) -> dict:
    """Loads the relation config and groups (drug, effect) pairs by sentence text.
    Returns {text: [{"drug": ..., "effect": ...}, ...]}."""
    relation_data = load_dataset(data_cfg["dataset_id"], data_cfg["relation_config"], split="train")
    grouped = defaultdict(list)
    for ex in relation_data:
        text = ex["text"].strip()
        pair = {"drug": ex["drug"].strip(), "effect": ex["effect"].strip()}
        if pair not in grouped[text]:
            grouped[text].append(pair)
    return grouped


def build_negative_examples(data_cfg: dict, positive_texts: set) -> list:
    """Loads the classification config and returns sentence texts labeled
    as NOT containing an adverse drug event, excluding any that overlap
    with the positive set (defensive — different configs of the same
    underlying corpus can occasionally have edge-case overlaps)."""
    cls_data = load_dataset(data_cfg["dataset_id"], data_cfg["classification_config"], split="train")
    # NOTE: verify the actual label convention (0/1 vs string labels) against
    # the current dataset card — written here assuming 0 = "not related".
    negatives = [
        ex["text"].strip()
        for ex in cls_data
        if ex["label"] == 0 and ex["text"].strip() not in positive_texts
    ]
    return negatives


def format_example(text: str, adverse_events: list, tokenizer) -> dict:
    target = json.dumps({"adverse_events": adverse_events}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": target},
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {
        "text": formatted_text,
        "input_sentence": text,
        "gold_adverse_events": adverse_events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora_ade.yaml")
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    print(f"Loading relation config: {data_cfg['relation_config']}")
    positives = build_positive_examples(data_cfg)
    print(f"Unique positive sentences (>=1 adverse event): {len(positives)}")

    print(f"Loading classification config: {data_cfg['classification_config']}")
    negatives = build_negative_examples(data_cfg, set(positives.keys()))
    print(f"Available negative sentences (no adverse event): {len(negatives)}")

    random.seed(cfg["training"]["seed"])
    random.shuffle(negatives)

    n_positive_available = len(positives)
    n_negative_available = len(negatives)
    ratio = data_cfg["negative_ratio"]
    configured_total = data_cfg["train_size"] + data_cfg["val_size"] + data_cfg["test_size"]

    print(f"Available: {n_positive_available} unique positive (ADE-containing) sentences, "
          f"{n_negative_available} negative sentences")
   
    n_positives = n_positive_available

    # ratio = negatives / (positives + negatives)
    # =>  negatives = positives * ratio / (1 - ratio)
    negatives_wanted = int(n_positives * ratio / (1 - ratio)) if ratio < 1 else n_negative_available
    n_negatives = min(negatives_wanted, n_negative_available)

    positive_items = list(positives.items())
    random.shuffle(positive_items)
    negative_items = [(t, []) for t in negatives[:n_negatives]]

    all_items = positive_items + negative_items
    random.shuffle(all_items)

    actual_negative_pct = (len(negative_items) / len(all_items) * 100) if all_items else 0.0
    print(f"Final dataset size: {len(all_items)} "
          f"({len(positive_items)} positive, {len(negative_items)} negative, "
          f"{actual_negative_pct:.1f}% negative, configured ratio was {ratio*100:.1f}%)")

    if len(all_items) != configured_total:
        print(
            f"NOTE: train_size + val_size + test_size in the config sums to {configured_total}, but "
            f"the actual dataset this corpus + ratio produces is {len(all_items)}. This isn't a bug — "
            f"positive supply and negative_ratio determine the real size; train_size/val_size/test_size "
            f"are a target the split below is scaled to match, not a hard cap. Update the config's split "
            f"sizes to these real numbers once you're happy with the ratio, so they stop implicitly relying "
            f"on this auto-scaling."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"])

    # --- token length sanity check before trusting max_seq_length ---
    sample = all_items[: min(500, len(all_items))]
    lengths = [
        len(tokenizer(format_example(t, e, tokenizer)["text"])["input_ids"])
        for t, e in sample
    ]
    lengths.sort()
    p50 = lengths[len(lengths) // 2]
    p99 = lengths[int(len(lengths) * 0.99)]
    print(f"Token length (sample of {len(lengths)}) -> median: {p50}, p99: {p99}")
    if p99 > model_cfg["max_length"]:
        print(f"WARNING: p99 length {p99} exceeds configured max_length "
              f"{model_cfg['max_length']} — consider raising it.")

    # --- split ---
    scale = len(all_items) / configured_total if configured_total > 0 else 0
    n_test = max(1, int(data_cfg["test_size"] * scale))
    n_val = max(1, int(data_cfg["val_size"] * scale))
    n_train = len(all_items) - n_test - n_val

    test_items = all_items[:n_test]
    val_items = all_items[n_test : n_test + n_val]
    train_items = all_items[n_test + n_val : n_test + n_val + n_train]

    if scale < 1.0:
        print(
            f"Scaling splits by {scale:.3f} of configured sizes -> "
            f"train: {data_cfg['train_size']}->{n_train}, "
            f"val: {data_cfg['val_size']}->{n_val}, "
            f"test: {data_cfg['test_size']}->{n_test}"
        )

    print(f"Split sizes -> train: {len(train_items)}, val: {len(val_items)}, test: {len(test_items)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for text, events in items:
                formatted = format_example(text, events, tokenizer)
                f.write(json.dumps(formatted) + "\n")
        print(f"Wrote {len(items)} examples to {out_path}")


if __name__ == "__main__":
    main()
