"""
Minimal CLI for running the fine-tuned model on a new clinical sentence.

Usage:
    python src/inference/inference.py \
        --sentence "The patient developed severe rhabdomyolysis after being started on high-dose atorvastatin."
"""

import argparse
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are a clinical information extraction system. Given a sentence "
    "from a medical case report, extract every drug and the adverse effect "
    "it caused, if any. Respond with ONLY a JSON object of the exact form:\n"
    '{"adverse_events": [{"drug": "<drug name>", "effect": "<adverse effect>"}]}\n'
    "If the sentence describes no adverse drug event, respond with "
    '{"adverse_events": []}. Do not include any text other than the JSON object.'
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_dir", default="outputs/qwen25-3b-ade-lora")
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.sentence},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print("\nRaw output:\n" + generated)
    match = re.search(r"\{.*\}", generated, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            print("\nParsed JSON:")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print("\n(Could not parse as valid JSON)")


if __name__ == "__main__":
    main()
