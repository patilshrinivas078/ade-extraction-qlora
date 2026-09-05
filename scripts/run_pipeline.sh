#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/qlora_ade.yaml"

echo "== 1/3: Preparing dataset =="
python src/data/prepare_dataset.py --config "$CONFIG"

echo "== 2/3: Training =="
python src/training/train.py --config "$CONFIG"

echo "== 3/3: Evaluating base vs fine-tuned =="
python src/evaluation/evaluate.py \
    --base_model_id "Qwen/Qwen2.5-3B-Instruct" \
    --adapter_dir "outputs/qwen25-3b-ade-lora" \
    --test_file "data/test.jsonl"

echo "Done. See results/comparison.json"
