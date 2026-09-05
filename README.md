# Clinical Adverse-Drug-Event Extraction — QLoRA Fine-Tuning

Fine-tunes `Qwen/Qwen2.5-3B-Instruct` with QLoRA on **ADE Corpus V2** to
extract drug / adverse-effect pairs from clinical case-report sentences
into a fixed JSON schema, evaluated with field-level precision/recall/F1
against gold-standard extractions.


## Why this project

Extraction quality here is judged by whether the model correctly identifies
the *exact* (drug, effect) pairs present in a sentence: a genuinely
objective signal, scored the same way an information-extraction system
would be evaluated in production (precision/recall/F1 against gold spans).
We deliberately include negative examples (sentences with no adverse
event) so the model has to learn to *not* hallucinate an ADE where none
exists which is a common and important failure mode in real extraction pipelines.

## Task format

**Input:** a clinical sentence.
**Output:**
```json
{"adverse_events": [{"drug": "atorvastatin", "effect": "rhabdomyolysis"}]}
```
or `{"adverse_events": []}` if no adverse event is present.

## Results

*(Filled in after running `src/evaluation/evaluate.py` — not fabricated.
Paste your actual `results/comparison.json` output here.)*

| Metric | Base Model | Fine-tuned |
|---|---|---|
| JSON Validity (%) | — | — |
| Exact Match (%) | — | — |
| Precision (%) | — | — |
| Recall (%) | — | — |
| F1 (%) | — | — |

**JSON Validity is worth watching separately from extraction accuracy** —
a common finding in extraction fine-tuning projects is that the base model
already "understands" the task reasonably well but fails to reliably
follow the exact output format (adds prose, wraps in markdown, uses a
different key name). If most of your improvement shows up in JSON
Validity rather than Precision/Recall, that's a real and worth-reporting
finding, not a weaker result — format reliability is a large part of what
makes extraction usable in a pipeline at all.

## Project structure

```
project/
├── data/                   # train/val/test JSONL (gitignored)
├── notebooks/
├── src/
│   ├── data/                 # dataset loading, negative sampling, formatting
│   ├── training/               # QLoRA SFT training script
│   ├── evaluation/             # base vs fine-tuned comparison harness
│   └── inference/               # CLI for new sentences
├── configs/                   # YAML hyperparameter configs
├── scripts/                   # run_pipeline.sh
├── tests/                     # unit tests for JSON parsing/scoring logic
├── results/                    # comparison.json (small, commit this)
├── outputs/                    # adapter weights (gitignored — push to HF Hub)
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup (Kaggle)

1. New Kaggle Notebook → enable GPU (T4 x2 or P100).
2. `pip install -r requirements.txt`
3. (Optional) set a W&B API key, or set `report_to: "none"` in the config.

## Pipeline

```bash
python src/data/prepare_dataset.py --config configs/qlora_ade.yaml
python src/training/train.py --config configs/qlora_ade.yaml
python src/evaluation/evaluate.py \
    --base_model_id Qwen/Qwen2.5-3B-Instruct \
    --adapter_dir outputs/qwen25-3b-ade-lora \
    --test_file data/test.jsonl
python src/inference/inference.py \
    --sentence "The patient developed severe rhabdomyolysis after being started on high-dose atorvastatin."
```


