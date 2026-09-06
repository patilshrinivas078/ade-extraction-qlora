# QLoRA Fine-Tuning for Clinical Adverse Drug Event (ADE) Extraction

Fine-tunes `Qwen/Qwen2.5-3B-Instruct` with QLoRA on **ADE Corpus V2** to extract
drug / adverse-effect relations from clinical case-report sentences into a
fixed JSON schema, evaluated across three complementary methodologies - 
strict deterministic matching, span-tolerant relaxed matching, and LLM-based
semantic evaluation.

## The Objective

ADE extraction is an information extraction and relationship-mapping
problem: given an unstructured clinical sentence, the system must identify
the drug involved and the adverse event associated with it, then represent
that relationship in a predictable JSON structure.

```json
{"adverse_events": [{"drug": "aspirin", "effect": "nausea"}]}
```

or `{"adverse_events": []}` if the sentence describes no adverse drug event.

This is different from search or retrieval. Retrieval finds relevant
documents or passages; ADE extraction requires the model to interpret a
sentence, identify entities, and determine which drug–effect relationship is
actually expressed. The project evaluates whether a compact
instruction-tuned LLM can be adapted to a narrowly defined clinical
extraction task through QLoRA fine-tuning.

Negative examples (sentences with no adverse event) are deliberately
included in training so the model learns to *not* hallucinate an ADE where
none exists which is a common and important failure mode that positive-only training data can't teach.

## Trained Model

- **LoRA adapter:** _https://huggingface.co/patilshrinivas/Qwen2.5-3B-Instruct-qlora-drug-ade-relation-extractor_
- **Merged full model:** _https://huggingface.co/patilshrinivas/Qwen2.5-3B-Instruct-drug-ade-relation-extractor_

## Training Configuration

| Detail | EXP-01 |
|---|---:|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Fine-tuning | QLoRA |
| Quantization | 4-bit NF4 + double quantization |
| Compute dtype | BF16 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | Attention + MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Effective batch size | 16 |
| Max sequence length | 512 |
| Train / Validation / Test | 5,570 / 500 / 500 |
| Trainable parameters | 29,933,568 (~0.96% of 3.1B total) |
| GPU | **NVIDIA L4** |

## Evaluation Methodology

Three complementary evaluation passes run on the same frozen 500-example
test set and the same model predictions. None of them replace the others;
each answers a different question about where the model actually stands.

| Experiment | Method | What it answers |
|---|---|---|
| **EXP-01: Strict** | Case-insensitive, whitespace-normalized exact matching on `(drug, effect)` pairs | Does the model recover the *exact* annotated relation? Captures both missed and unsupported relations via Precision/Recall/F1. |
| **EXP-02: Relaxed** | Same predictions and test set, plus conservative whole-word containment matching (e.g. `"severe mucositis"` ↔ `"mucositis"`) with deterministic one-to-one matching | How much of the strict error rate is span-boundary/surface-form disagreement rather than genuine extraction failure? Gold annotations are never modified. |
| **EXP-03: LLM-as-a-Judge** | DeepEval G-Eval, judge model `gpt-4o-mini` | A broader clinical-alignment signal beyond deterministic string matching. The G-Eval correctness score is a semantic evaluation score, not Precision/Recall/F1. Thus, treated as complementary evidence, not a replacement for the deterministic metrics. |

Drug-level and effect-level metrics are also reported independently from the
full pair, since "found the right drug" and "found the right effect" are
different capabilities that can fail separately, collapsing everything
into one pair-level number would hide that distinction.

### EXP-01: Strict Evaluation

| Metric | Base Model | Fine-tuned Model |
|---|---:|---:|
| Pair Precision | 72.31% | **73.27%** |
| Pair Recall | 26.96% | **70.75%** |
| Pair F1 | 39.28% | **71.98%** |
| Drug-level F1 | 57.71% | **92.82%** |
| Effect-level F1 | 42.99% | **76.44%** |
| JSON Validity | 92.00% | **99.00%** |
| Exact Match | 51.20% | **76.20%** |

### EXP-02: Relaxed Evaluation

| Metric | Base Model | Fine-tuned Model |
|---|---:|---:|
| Pair Precision | 88.72% | **86.34%** |
| Pair Recall | 33.08% | **83.37%** |
| Pair F1 | 48.19% | **84.82%** |
| Drug-level F1 | 59.89% | **93.88%** |
| Effect-level F1 | 51.52% | **86.86%** |
| Relaxed Exact Match | 53.80% | **82.40%** |

### EXP-03: LLM-as-a-Judge

| Metric | Base Model | Fine-tuned Model |
|---|---:|---:|
| Judge | GPT-4o-mini | GPT-4o-mini |
| Average G-Eval Correctness Score | 0.5961 | **0.8749** |
| Pass Rate | 54.40% | **82.00%** |

## Empirical Findings

- **Schema reliability.** JSON validity rose from 92.0% to 99.0%, and Exact
  Match from 51.2% to 76.2%. An extraction component needs machine-readable
  output consistently, not just occasionally. This shows fine-tuning taught
  both the extraction behavior and the expected response format.

- **Extraction recall, not just verbosity.** Strict pair Recall rose from
  26.96% to 70.75% while Precision stayed essentially flat (72.31% →
  73.27%). Pair F1 followed, 39.28% → 71.98%. Fine-tuning primarily
  recovered relations the base model was missing, rather than simply
  producing more (and more often wrong) predictions which is the main quantitative
  evidence that QLoRA training materially changed extraction behavior
  rather than just extraction volume.

- **Drugs vs. effects.** Drug-level F1 reached 92.82% while Effect-level F1
  was lower at 76.44%. Manual analysis of the 93 strict `partial_match`
  examples found repeated span/granularity differences, over-extraction of
  related findings, and occasional multi-drug over-attribution i.e sentences
  with several drugs, effects, or related clinical findings remain harder
  than single-relation sentences.

- **Span and representation differences account for real, quantified gap.**
  EXP-02 raised fine-tuned Pair F1 from 71.98% (strict) to 84.82%
  (relaxed), with 31/500 examples flipping from strict-error to
  relaxed-correct without touching the gold annotations. Relaxed is treated
  as a diagnostic view.

- **Remaining error profile (EXP-01, fine-tuned, 500 examples):** 381
  correct, 93 `partial_match`, 12 `missed_all`, 9 hallucinated, 5 invalid
  JSON. Errors concentrate in partial matches rather than wholesale
  failures, combined with strong drug-level performance, this points to
  precise ADE representation and selective over-extraction as the main
  remaining challenge, not a lack of task understanding.

- **Semantic evaluation is directionally consistent.** EXP-03's average
  G-Eval score rose from 0.5961 to 0.8749, pass rate from 54.4% to 82.0% at
  a 0.70 threshold, agreeing with the deterministic results without being
  treated as a definitive replacement for them, since G-Eval is itself an
  LLM-based judgment.

## Findings and Limitations

A relatively small 3B-parameter model can learn a narrowly defined clinical
information-extraction task effectively when the input, output schema, and
supervision are tightly constrained. The strongest evidence is strict Pair
F1 rising from 39.28% to 71.98%, alongside 92.82% drug-level F1, 99.0% JSON
validity, and 84.82% relaxed Pair F1.

These results should be read as a controlled extraction experiment, not
evidence the model can perform medical diagnosis or replace clinical
judgment. Findings are bounded by ADE Corpus V2's annotation conventions,
dataset size, the fixed 500-example test set, and the inherent limitations
of LLM-based evaluation.

## Setup

Trained on a single **NVIDIA L4** GPU. Any single-GPU environment with
sufficient VRAM and CUDA support should work.

```bash
uv sync
```

Set a Weights & Biases API key for experiment tracking, or pass `--no_wandb`
to the evaluation scripts / set `report_to: "none"` in the training config
to skip it.

## Pipeline

```bash
# 1. Prepare data (no GPU needed)
python src/data/prepare_dataset.py --config configs/qlora_ade.yaml

# 2. Train (QLoRA)
python src/training/train.py --config configs/qlora_ade.yaml

# 3. EXP-01: strict evaluation
python src/evaluation/evaluate.py \
    --base_model_id Qwen/Qwen2.5-3B-Instruct \
    --adapter_dir outputs/qwen25-3b-ade-lora \
    --test_file data/test.jsonl

# 4. EXP-02: relaxed (span-tolerant) evaluation, same predictions and test set
python src/evaluation/relaxed_evaluate.py \
    --base_model_id Qwen/Qwen2.5-3B-Instruct \
    --adapter_dir outputs/qwen25-3b-ade-lora \
    --test_file data/test.jsonl

# 5. EXP-03: LLM semantic evaluation (DeepEval G-Eval, gpt-4o-mini)
# adjust the script path below to match your actual EXP-03 filename
python src/evaluation/llm_evaluate.py \
    --base_model_id Qwen/Qwen2.5-3B-Instruct \
    --adapter_dir outputs/qwen25-3b-ade-lora \
    --test_file data/test.jsonl

# 6. Try it interactively
python src/inference/inference.py \
    --sentence "Patients receiving amifostine who develop only fever should be evaluated for an adverse drug reaction, as well as for sepsis and fevers of neutropenia, and it may be necessary to discontinue the drug."
```
