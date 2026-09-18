# KoNeoBench

KoNeoBench is a benchmark for evaluating how well large language models understand Korean neologisms. This repository contains task-level command-line scripts reconstructed from the original experiment notebooks.

The repository currently supports:

| Task | Description | Main metric |
|---|---|---|
| Task 1 | Context-based neologism identification | Accuracy |
| Task 2 | Component reconstruction for blends and abbreviations | Coverage, Precision, F1 |
| Task 3 Type 1 | Odd-one-out classification | Accuracy |
| Task 3 Type 2 | Direct semantic/domain classification | Accuracy |
| Task 4 | One-sentence definition generation | 6-3-1 LLM-judge score |

`semantic` denotes **semantic category(의미범주)**, and `special` denotes **special domain(전문 분야)** throughout the repository.

---

## 1. Repository structure

```text
KoNeoBench/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── models.yaml
│   ├── task1.yaml
│   ├── task2.yaml
│   ├── task3.yaml
│   ├── task4.yaml
│   └── task4_judge.yaml
├── data/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── prompts/
│   ├── task1.yaml
│   ├── task2.yaml
│   ├── task3.yaml
│   ├── task4.yaml
│   └── task4_judge.yaml
├── results/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── scripts/
│   ├── task1.py
│   ├── task2.py
│   ├── task3.py
│   ├── task4.py
│   └── judge_task4.py
└── utils/
    ├── config.py
    ├── definition_generation.py
    ├── definition_judging.py
    ├── io.py
    ├── model_backends.py
    ├── prompting.py
    └── scoring.py
```

Dataset-construction and resampling code is intentionally excluded. Each script consumes a finalized benchmark file.

---

## 2. Installation

Run the following commands from the repository root.

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

On Windows Command Prompt:

```bat
copy .env.example .env
```

Fill only the credentials needed for the selected models:

```dotenv
OPENAI_API_KEY=
UPSTAGE_API_KEY=
HF_TOKEN=
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

Never commit `.env` or hard-code API keys in Python files.

Local Transformers models require a CUDA environment with sufficient GPU memory. Quantization, dtype, checkpoint IDs, and task-specific overrides are defined in `configs/models.yaml` and the corresponding task configuration file.

---

## 3. Supported model aliases

Use the following aliases with `--model`:

```text
gpt-4.1
gpt-5.4
solar-pro3
solar-10.7b-instruct
exaone-3.5-7.8b-instruct
exaone-4.0-32b
qwen2.5-7b-instruct
qwen3.5-9b
llama-3.1-8b-instruct
```

The alias-to-checkpoint and provider mapping is stored in `configs/models.yaml`.

---

## 4. Data preparation

Place the finalized benchmark files in the following locations:

```text
data/
├── task1/
│   ├── Task1_type1.csv
│   └── Task1_type2.csv
├── task2/
│   └── Task2_Source Word Identification.csv
├── task3/
│   ├── Task3_type 1_semantic.csv
│   ├── Task3_type 1_specialized_domain.csv
│   ├── Task3_type 2_semantic.csv
│   └── Task3_type 2_specialized_domain.csv
└── task4/
    ├── Task4_without_ex.csv
    └── Task4_with_ex.csv
```

CSV and JSONL inputs are also supported where the script does not require an Excel worksheet.

### Required columns

The scripts accept multiple aliases for the same field. The complete alias lists are defined in `configs/task*.yaml`.

| Task | Required information |
|---|---|
| Task 1 | usage example, correct headword, confusing distractor, three random distractors |
| Task 2 | neologism, gold component decomposition |
| Task 3 Type 1 | answer position, target term, target label, distractor label, options A–E |
| Task 3 Type 2 semantic | headword, semantic category, optional usage example |
| Task 3 Type 2 special | headword, specialized domain, optional usage example |
| Task 4 | headword, gold definition, collection year, and usage example for `term_example` |

Use `--sheet` when the target data is not in the default worksheet. The value may be a worksheet name or a zero-based index.

```bash
--sheet 1
```

---

## 5. Quick smoke test

Before running the full benchmark, inspect five prepared prompts without calling a model:

```bash
python scripts/task3.py \
  --mode prepare \
  --type type2 \
  --category semantic \
  --context term_only \
  --model gpt-5.4 \
  --input data/task3/task3_type2_semantic.xlsx \
  --limit 5
```

Prepared prompts are written to the corresponding `results/` directory. Use this mode to verify column matching, label mappings, and prompt formatting.

---

# 6. Running Task 1

Task 1 has two finalized input conditions:

- `type1`: particle-removed condition
- `type2`: particle-included condition

The condition name is used for result naming. The actual difference between Type 1 and Type 2 must already be reflected in the input file.

## Task 1 Type 1

```bash
python scripts/task1.py \
  --mode all \
  --model gpt-5.4 \
  --condition type1 \
  --input data/task1/task1_type1.xlsx
```

## Task 1 Type 2

```bash
python scripts/task1.py \
  --mode all \
  --model gpt-5.4 \
  --condition type2 \
  --input data/task1/task1_type2.xlsx
```

`--mode all` runs inference and then evaluates the generated JSONL file.

### Task 1 outputs

```text
results/task1/
├── gpt-5.4_type1.jsonl
├── gpt-5.4_type1.metrics.json
├── gpt-5.4_type1.errors.jsonl
├── gpt-5.4_type2.jsonl
├── gpt-5.4_type2.metrics.json
└── gpt-5.4_type2.errors.jsonl
```

The metrics file includes valid-response accuracy, strict accuracy, prediction coverage, generation failures, year-level accuracy, and distractor error statistics.

Evaluate an existing result without rerunning inference:

```bash
python scripts/task1.py \
  --mode evaluate \
  --model gpt-5.4 \
  --condition type1 \
  --output results/task1/gpt-5.4_type1.jsonl
```

---

# 7. Running Task 2

Task 2 reconstructs the original components used to form a neologism. 

## Evaluation

```bash
python scripts/task2.py \
  --mode all \
  --model gpt-5.4 \
  --input data/task2/task2_data.xlsx \
  --shots 5
```

Rows that overlap with the fixed few-shot demonstrations are automatically excluded from the evaluated benchmark rows.

### Task 2 outputs

```text
results/task2/
├── gpt-5.4_5shot_predictions.jsonl
├── gpt-5.4_5shot_predictions.metrics.json
└── gpt-5.4_5shot_predictions.errors.jsonl
```

The metrics file includes:

- coverage
- precision
- F1

Repeated components are evaluated using multiset matching.

Evaluate an existing result:

```bash
python scripts/task2.py \
  --mode evaluate \
  --model gpt-5.4 \
  --shots 5 \
  --output results/task2/gpt-5.4_5shot_predictions.jsonl
```

---

# 8. Running Task 3

Task 3 uses two command-line axes:

```text
--type type1|type2
--category semantic|special
```

- `semantic`: semantic category(의미범주)
- `special`: special domain(전문분야)

## 8.1 Task 3 Type 1 — semantic

```bash
python scripts/task3.py \
  --mode all \
  --type type1 \
  --category semantic \
  --model gpt-5.4 \
  --input data/task3/task3_type1_semantic.xlsx
```

## 8.2 Task 3 Type 1 — special

```bash
python scripts/task3.py \
  --mode all \
  --type type1 \
  --category special \
  --model gpt-5.4 \
  --input data/task3/task3_type1_special.xlsx
```

## 8.3 Task 3 Type 2 — semantic

Task 3 Type 2 semantic maps the sorted 16-category list to `A`–`P`.

Headword only:

```bash
python scripts/task3.py \
  --mode all \
  --type type2 \
  --category semantic \
  --context term_only \
  --model gpt-5.4 \
  --input data/task3/task3_type2_semantic.xlsx
```

Headword plus usage example:

```bash
python scripts/task3.py \
  --mode all \
  --type type2 \
  --category semantic \
  --context term_example \
  --model gpt-5.4 \
  --input data/task3/task3_type2_semantic.xlsx
```

In `term_example`, `[MASK]` in the example is replaced with the target headword.

## 8.4 Task 3 Type 2 — special

Task 3 Type 2 special maps the sorted specialized-domain list to numeric labels.

Headword only:

```bash
python scripts/task3.py \
  --mode all \
  --type type2 \
  --category special \
  --context term_only \
  --model gpt-5.4 \
  --input data/task3/task3_type2_special.xlsx
```

Headword plus usage example:

```bash
python scripts/task3.py \
  --mode all \
  --type type2 \
  --category special \
  --context term_example \
  --model gpt-5.4 \
  --input data/task3/task3_type2_special.xlsx
```

### Task 3 outputs

Type 1 example:

```text
results/task3/type1_semantic/
├── gpt-5.4_predictions.jsonl
├── gpt-5.4_predictions.metrics.json
└── gpt-5.4_predictions.errors.jsonl
```

Type 2 example:

```text
results/task3/type2_semantic/
├── gpt-5.4_term_only_predictions.jsonl
├── gpt-5.4_term_only_predictions.metrics.json
└── gpt-5.4_term_only_predictions.errors.jsonl
```

Task 3 metrics include valid-response accuracy, strict accuracy, prediction coverage, parsing failures, generation failures, per-label accuracy, per-year accuracy when available, and a sparse confusion table.

---

# 9. Running Task 4

Task 4 generates one-sentence lexicographic definitions under two conditions:

- `term_only`: headword only
- `term_example`: headword plus one usage example

## 9.1 Headword only

```bash
python scripts/task4.py \
  --mode run \
  --condition term_only \
  --model gpt-5.4 \
  --input data/task4/task4_data.xlsx
```

## 9.2 Headword plus usage example

```bash
python scripts/task4.py \
  --mode run \
  --condition term_example \
  --model gpt-5.4 \
  --input data/task4/task4_data.xlsx
```

In `term_example`, `[MASK]` is replaced with the normalized target term before inference.

Inspect the exact prepared requests without loading a model:

```bash
python scripts/task4.py \
  --mode prepare \
  --condition term_example \
  --model gpt-5.4 \
  --input data/task4/task4_data.xlsx \
  --limit 5
```

### Task 4 generation outputs

```text
results/task4/
├── gpt-5.4_term_only_definitions.jsonl
└── gpt-5.4_term_example_definitions.jsonl
```

Each record contains the item ID, term, example condition, gold definition, generated definition, raw model output, status, and error message.

---

# 10. Judging Task 4 outputs

Task 4 generation and evaluation are intentionally separated. `judge_task4.py` reads an existing definition JSONL file and applies the fixed 6-3-1 rubric with Gemini.

## 10.1 Judge `term_only` results

```bash
python scripts/judge_task4.py \
  --mode run \
  --predictions results/task4/gpt-5.4_term_only_definitions.jsonl \
  --gold data/task4/task4_data.xlsx \
  --condition term_only
```

## 10.2 Judge `term_example` results

```bash
python scripts/judge_task4.py \
  --mode run \
  --predictions results/task4/gpt-5.4_term_example_definitions.jsonl \
  --gold data/task4/task4_data.xlsx \
  --condition term_example
```

The judge first attempts to merge gold and prediction rows using an item ID. If a shared ID is unavailable, it falls back to `row_index`.

### 6-3-1 rubric

| Dimension | Range |
|---|---:|
| Semantic adequacy | 0–6 |
| Fluency | 0–3 |
| Factuality | 0–1 |
| Total | 0–10 |

The judge returns the sub-scores. Python calculates all subtotal and total scores.

### Judge outputs

```text
results/task4/judgments/
├── gpt-5.4_term_example_definitions__judge_gemini-2.5-flash.jsonl
├── gpt-5.4_term_example_definitions__judge_gemini-2.5-flash.csv
└── gpt-5.4_term_example_definitions__judge_gemini-2.5-flash.summary.json
```

Recompute only the aggregate summary from an existing judgment JSONL file:

```bash
python scripts/judge_task4.py \
  --mode summarize \
  --input results/task4/judgments/gpt-5.4_term_example_definitions__judge_gemini-2.5-flash.jsonl
```

---

# 11. Viewing results

## Pretty-print a metrics JSON file

```bash
python -m json.tool results/task3/type2_semantic/gpt-5.4_term_only_predictions.metrics.json
```

## View the first five prediction records

Linux or macOS:

```bash
head -n 5 results/task3/type2_semantic/gpt-5.4_term_only_predictions.jsonl
```

PowerShell:

```powershell
Get-Content results/task3/type2_semantic/gpt-5.4_term_only_predictions.jsonl -TotalCount 5
```

## Read Task 4 judge results in a spreadsheet

Open the generated UTF-8 CSV file:

```text
results/task4/judgments/<prediction-file>__judge_<judge-model>.csv
```

## Output-file meanings

| Suffix | Meaning |
|---|---|
| `.jsonl` | Item-level predictions or judgments |
| `.metrics.json` | Aggregate metrics for Tasks 1–3 |
| `.errors.jsonl` | Incorrect, invalid, or failed records |
| `.summary.json` | Aggregate Task 4 judge scores |
| `.csv` | Spreadsheet-friendly Task 4 judgment results |

---

# 12. Custom output paths

Use `--output` to select a prediction path for Tasks 1–4.

```bash
python scripts/task3.py \
  --mode all \
  --type type2 \
  --category semantic \
  --context term_only \
  --model gpt-5.4 \
  --input data/task3/task3_type2_semantic.xlsx \
  --output results/custom/gpt-5.4_semantic.jsonl
```

For Tasks 1–3, aggregate and error paths can also be set explicitly:

```bash
--metrics-output results/custom/metrics.json
--errors-output results/custom/errors.jsonl
```

For the Task 4 judge:

```bash
--output-jsonl results/custom/judgments.jsonl
--output-csv results/custom/judgments.csv
--summary-output results/custom/judgments.summary.json
```

---

# Citation

Citation information will be added after the anonymous review period.

```bibtex
@inproceedings{koneobench2026, 
  title = {{KoNeoBench}: A Curated Evaluation Dataset for LLM Understanding of Korean Neologisms}, 
  author = {Lee, Soha and Lee, Soojin and Yang, Heesung and Song, Hyunju and Lee, Hyunji and An, Jinsan and Shin, Jeongwan and Park, Jin Hyun and Lee, Jun and Park, Hyeyoung and Nam, Kilim}, 
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026}, 
  year = {2026} }
```