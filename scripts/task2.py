from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_yaml
from utils.io import append_jsonl, clean_cell, completed_ids, load_table, resolve_column
from utils.model_backends import create_backend
from utils.prompting import build_task2_prompt
from utils.scoring import (
    evaluate_task2_file,
    normalize_task2_answer,
    score_task2_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or evaluate K-NeoBench Task 2 component reconstruction."
    )
    parser.add_argument("--mode", choices=["run", "evaluate", "all", "prepare"], default="all")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "task2" / "task2_data.csv",
    )
    parser.add_argument("--sheet", default=None, help="Excel sheet name or zero-based index")
    parser.add_argument("--shots", type=int, default=None, help="Supported values: 0 to 5")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--errors-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=REPO_ROOT / "configs" / "task2.yaml",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=REPO_ROOT / "configs" / "models.yaml",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=REPO_ROOT / "prompts" / "task2.yaml",
    )
    return parser.parse_args()


def _coerce_sheet(value: Any) -> str | int:
    if value is None:
        return 0
    text = str(value)
    return int(text) if text.isdigit() else text


def _resolve_columns(df, aliases: dict[str, list[str]]) -> dict[str, str | None]:
    return {
        "id": resolve_column(df.columns, aliases.get("id", []), required=False),
        "term": resolve_column(df.columns, aliases["term"]),
        "gold": resolve_column(df.columns, aliases["gold"]),
        "year": resolve_column(df.columns, aliases.get("year", []), required=False),
        "word_group": resolve_column(df.columns, aliases.get("word_group", []), required=False),
    }


def _record_id(row_index: int, row, id_column: str | None) -> str:
    if id_column:
        value = clean_cell(row[id_column])
        if value:
            return value
    return f"T2_{row_index + 1:04d}"


def _merged_model_config(
    *,
    model_alias: str,
    models_config: dict[str, dict[str, Any]],
    task_config: dict[str, Any],
) -> dict[str, Any]:
    if model_alias not in models_config:
        raise KeyError(
            f"Unknown model '{model_alias}'. Available: {', '.join(models_config)}"
        )
    merged = dict(models_config[model_alias])
    merged.update(task_config.get("backend_overrides", {}).get(model_alias, {}))
    return merged


def run(args: argparse.Namespace) -> Path:
    task_config = load_yaml(args.task_config)
    prompt_config = load_yaml(args.prompt_config)
    models_config = load_yaml(args.models_config).get("models", {})
    model_config = _merged_model_config(
        model_alias=args.model,
        models_config=models_config,
        task_config=task_config,
    )

    shots = args.shots if args.shots is not None else int(task_config.get("shots", 5))
    # Build once to validate the requested number of fixed examples.
    prompt_probe = build_task2_prompt(prompt_config=prompt_config, term="검증용", shots=shots)
    example_terms = set(prompt_probe.example_terms)

    sheet = _coerce_sheet(args.sheet if args.sheet is not None else task_config.get("sheet_name", 0))
    suffix = "prepared" if args.mode == "prepare" else "predictions"
    output = args.output or (
        REPO_ROOT / "results" / "task2" / f"{args.model}_{shots}shot_{suffix}.jsonl"
    )

    df = load_table(args.input, sheet_name=sheet).reset_index(drop=True)
    columns = _resolve_columns(df, task_config["column_aliases"])
    if example_terms:
        before = len(df)
        df = df[~df[columns["term"]].astype(str).isin(example_terms)].copy()
        excluded = before - len(df)
        if excluded:
            print(f"Excluded {excluded} benchmark rows overlapping fixed few-shot examples.")
    if args.limit is not None:
        df = df.head(args.limit)

    done = completed_ids(output, retry_errors=args.retry_errors) if args.resume else set()
    if not args.resume and output.exists():
        output.unlink()

    backend = None if args.mode == "prepare" else create_backend(model_config)
    delay = float(task_config.get("request_delay_seconds", 0.0))
    if model_config.get("provider") == "transformers":
        delay = 0.0

    failures = 0
    for row_index, row in tqdm(df.iterrows(), total=len(df), desc=f"Task2/{args.model}"):
        source_index = int(row_index)
        record_id = _record_id(source_index, row, columns["id"])
        if record_id in done:
            continue

        term = clean_cell(row[columns["term"]])
        gold = normalize_task2_answer(row[columns["gold"]])
        year = clean_cell(row[columns["year"]]) if columns["year"] else ""
        word_group = clean_cell(row[columns["word_group"]]) if columns["word_group"] else ""
        prompt = build_task2_prompt(prompt_config=prompt_config, term=term, shots=shots)

        record: dict[str, Any] = {
            "id": record_id,
            "row_index": source_index,
            "task": "task2",
            "model": args.model,
            "model_id": model_config["model_id"],
            "provider": model_config["provider"],
            "shots": shots,
            "year": year,
            "term": term,
            "gold": gold,
            "word_group": word_group,
            "fewshot_terms": list(prompt.example_terms),
        }

        try:
            if args.mode == "prepare":
                record.update(
                    {
                        "system_prompt": prompt.system_prompt,
                        "user_prompt": prompt.user_prompt,
                        "model_answer": "",
                        "raw_output": "",
                        "status": "prepared",
                        "error": "",
                    }
                )
            else:
                result = backend.generate(prompt.system_prompt, prompt.user_prompt)  # type: ignore[union-attr]
                prediction = normalize_task2_answer(result.raw_output)
                score = score_task2_prediction(gold, prediction)
                record.update(
                    {
                        "model_answer": prediction,
                        "raw_output": result.raw_output,
                        "status": "ok",
                        "error": "",
                        "coverage": score["coverage"],
                        "recall": score["recall"],
                        "precision": score["precision"],
                        "f1": score["f1"],
                        "exact_match": score["exact_match"],
                        "unordered_exact_match": score["unordered_exact_match"],
                    }
                )
        except Exception as exc:
            failures += 1
            zero_score = score_task2_prediction(gold, "")
            record.update(
                {
                    "model_answer": "",
                    "raw_output": "",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "coverage": zero_score["coverage"],
                    "recall": zero_score["recall"],
                    "precision": zero_score["precision"],
                    "f1": zero_score["f1"],
                    "exact_match": False,
                    "unordered_exact_match": False,
                }
            )
            if args.fail_fast:
                raise
            print(f"[ERROR] {record_id}: {record['error']}", file=sys.stderr)
            if not isinstance(exc, (ValueError, KeyError)):
                traceback.print_exc()

        append_jsonl(output, record)
        if delay > 0 and args.mode != "prepare":
            time.sleep(delay)

    print(f"Saved Task 2 records: {output}")
    print(f"Failures in this run: {failures}")
    return output


def evaluate(args: argparse.Namespace, prediction_path: Path | None = None) -> dict[str, Any]:
    task_config = load_yaml(args.task_config)
    shots = args.shots if args.shots is not None else int(task_config.get("shots", 5))
    prediction_path = prediction_path or args.output or (
        REPO_ROOT / "results" / "task2" / f"{args.model}_{shots}shot_predictions.jsonl"
    )
    metrics_path = args.metrics_output or prediction_path.with_suffix(".metrics.json")
    errors_path = args.errors_output or prediction_path.with_suffix(".errors.jsonl")
    metrics = evaluate_task2_file(
        prediction_path,
        metrics_path=metrics_path,
        errors_path=errors_path,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved non-exact records: {errors_path}")
    return metrics


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    if args.mode in {"run", "prepare"}:
        run(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        predictions = run(args)
        evaluate(args, predictions)


if __name__ == "__main__":
    main()
