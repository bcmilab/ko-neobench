from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_yaml
from utils.io import (
    append_jsonl,
    clean_cell,
    completed_ids,
    load_table,
    resolve_column,
)
from utils.model_backends import create_backend
from utils.prompting import build_task1_mcq
from utils.scoring import evaluate_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or evaluate K-NeoBench Task 1 across supported models."
    )
    parser.add_argument("--mode", choices=["run", "evaluate", "all", "prepare"], default="all")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "task1" / "Task1_type1_50.xlsx",
    )
    parser.add_argument("--sheet", default=None, help="Excel sheet name or zero-based index")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--errors-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=REPO_ROOT / "configs" / "task1.yaml",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=REPO_ROOT / "configs" / "models.yaml",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=REPO_ROOT / "prompts" / "task1.yaml",
    )
    return parser.parse_args()


def _coerce_sheet(value: Any) -> str | int:
    if value is None:
        return 0
    text = str(value)
    return int(text) if text.isdigit() else text


def _resolve_task1_columns(df, aliases: dict[str, list[str]]) -> dict[str, str | None]:
    return {
        "id": resolve_column(df.columns, aliases.get("id", []), required=False),
        "question": resolve_column(df.columns, aliases["question"]),
        "answer": resolve_column(df.columns, aliases["answer"]),
        "confusion": resolve_column(df.columns, aliases["confusion"]),
        "random_1": resolve_column(df.columns, aliases["random_1"]),
        "random_2": resolve_column(df.columns, aliases["random_2"]),
        "random_3": resolve_column(df.columns, aliases["random_3"]),
        "year": resolve_column(df.columns, aliases.get("year", []), required=False),
    }


def _record_id(row_index: int, row, id_column: str | None) -> str:
    if id_column:
        value = clean_cell(row[id_column])
        if value:
            return value
    return f"T1_{row_index + 1:04d}"


def run(args: argparse.Namespace) -> Path:
    task_config = load_yaml(args.task_config)
    models_config = load_yaml(args.models_config).get("models", {})
    prompt_config = load_yaml(args.prompt_config)

    if args.model not in models_config:
        raise KeyError(
            f"Unknown model '{args.model}'. Available: {', '.join(models_config)}"
        )

    seed = args.seed if args.seed is not None else int(task_config.get("seed", 42))
    condition = args.condition or str(task_config.get("condition", "default"))
    sheet = _coerce_sheet(args.sheet if args.sheet is not None else task_config.get("sheet_name", 0))
    default_name = (
        f"{args.model}_{condition}_prepared.jsonl"
        if args.mode == "prepare"
        else f"{args.model}_{condition}.jsonl"
    )
    output = args.output or (REPO_ROOT / "results" / "task1" / default_name)

    df = load_table(args.input, sheet_name=sheet)
    if args.limit is not None:
        df = df.head(args.limit)
    columns = _resolve_task1_columns(df, task_config["column_aliases"])

    done = completed_ids(output, retry_errors=args.retry_errors) if args.resume else set()
    if not args.resume and output.exists():
        output.unlink()

    backend = None if args.mode == "prepare" else create_backend(models_config[args.model])
    failures = 0

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc=f"Task1/{args.model}"):
        record_id = _record_id(int(row_index), row, columns["id"])
        if record_id in done:
            continue

        question = clean_cell(row[columns["question"]])
        correct = clean_cell(row[columns["answer"]])
        confusion = clean_cell(row[columns["confusion"]])
        random_distractors = [
            clean_cell(row[columns["random_1"]]),
            clean_cell(row[columns["random_2"]]),
            clean_cell(row[columns["random_3"]]),
        ]
        year = clean_cell(row[columns["year"]]) if columns["year"] else ""
        row_seed = seed + int(row_index)

        base_record: dict[str, Any] = {
            "id": record_id,
            "row_index": int(row_index),
            "task": "task1",
            "condition": condition,
            "model": args.model,
            "model_id": models_config[args.model]["model_id"],
            "year": year,
            "seed": row_seed,
            "question": question,
        }

        try:
            mcq = build_task1_mcq(
                question=question,
                correct=correct,
                confusion=confusion,
                random_distractors=random_distractors,
                system_template=prompt_config["system"],
                user_template=prompt_config["user"],
                seed=row_seed,
            )
            base_record.update(
                {
                    "options": mcq.options,
                    "gold": mcq.gold_letter,
                    "gold_text": mcq.gold_text,
                    "confusion_letter": mcq.confusion_letter,
                    "confusion_text": mcq.confusion_text,
                    "prompt": mcq.user_prompt if args.mode == "prepare" else None,
                }
            )

            if args.mode == "prepare":
                base_record.update(
                    {"model_answer": "", "raw_output": "", "status": "prepared", "error": ""}
                )
            else:
                result = backend.generate(mcq.system_prompt, mcq.user_prompt)  # type: ignore[union-attr]
                base_record.update(
                    {
                        "model_answer": result.parsed_answer,
                        "raw_output": result.raw_output,
                        "status": "ok",
                        "error": "",
                    }
                )
        except Exception as exc:
            failures += 1
            base_record.update(
                {
                    "model_answer": "",
                    "raw_output": "",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if args.fail_fast:
                raise
            print(f"[ERROR] {record_id}: {base_record['error']}", file=sys.stderr)
            if not isinstance(exc, (ValueError, KeyError)):
                traceback.print_exc()

        append_jsonl(output, base_record)

    print(f"Saved predictions: {output}")
    print(f"Failures in this run: {failures}")
    return output


def evaluate(args: argparse.Namespace, prediction_path: Path | None = None) -> dict[str, Any]:
    task_config = load_yaml(args.task_config)
    condition = args.condition or str(task_config.get("condition", "default"))
    prediction_path = prediction_path or args.output or (
        REPO_ROOT / "results" / "task1" / f"{args.model}_{condition}.jsonl"
    )
    metrics_path = args.metrics_output or prediction_path.with_suffix(".metrics.json")
    errors_path = args.errors_output or prediction_path.with_suffix(".errors.jsonl")
    metrics = evaluate_jsonl(
        prediction_path, metrics_path=metrics_path, errors_path=errors_path
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved error records: {errors_path}")
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
