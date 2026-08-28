from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.definition_judging import (
    GeminiDefinitionJudge,
    JudgeRuntimeConfig,
    MalformedJudgeOutputError,
    append_jsonl,
    clean_text,
    failed_result,
    jsonify_dataframe_for_csv,
    load_jsonl,
    load_table,
    load_yaml,
    merge_gold_predictions,
    prompt_sha256,
    save_jsonl,
    save_summary,
    summarize_judgments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge K-NeoBench Task 4 definition outputs with the fixed 6-3-1 rubric."
    )
    parser.add_argument("--mode", choices=["run", "summarize"], default="run")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument(
        "--condition", choices=["term_only", "term_example"], default="term_only"
    )
    parser.add_argument("--sheet", default=None, help="Excel sheet name or zero-based index")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None, help="Judgment JSONL for summarize mode")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "task4_judge.yaml",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=REPO_ROOT / "prompts" / "task4_judge.yaml",
    )
    return parser.parse_args()


def coerce_sheet(value: Any) -> str | int:
    if value is None:
        return 0
    text = str(value)
    return int(text) if text.isdigit() else text


def slugify(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_")


def default_outputs(
    *, predictions: Path, judge_model: str
) -> tuple[Path, Path, Path]:
    stem = predictions.stem
    judge_slug = slugify(judge_model)
    base = REPO_ROOT / "results" / "task4" / "judgments"
    prefix = base / f"{stem}__judge_{judge_slug}"
    return (
        Path(str(prefix) + ".jsonl"),
        Path(str(prefix) + ".csv"),
        Path(str(prefix) + ".summary.json"),
    )


def done_keys(path: Path, *, retry_errors: bool) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in load_jsonl(path):
        if retry_errors and row.get("judge_status") != "ok":
            continue
        key = clean_text(row.get("item_key"))
        if key:
            done.add(key)
    return done


def existing_rows(path: Path, *, retry_errors: bool) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = load_jsonl(path)
    if not retry_errors:
        return rows
    return [row for row in rows if row.get("judge_status") == "ok"]


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.predictions is None or args.gold is None:
        raise ValueError("--predictions and --gold are required in run mode.")

    load_dotenv(REPO_ROOT / ".env")
    config = load_yaml(args.config)
    prompt_config = load_yaml(args.prompt_config)
    prompt = clean_text(prompt_config["system_prompt"])

    runtime_raw = config["judge"]
    judge_model = args.judge_model or os.getenv("GEMINI_MODEL") or str(runtime_raw["model"])
    runtime = JudgeRuntimeConfig(
        model=judge_model,
        temperature=float(runtime_raw.get("temperature", 0.0)),
        thinking_budget=runtime_raw.get("thinking_budget"),
        max_output_tokens=int(runtime_raw.get("max_output_tokens", 4096)),
        api_retries=int(runtime_raw.get("api_retries", 4)),
        request_delay_seconds=float(runtime_raw.get("request_delay_seconds", 0.2)),
        api_key_env=str(runtime_raw.get("api_key_env", "GEMINI_API_KEY")),
    )

    default_jsonl, default_csv, default_summary = default_outputs(
        predictions=args.predictions, judge_model=judge_model
    )
    output_jsonl = args.output_jsonl or default_jsonl
    output_csv = args.output_csv or default_csv
    summary_output = args.summary_output or default_summary

    gold_df = load_table(
        args.gold,
        sheet_name=coerce_sheet(
            args.sheet if args.sheet is not None else config.get("sheet_name", 0)
        ),
    )
    prediction_df = load_table(args.predictions)
    merged, merge_key, resolved = merge_gold_predictions(
        gold_df=gold_df,
        prediction_df=prediction_df,
        column_aliases=config["column_aliases"],
    )
    if args.limit is not None:
        merged = merged.head(args.limit)

    if not args.resume and output_jsonl.exists():
        output_jsonl.unlink()
    prior_rows = existing_rows(output_jsonl, retry_errors=args.retry_errors) if args.resume else []
    done = done_keys(output_jsonl, retry_errors=args.retry_errors) if args.resume else set()
    if args.retry_errors and output_jsonl.exists():
        save_jsonl(output_jsonl, prior_rows)

    judge = GeminiDefinitionJudge(system_prompt=prompt, config=runtime)
    rubric = config.get("rubric", {})
    prompt_hash = prompt_sha256(prompt)

    term_column = resolved["term"]
    gold_definition_column = resolved["gold_definition"]
    example_column = resolved["example"]
    prediction_definition_column = resolved["prediction_definition"]

    failures = 0
    for _, row in tqdm(
        merged.iterrows(), total=len(merged), desc=f"Task4 judge/{args.condition}"
    ):
        item_key = clean_text(row["item_key"])
        if item_key in done:
            continue

        item_id = clean_text(row.get("__item_id", ""))
        row_index = int(row["__row_index"])
        term = clean_text(row[term_column])
        gold_definition = clean_text(row[gold_definition_column])
        predicted_definition = clean_text(row[prediction_definition_column])
        example: Optional[str] = None
        if example_column:
            example_value = clean_text(row[example_column])
            example = example_value or None

        base: dict[str, Any] = {
            "item_key": item_key,
            "id": item_id or None,
            "row_index": row_index,
            "task": "task4",
            "condition": args.condition,
            "evaluated_model": clean_text(row.get("model", "")) or None,
            "judge_model": judge_model,
            "rubric": rubric.get("name", "6-3-1"),
            "rubric_version": str(rubric.get("version", "1.0")),
            "judge_prompt_sha256": prompt_hash,
            "merge_key": merge_key,
            "term": term,
            "example": example,
            "gold_definition": gold_definition,
            "predicted_definition": predicted_definition,
        }

        prediction_status = clean_text(row.get("status", ""))
        if prediction_status and prediction_status not in {"ok", "success"}:
            result = {
                **base,
                **failed_result(
                    reason="prediction_error",
                    comment=f"Prediction status was '{prediction_status}'.",
                ),
            }
        elif not predicted_definition:
            result = {
                **base,
                **failed_result(
                    reason="empty_prediction",
                    comment="Prediction definition is empty.",
                ),
            }
        else:
            try:
                response = judge.judge(
                    term=term,
                    gold_definition=gold_definition,
                    predicted_definition=predicted_definition,
                    example=example,
                )
                result = {
                    **base,
                    "judge_status": "ok",
                    "judge_error": "",
                    **response.scores,
                    "judge_raw_output": response.raw_output,
                }
            except MalformedJudgeOutputError as exc:
                failures += 1
                result = {
                    **base,
                    **failed_result(
                        reason="malformed_judge_output",
                        comment=str(exc)[:2000],
                    ),
                    "judge_raw_output": exc.raw_output,
                }
                if args.fail_fast:
                    raise
            except Exception as exc:
                failures += 1
                result = {
                    **base,
                    **failed_result(
                        reason="judge_api_error",
                        comment=f"{type(exc).__name__}: {str(exc)[:2000]}",
                    ),
                }
                if args.fail_fast:
                    raise

        append_jsonl(output_jsonl, result)
        done.add(item_key)
        if runtime.request_delay_seconds:
            time.sleep(runtime.request_delay_seconds)

    rows = load_jsonl(output_jsonl)
    output_frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    jsonify_dataframe_for_csv(output_frame).to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )

    summary = summarize_judgments(output_frame)
    summary.update(
        {
            "task": "task4",
            "condition": args.condition,
            "judge_model": judge_model,
            "rubric": rubric.get("name", "6-3-1"),
            "rubric_version": str(rubric.get("version", "1.0")),
            "judge_prompt_sha256": prompt_hash,
            "prediction_file": str(args.predictions),
            "gold_file": str(args.gold),
            "merge_key": merge_key,
            "n_gold_rows": int(len(gold_df)),
            "n_prediction_rows": int(len(prediction_df)),
            "n_merged_rows": int(len(merged)),
        }
    )
    save_summary(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        print(f"Completed with {failures} judge failures.", file=sys.stderr)
    return output_jsonl, output_csv, summary_output


def summarize(args: argparse.Namespace) -> Path:
    input_path = args.input or args.output_jsonl
    if input_path is None:
        raise ValueError("--input is required in summarize mode.")
    rows = load_jsonl(input_path)
    summary = summarize_judgments(rows)
    output = args.summary_output or Path(str(input_path.with_suffix("")) + ".summary.json")
    save_summary(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
