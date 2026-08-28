from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_yaml
from utils.definition_generation import (
    build_local_user_prompt,
    build_structured_payload,
    create_definition_backend,
    definition_schema,
    normalize_term_plain,
)
from utils.io import append_jsonl, clean_cell, completed_ids, load_table, resolve_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate definitions for K-NeoBench Task 4."
    )
    parser.add_argument("--mode", choices=["run", "prepare"], default="run")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument(
        "--condition",
        choices=["term_only", "term_example"],
        default="term_only",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--sheet", default=None, help="Excel sheet name or zero-based index")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=REPO_ROOT / "configs" / "task4.yaml",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=REPO_ROOT / "configs" / "models.yaml",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=REPO_ROOT / "prompts" / "task4.yaml",
    )
    return parser.parse_args()


def _coerce_sheet(value: Any) -> str | int:
    if value is None:
        return 0
    text = str(value)
    return int(text) if text.isdigit() else text


def _resolve_columns(df, aliases: dict[str, list[str]], condition: str):
    return {
        "id": resolve_column(df.columns, aliases.get("id", []), required=False),
        "term": resolve_column(df.columns, aliases["term"]),
        "example": resolve_column(
            df.columns,
            aliases["example"],
            required=(condition == "term_example"),
        ),
        "gold": resolve_column(df.columns, aliases["gold"]),
        "year": resolve_column(df.columns, aliases["year"]),
        "word_group": resolve_column(df.columns, aliases.get("word_group", []), required=False),
    }


def _record_id(row_index: int, row, id_column: Optional[str]) -> str:
    if id_column:
        value = clean_cell(row[id_column])
        if value:
            return value
    return f"T4_{row_index + 1:04d}"


def _model_config(
    model_alias: str,
    task_config: dict[str, Any],
    models_config: dict[str, Any],
) -> dict[str, Any]:
    supported = list(task_config.get("supported_models", []))
    if model_alias not in supported:
        raise ValueError(
            f"Task 4 has no uploaded-notebook implementation for '{model_alias}'. "
            f"Supported: {', '.join(supported)}"
        )
    if model_alias not in models_config:
        raise KeyError(f"Unknown model alias in configs/models.yaml: {model_alias}")
    merged = dict(models_config[model_alias])
    merged.update(task_config["backend_overrides"][model_alias])
    return merged


def _prepare_request(
    *,
    model_config: dict[str, Any],
    term: str,
    example: Optional[str],
    system_prompt: str,
) -> tuple[str, str]:
    provider = str(model_config["provider"])
    if provider in {"openai_structured", "upstage_structured"}:
        return system_prompt, build_structured_payload(term, example)

    prepared_system = system_prompt
    if model_config.get("append_json_instruction", False):
        prepared_system += (
            "\n\n아래 사용자 입력은 JSON이다. 반드시 단일 JSON 객체만 출력하라. "
            "키는 term, definition 두 개만 사용하라."
        )
    user_prompt = build_local_user_prompt(
        term,
        example,
        forbid_scripts=bool(model_config.get("forbid_scripts", False)),
    )
    return prepared_system, user_prompt


def run(args: argparse.Namespace) -> Path:
    task_config = load_yaml(args.task_config)
    prompt_config = load_yaml(args.prompt_config)
    models_config = load_yaml(args.models_config).get("models", {})
    model_config = _model_config(args.model, task_config, models_config)

    condition_config = task_config["conditions"][args.condition]
    input_path = args.input or REPO_ROOT / condition_config["default_input"]
    sheet = _coerce_sheet(
        args.sheet if args.sheet is not None else task_config.get("sheet_name", 0)
    )
    suffix = "prepared" if args.mode == "prepare" else "definitions"
    output = args.output or (
        REPO_ROOT
        / "results"
        / "task4"
        / f"{args.model}_{args.condition}_{suffix}.jsonl"
    )

    df = load_table(input_path, sheet_name=sheet).reset_index(drop=True)
    columns = _resolve_columns(df, task_config["column_aliases"], args.condition)
    if args.limit is not None:
        df = df.head(args.limit)

    done = completed_ids(output, retry_errors=args.retry_errors) if args.resume else set()
    if not args.resume and output.exists():
        output.unlink()

    backend = None if args.mode == "prepare" else create_definition_backend(model_config)
    delay = float(task_config.get("request_delay_seconds", 0.0))
    if model_config["provider"] in {
        "transformers_definition"
    }:
        delay = 0.0

    system_prompt = clean_cell(prompt_config[args.condition]["system"])
    replace_mask = bool(model_config.get("replace_mask", True))
    failures = 0

    for row_index, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Task4/{args.condition}/{args.model}"
    ):
        record_id = _record_id(int(row_index), row, columns["id"])
        if record_id in done:
            continue

        raw_term = clean_cell(row[columns["term"]])
        if not raw_term:
            continue
        term = normalize_term_plain(raw_term)
        gold = clean_cell(row[columns["gold"]])
        year = clean_cell(row[columns["year"]])
        word_group = clean_cell(row[columns["word_group"]]) if columns.get("word_group") else None

        example: Optional[str] = None
        if args.condition == "term_example":
            example_text = clean_cell(row[columns["example"]])  # type: ignore[index]
            if replace_mask and example_text:
                example_text = example_text.replace("[MASK]", term).replace(
                    " [MASK] ", f" {term} "
                )
            example = example_text or None

        record: dict[str, Any] = {
            "id": record_id,
            "row_index": int(row_index),
            "task": "task4",
            "condition": args.condition,
            "model": args.model,
            "model_id": model_config["model_id"],
            "provider": model_config["provider"],
            "term": term,
            "term_raw": raw_term,
            "example": example,
            "gold": gold,
            "year": year,
            "word_group": word_group,
            "replace_mask": replace_mask,
        }

        try:
            if args.mode == "prepare":
                prepared_system, prepared_user = _prepare_request(
                    model_config=model_config,
                    term=term,
                    example=example,
                    system_prompt=system_prompt,
                )
                record.update(
                    {
                        "system_prompt": prepared_system,
                        "user_prompt": prepared_user,
                        "definition": "",
                        "raw_output": "",
                        "status": "prepared",
                        "error": "",
                    }
                )
            else:
                result = backend.generate(  # type: ignore[union-attr]
                    term=term,
                    example=example,
                    system_prompt=system_prompt,
                )
                record.update(
                    {
                        "generated_term": result.term,
                        "definition": result.definition,
                        "raw_output": result.raw_output,
                        "status": "ok",
                        "error": "",
                    }
                )
        except Exception as exc:
            failures += 1
            record.update(
                {
                    "generated_term": "",
                    "definition": "",
                    "raw_output": "",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            )
            if args.fail_fast:
                raise
            print(f"[ERROR] {record_id}: {record['error']}", file=sys.stderr)
            traceback.print_exc()

        append_jsonl(output, record)
        if delay > 0 and args.mode == "run":
            time.sleep(delay)

    print(f"Saved Task 4 records: {output}")
    print(f"Failures in this run: {failures}")
    return output


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    run(parse_args())


if __name__ == "__main__":
    main()
