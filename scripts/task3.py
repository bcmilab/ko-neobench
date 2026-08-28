from __future__ import annotations

import argparse
import json
import re
import string
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
from utils.prompting import (
    LETTERS,
    build_task3_type1_prompt,
    build_task3_type2_prompt,
)
from utils.scoring import (
    evaluate_task3_type1_file,
    evaluate_task3_type2_file,
    extract_choice,
    parse_task3_type1_output,
    parse_task3_type2_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or evaluate K-NeoBench Task 3. Use --category semantic for 의미범주 "
            "and --category special for 전문 분야."
        )
    )
    parser.add_argument("--mode", choices=["run", "evaluate", "all", "prepare"], default="all")
    parser.add_argument("--type", dest="task_type", choices=["type1", "type2"], default="type1")
    parser.add_argument("--category", choices=["semantic", "special"], default="semantic")
    parser.add_argument(
        "--context",
        choices=["term_only", "term_example"],
        default=None,
        help=(
            "Task 3 Type 2 input condition. term_only uses the headword only; "
            "term_example also supplies the usage example after replacing [MASK]. "
            "If omitted, the value in configs/task3.yaml is used."
        ),
    )
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--sheet", default=None, help="Excel sheet name or zero-based index")
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
        default=REPO_ROOT / "configs" / "task3.yaml",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=REPO_ROOT / "configs" / "models.yaml",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=REPO_ROOT / "prompts" / "task3.yaml",
    )
    return parser.parse_args()


def _coerce_sheet(value: Any) -> str | int:
    if value is None:
        return 0
    text = str(value)
    return int(text) if text.isdigit() else text


def _subtask_config(task_config: dict[str, Any], task_type: str, category: str) -> dict[str, Any]:
    try:
        config = task_config[task_type][category]
    except KeyError as exc:
        raise KeyError(f"Unknown Task 3 branch: {task_type}/{category}") from exc
    if not bool(config.get("implemented", False)):
        raise NotImplementedError(f"Task 3 {task_type}/{category} is not implemented.")
    return config


def _resolve_type1_columns(df, aliases: dict[str, list[str]]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {
        "id": resolve_column(df.columns, aliases.get("id", []), required=False),
        "answer": resolve_column(df.columns, aliases["answer"]),
        "target_term": resolve_column(df.columns, aliases["target_term"]),
        "target_label": resolve_column(df.columns, aliases["target_label"]),
        "distractor_label": resolve_column(df.columns, aliases["distractor_label"]),
        "question": resolve_column(df.columns, aliases.get("question", []), required=False),
        "year": resolve_column(df.columns, aliases.get("year", []), required=False),
    }
    for letter in LETTERS:
        resolved[f"option_{letter}"] = resolve_column(df.columns, aliases[f"option_{letter}"])
    return resolved


def _resolve_type2_columns(df, aliases: dict[str, list[str]]) -> dict[str, str | None]:
    return {
        "id": resolve_column(df.columns, aliases.get("id", []), required=False),
        "headword": resolve_column(df.columns, aliases["headword"]),
        "label": resolve_column(df.columns, aliases["label"]),
        "example": resolve_column(df.columns, aliases.get("example", []), required=False),
        "year": resolve_column(df.columns, aliases.get("year", []), required=False),
    }


def _merged_model_config(
    *,
    model_alias: str,
    models_config: dict[str, dict[str, Any]],
    task_config: dict[str, Any],
    subtask_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model_alias not in models_config:
        raise KeyError(f"Unknown model '{model_alias}'. Available: {', '.join(models_config)}")
    merged = dict(models_config[model_alias])
    merged.update(task_config.get("backend_overrides", {}).get(model_alias, {}))
    if subtask_config:
        merged.update(subtask_config.get("backend_defaults", {}))
        merged.update(subtask_config.get("backend_overrides", {}).get(model_alias, {}))
    return merged


def _record_id(
    *,
    row_position: int,
    row,
    id_column: str | None,
    task_type: str,
    category: str,
    context: str | None = None,
) -> str:
    context_part = f"_{context}" if context else ""
    if id_column:
        value = clean_cell(row[id_column])
        if value:
            return f"T3_{task_type}_{category}{context_part}_{value}"
    return f"T3_{task_type}_{category}{context_part}_{row_position + 1:04d}"


def _default_input(subtask_config: dict[str, Any]) -> Path:
    configured = Path(str(subtask_config["default_input"]))
    return configured if configured.is_absolute() else REPO_ROOT / configured


def _resolved_context(args: argparse.Namespace, subtask_config: dict[str, Any]) -> str | None:
    if args.task_type == "type1":
        return None
    return args.context or str(subtask_config.get("default_context", "term_only"))


def _default_output(
    args: argparse.Namespace,
    *,
    prepared: bool = False,
    context: str | None = None,
) -> Path:
    suffix = "prepared" if prepared else "predictions"
    model_part = args.model
    if args.task_type == "type2" and context:
        model_part = f"{model_part}_{context}"
    return (
        REPO_ROOT
        / "results"
        / "task3"
        / f"{args.task_type}_{args.category}"
        / f"{model_part}_{suffix}.jsonl"
    )


def _fill_mask(example: Any, headword: str) -> str:
    text = clean_cell(example)
    if not text:
        return ""
    return re.sub(r"\[\s*MASK\s*\]", headword.strip(), text, flags=re.IGNORECASE)


def _sheet_for(args: argparse.Namespace, task_config: dict[str, Any], subtask_config: dict[str, Any]) -> str | int:
    selected = args.sheet
    if selected is None:
        selected = subtask_config.get("sheet_name", task_config.get("sheet_name", 0))
    return _coerce_sheet(selected)


def run_type1(args: argparse.Namespace) -> Path:
    task_config = load_yaml(args.task_config)
    prompt_root = load_yaml(args.prompt_config)
    models_config = load_yaml(args.models_config).get("models", {})
    subtask_config = _subtask_config(task_config, args.task_type, args.category)
    prompt_config = prompt_root[args.task_type][args.category]
    model_config = _merged_model_config(
        model_alias=args.model,
        models_config=models_config,
        task_config=task_config,
        subtask_config=subtask_config,
    )

    input_path = args.input or _default_input(subtask_config)
    sheet = _sheet_for(args, task_config, subtask_config)
    output = args.output or _default_output(args, prepared=args.mode == "prepare")

    df = load_table(input_path, sheet_name=sheet).reset_index(drop=True)
    columns = _resolve_type1_columns(df, subtask_config["column_aliases"])
    if args.limit is not None:
        df = df.head(args.limit).copy()

    category_labels = sorted(
        {
            clean_cell(value)
            for value in df[columns["target_label"]].tolist()  # type: ignore[index]
            if clean_cell(value)
        }
    )
    if not category_labels:
        raise ValueError("No target categories/domains were found in the Task 3 input file")

    done = completed_ids(output, retry_errors=args.retry_errors) if args.resume else set()
    if not args.resume and output.exists():
        output.unlink()

    backend = None if args.mode == "prepare" else create_backend(model_config)
    delay = float(task_config.get("request_delay_seconds", 0.0))
    if model_config.get("provider") == "transformers":
        delay = 0.0

    failures = 0
    for row_position, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Task3/{args.task_type}/{args.category}/{args.model}"
    ):
        record_id = _record_id(
            row_position=int(row_position),
            row=row,
            id_column=columns["id"],
            task_type=args.task_type,
            category=args.category,
        )
        if record_id in done:
            continue

        options = {
            letter: clean_cell(row[columns[f"option_{letter}"]])  # type: ignore[index]
            for letter in LETTERS
        }
        gold = extract_choice(row[columns["answer"]])  # type: ignore[index]
        gold_term = clean_cell(row[columns["target_term"]])  # type: ignore[index]
        target_label = clean_cell(row[columns["target_label"]])  # type: ignore[index]
        distractor_label = clean_cell(row[columns["distractor_label"]])  # type: ignore[index]
        question = (
            clean_cell(row[columns["question"]])
            if columns["question"]
            else clean_cell(prompt_config.get("default_question"))
        )
        year = clean_cell(row[columns["year"]]) if columns["year"] else ""

        prompt = build_task3_type1_prompt(
            prompt_config=prompt_config,
            options=options,
            category_labels=category_labels,
        )

        record: dict[str, Any] = {
            "id": record_id,
            "row_index": int(row_position),
            "task": "task3",
            "task_type": args.task_type,
            "category": args.category,
            "model": args.model,
            "model_id": model_config["model_id"],
            "provider": model_config["provider"],
            "question": question,
            "options": options,
            "gold": gold,
            "gold_term": gold_term,
            "target_label": target_label,
            "distractor_label": distractor_label,
            "year": year,
            "category_labels": category_labels,
        }
        if args.category == "semantic":
            record.update({"target_category": target_label, "distractor_category": distractor_label})
        else:
            record.update({"target_domain": target_label, "distractor_domain": distractor_label})

        try:
            if args.mode == "prepare":
                record.update(
                    {
                        "system_prompt": prompt.system_prompt,
                        "user_prompt": prompt.user_prompt,
                        "model_answer": "",
                        "raw_output": "",
                        "reason": "",
                        "is_correct": False,
                        "status": "prepared",
                        "error": "",
                    }
                )
            else:
                result = backend.generate(prompt.system_prompt, prompt.user_prompt)  # type: ignore[union-attr]
                prediction, reason = parse_task3_type1_output(result.raw_output)
                if not prediction:
                    prediction = extract_choice(result.parsed_answer)
                record.update(
                    {
                        "model_answer": prediction,
                        "pred": prediction,
                        "raw_output": result.raw_output,
                        "reason": reason,
                        "is_correct": prediction == gold,
                        "status": "ok",
                        "error": "",
                    }
                )
        except Exception as exc:
            failures += 1
            record.update(
                {
                    "model_answer": "",
                    "pred": "",
                    "raw_output": "",
                    "reason": "",
                    "is_correct": False,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
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

    print(f"Saved Task 3 records: {output}")
    print(f"Failures in this run: {failures}")
    return output


def _type2_mapping(category: str, labels: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    if category == "semantic":
        if len(labels) > len(string.ascii_uppercase):
            raise ValueError("Semantic Type 2 supports at most 26 categories")
        index_to_label = {
            string.ascii_uppercase[i]: label for i, label in enumerate(labels)
        }
    else:
        index_to_label = {str(i + 1): label for i, label in enumerate(labels)}
    label_to_index = {label: index for index, label in index_to_label.items()}
    return index_to_label, label_to_index


def run_type2(args: argparse.Namespace) -> Path:
    task_config = load_yaml(args.task_config)
    prompt_root = load_yaml(args.prompt_config)
    models_config = load_yaml(args.models_config).get("models", {})
    subtask_config = _subtask_config(task_config, args.task_type, args.category)
    prompt_config = prompt_root[args.task_type][args.category]
    context = _resolved_context(args, subtask_config)
    assert context is not None

    model_config = _merged_model_config(
        model_alias=args.model,
        models_config=models_config,
        task_config=task_config,
        subtask_config=subtask_config,
    )
    input_path = args.input or _default_input(subtask_config)
    sheet = _sheet_for(args, task_config, subtask_config)
    output = args.output or _default_output(
        args,
        prepared=args.mode == "prepare",
        context=context,
    )

    df = load_table(input_path, sheet_name=sheet).reset_index(drop=True)
    columns = _resolve_type2_columns(df, subtask_config["column_aliases"])

    # Build the fixed label mapping from the complete benchmark table before applying --limit.
    labels = sorted(
        {
            clean_cell(value)
            for value in df[columns["label"]].tolist()  # type: ignore[index]
            if clean_cell(value)
        }
    )
    if not labels:
        raise ValueError("No semantic categories/special domains were found in the input file")

    expected_count = subtask_config.get("expected_label_count")
    if expected_count is not None and len(labels) != int(expected_count):
        print(
            f"[WARN] Expected {expected_count} labels for Task 3 Type 2 "
            f"{args.category}, but found {len(labels)} in this input file.",
            file=sys.stderr,
        )

    index_to_label, label_to_index = _type2_mapping(args.category, labels)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    done = completed_ids(output, retry_errors=args.retry_errors) if args.resume else set()
    if not args.resume and output.exists():
        output.unlink()

    backend = None if args.mode == "prepare" else create_backend(model_config)
    delay = float(task_config.get("request_delay_seconds", 0.0))
    if model_config.get("provider") == "transformers":
        delay = 0.0

    failures = 0
    skipped_missing_gold = 0
    for row_position, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Task3/type2/{args.category}/{context}/{args.model}"
    ):
        gold_label = clean_cell(row[columns["label"]])  # type: ignore[index]
        if not gold_label:
            skipped_missing_gold += 1
            continue

        record_id = _record_id(
            row_position=int(row_position),
            row=row,
            id_column=columns["id"],
            task_type=args.task_type,
            category=args.category,
            context=context,
        )
        if record_id in done:
            continue

        headword = clean_cell(row[columns["headword"]])  # type: ignore[index]
        if not headword:
            raise ValueError(f"Empty headword at row {row_position}")

        example = ""
        if context == "term_example":
            if not columns["example"]:
                raise KeyError("term_example requires a usage-example column")
            example = _fill_mask(row[columns["example"]], headword)
            if not example:
                raise ValueError(f"Empty usage example at row {row_position}")

        year = clean_cell(row[columns["year"]]) if columns["year"] else ""
        gold_index = label_to_index.get(gold_label, "")
        if not gold_index:
            raise ValueError(f"Gold label is not in the category mapping: {gold_label}")

        prompt = build_task3_type2_prompt(
            prompt_config=prompt_config,
            headword=headword,
            example=example if context == "term_example" else None,
            index_to_label=index_to_label,
        )

        record: dict[str, Any] = {
            "id": record_id,
            "row_index": int(row_position),
            "task": "task3",
            "task_type": "type2",
            "category": args.category,
            "context": context,
            "model": args.model,
            "model_id": model_config["model_id"],
            "provider": model_config["provider"],
            "headword": headword,
            "example": example,
            "year": year,
            "gold": gold_label,
            "gold_index": gold_index,
            "label_mapping": index_to_label,
        }

        try:
            if args.mode == "prepare":
                record.update(
                    {
                        "system_prompt": prompt.system_prompt,
                        "user_prompt": prompt.user_prompt,
                        "pred": "",
                        "pred_index": "",
                        "model_answer": "",
                        "raw_output": "",
                        "is_correct": False,
                        "status": "prepared",
                        "error": "",
                    }
                )
            else:
                result = backend.generate(prompt.system_prompt, prompt.user_prompt)  # type: ignore[union-attr]
                pred_index = parse_task3_type2_output(
                    result.raw_output,
                    valid_indices=index_to_label.keys(),
                )
                pred_label = index_to_label.get(pred_index, "")
                record.update(
                    {
                        "pred": pred_label,
                        "pred_index": pred_index,
                        "model_answer": pred_index,
                        "raw_output": result.raw_output,
                        "is_correct": pred_label == gold_label,
                        "status": "ok",
                        "error": "",
                    }
                )
        except Exception as exc:
            failures += 1
            record.update(
                {
                    "pred": "",
                    "pred_index": "",
                    "model_answer": "",
                    "raw_output": "",
                    "is_correct": False,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
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

    print(f"Saved Task 3 records: {output}")
    print(f"Failures in this run: {failures}")
    print(f"Rows skipped because gold label was missing: {skipped_missing_gold}")
    return output


def evaluate_type1(args: argparse.Namespace, prediction_path: Path | None = None) -> dict[str, Any]:
    prediction_path = prediction_path or args.output or _default_output(args)
    metrics_path = args.metrics_output or prediction_path.with_suffix(".metrics.json")
    errors_path = args.errors_output or prediction_path.with_suffix(".errors.jsonl")
    metrics = evaluate_task3_type1_file(
        prediction_path,
        metrics_path=metrics_path,
        errors_path=errors_path,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved error records: {errors_path}")
    return metrics


def evaluate_type2(args: argparse.Namespace, prediction_path: Path | None = None) -> dict[str, Any]:
    task_config = load_yaml(args.task_config)
    subtask_config = _subtask_config(task_config, args.task_type, args.category)
    context = _resolved_context(args, subtask_config)
    prediction_path = prediction_path or args.output or _default_output(args, context=context)
    metrics_path = args.metrics_output or prediction_path.with_suffix(".metrics.json")
    errors_path = args.errors_output or prediction_path.with_suffix(".errors.jsonl")
    metrics = evaluate_task3_type2_file(
        prediction_path,
        metrics_path=metrics_path,
        errors_path=errors_path,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved error records: {errors_path}")
    return metrics


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    task_config = load_yaml(args.task_config)
    _subtask_config(task_config, args.task_type, args.category)

    if args.task_type == "type1":
        if args.context is not None:
            raise ValueError("--context is only used by Task 3 Type 2")
        if args.mode in {"run", "prepare"}:
            run_type1(args)
        elif args.mode == "evaluate":
            evaluate_type1(args)
        else:
            predictions = run_type1(args)
            evaluate_type1(args, predictions)
        return

    if args.mode in {"run", "prepare"}:
        run_type2(args)
    elif args.mode == "evaluate":
        evaluate_type2(args)
    else:
        predictions = run_type2(args)
        evaluate_type2(args, predictions)


if __name__ == "__main__":
    main()
