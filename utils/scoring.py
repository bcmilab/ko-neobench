from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from utils.io import clean_cell, read_jsonl, write_json

VALID_CHOICES = {"A", "B", "C", "D", "E"}


def extract_choice(text: Any) -> str:
    """Extract the final A-E answer without confusing option text for the answer."""
    raw = clean_cell(text)
    if not raw:
        return ""

    upper = raw.upper()
    if upper in VALID_CHOICES:
        return upper

    lines = [line.strip() for line in upper.splitlines() if line.strip()]
    for line in reversed(lines):
        exact = re.fullmatch(r"[\(\[]?([A-E])[\)\].]?", line)
        if exact:
            return exact.group(1)
        labeled = re.search(
            r"(?:FINAL\s+ANSWER|ANSWER|OUTPUT|SELECTION|정답)\s*(?:IS|은|는)?\s*[:\-]?\s*([A-E])\b",
            line,
        )
        if labeled:
            return labeled.group(1)

    filtered = [line for line in lines if not re.match(r"^[A-E][\.)]\s+", line)]
    matches = re.findall(r"\b([A-E])\b", "\n".join(filtered))
    return matches[-1] if matches else ""


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_mcq_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = len(rows)
    valid = 0
    correct = 0
    invalid = 0
    failed = 0
    error_types: Counter[str] = Counter()
    by_year: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        gold = extract_choice(row.get("gold"))
        pred = extract_choice(row.get("model_answer"))
        status = clean_cell(row.get("status")) or "ok"
        year = clean_cell(row.get("year")) or "unknown"

        if status == "error":
            failed += 1

        by_year[year]["total"] += 1
        if pred in VALID_CHOICES:
            valid += 1
            by_year[year]["valid"] += 1
            if pred == gold:
                correct += 1
                by_year[year]["correct"] += 1
            else:
                confusion_letter = extract_choice(row.get("confusion_letter"))
                if confusion_letter and pred == confusion_letter:
                    error_types["confusion_distractor"] += 1
                else:
                    error_types["random_distractor"] += 1
        else:
            invalid += 1
            error_types["invalid_or_missing"] += 1

    per_year: dict[str, Any] = {}
    for year, counts in sorted(by_year.items()):
        per_year[year] = {
            "total": counts["total"],
            "valid": counts["valid"],
            "correct": counts["correct"],
            "valid_accuracy": _safe_rate(counts["correct"], counts["valid"]),
            "accuracy": _safe_rate(counts["correct"], counts["total"]),
        }

    return {
        "total_records": total,
        "valid_predictions": valid,
        "invalid_or_missing_predictions": invalid,
        "generation_failures": failed,
        "correct": correct,
        # Matches the repeated notebook calculation: blank predictions are excluded.
        "valid_accuracy": _safe_rate(correct, valid),
        # Recommended reproducibility metric: blank/failed predictions count as incorrect.
        "accuracy": _safe_rate(correct, total),
        "coverage_of_predictions": _safe_rate(valid, total),
        "error_types": dict(error_types),
        "per_year": per_year,
    }


def classify_record(row: dict[str, Any]) -> dict[str, Any]:
    gold = extract_choice(row.get("gold"))
    pred = extract_choice(row.get("model_answer"))
    if pred not in VALID_CHOICES:
        outcome = "invalid_or_missing"
    elif pred == gold:
        outcome = "correct"
    elif extract_choice(row.get("confusion_letter")) == pred:
        outcome = "confusion_distractor"
    else:
        outcome = "random_distractor"
    return {
        "id": row.get("id"),
        "row_index": row.get("row_index"),
        "year": row.get("year"),
        "gold": gold,
        "pred": pred,
        "outcome": outcome,
        "status": row.get("status", "ok"),
        "question": row.get("question"),
        "options": row.get("options"),
        "raw_output": row.get("raw_output", row.get("raw_model_answer", "")),
        "error": row.get("error", row.get("notes", "")),
    }


def evaluate_jsonl(
    input_path: str | Path,
    *,
    metrics_path: str | Path | None = None,
    errors_path: str | Path | None = None,
) -> dict[str, Any]:
    records = list(read_jsonl(input_path) or [])
    metrics = evaluate_mcq_records(records)
    if metrics_path is not None:
        write_json(metrics_path, metrics)
    if errors_path is not None:
        from utils.io import append_jsonl

        errors_path = Path(errors_path)
        if errors_path.exists():
            errors_path.unlink()
        for row in records:
            classified = classify_record(row)
            if classified["outcome"] != "correct":
                append_jsonl(errors_path, classified)
    return metrics


_TASK2_COMPONENT_PATTERN = re.compile(
    r"[가-힣A-Za-z0-9]+(?:\+[가-힣A-Za-z0-9]+)+"
)


def normalize_task2_answer(text: Any) -> str:
    """Extract the final '+'-joined component analysis from a Task 2 response.

    The V2 notebook requests short reasoning followed by a final line beginning with
    ``정답:``. This parser preserves the notebook's last-answer behavior while adding
    fallbacks for direct-answer and local-model outputs.
    """
    raw = clean_cell(text)
    if not raw:
        return ""

    raw = raw.replace("＋", "+")
    labeled = re.findall(
        r"(?:정답|답|answer|output)\s*:\s*([^\n\r]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if labeled:
        candidate = labeled[-1].strip()
        match = _TASK2_COMPONENT_PATTERN.search(re.sub(r"\s+", "", candidate))
        if match:
            return match.group(0)
        candidate = re.sub(r"\s+", "", candidate)
        candidate = re.sub(r"^[\"'`\[\(]+|[\"'`\]\)\.,;:]+$", "", candidate)
        return candidate

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        compact = re.sub(r"\s+", "", line)
        match = _TASK2_COMPONENT_PATTERN.search(compact)
        if match:
            return match.group(0)

    matches = _TASK2_COMPONENT_PATTERN.findall(re.sub(r"\s+", "", raw))
    if matches:
        return matches[-1]

    # Direct single-line fallback. It is retained for diagnostic completeness, but
    # single-component outputs will normally score poorly on this reconstruction task.
    if (
        len(lines) == 1
        and len(lines[0]) <= 100
        and not re.search(r"\s", lines[0])
        and re.fullmatch(r"[가-힣A-Za-z0-9]+", lines[0])
    ):
        return lines[0]
    return ""


def split_task2_components(text: Any) -> list[str]:
    normalized = normalize_task2_answer(text)
    if not normalized:
        return []
    return [component for component in normalized.split("+") if component]


def score_task2_prediction(gold: Any, prediction: Any) -> dict[str, Any]:
    """Score Task 2 with multiset overlap, matching the source notebook logic."""
    gold_components = split_task2_components(gold)
    pred_components = split_task2_components(prediction)

    remaining = Counter(pred_components)
    common = 0
    for component in gold_components:
        if remaining[component] > 0:
            common += 1
            remaining[component] -= 1

    coverage = _safe_rate(common, len(gold_components))
    precision = _safe_rate(common, len(pred_components))
    f1 = (
        2 * coverage * precision / (coverage + precision)
        if coverage + precision > 0
        else 0.0
    )
    exact_match = bool(gold_components) and gold_components == pred_components
    unordered_exact_match = bool(gold_components) and Counter(gold_components) == Counter(pred_components)

    return {
        "gold_components": gold_components,
        "pred_components": pred_components,
        "matched_components": common,
        "coverage": coverage,
        # ``recall`` is retained as a compatibility alias for the original notebooks.
        "recall": coverage,
        "precision": precision,
        "f1": f1,
        "exact_match": exact_match,
        "unordered_exact_match": unordered_exact_match,
        "valid_prediction": bool(pred_components),
    }


def evaluate_task2_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = len(rows)
    failed = 0
    valid = 0
    exact = 0
    unordered_exact = 0
    coverage_sum = 0.0
    precision_sum = 0.0
    f1_sum = 0.0
    valid_coverage_sum = 0.0
    valid_precision_sum = 0.0
    valid_f1_sum = 0.0
    by_year: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in rows:
        status = clean_cell(row.get("status")) or "ok"
        if status == "error":
            failed += 1

        gold = row.get("gold", row.get("정답", row.get("gold_components", "")))
        pred = row.get(
            "model_answer",
            row.get("prediction", row.get("예측", row.get("predicted_components", ""))),
        )
        score = score_task2_prediction(gold, pred)
        coverage_sum += score["coverage"]
        precision_sum += score["precision"]
        f1_sum += score["f1"]
        exact += int(score["exact_match"])
        unordered_exact += int(score["unordered_exact_match"])

        if score["valid_prediction"]:
            valid += 1
            valid_coverage_sum += score["coverage"]
            valid_precision_sum += score["precision"]
            valid_f1_sum += score["f1"]

        year = clean_cell(row.get("year")) or "unknown"
        bucket = by_year[year]
        bucket["total"] += 1
        bucket["coverage_sum"] += score["coverage"]
        bucket["precision_sum"] += score["precision"]
        bucket["f1_sum"] += score["f1"]
        bucket["exact"] += int(score["exact_match"])

    per_year: dict[str, Any] = {}
    for year, bucket in sorted(by_year.items()):
        year_total = int(bucket["total"])
        per_year[year] = {
            "total": year_total,
            "macro_coverage": _safe_rate(bucket["coverage_sum"], year_total),
            "macro_precision": _safe_rate(bucket["precision_sum"], year_total),
            "macro_f1": _safe_rate(bucket["f1_sum"], year_total),
            "exact_match_rate": _safe_rate(int(bucket["exact"]), year_total),
        }

    return {
        "total_records": total,
        "valid_predictions": valid,
        "invalid_or_missing_predictions": total - valid,
        "generation_failures": failed,
        "prediction_coverage": _safe_rate(valid, total),
        # Primary metrics: every benchmark item is included; failures receive zero.
        "macro_coverage": _safe_rate(coverage_sum, total),
        "macro_recall": _safe_rate(coverage_sum, total),
        "macro_precision": _safe_rate(precision_sum, total),
        "macro_f1": _safe_rate(f1_sum, total),
        "exact_matches": exact,
        "exact_match_rate": _safe_rate(exact, total),
        "unordered_exact_matches": unordered_exact,
        "unordered_exact_match_rate": _safe_rate(unordered_exact, total),
        # Diagnostic valid-only values make parsing failures visible rather than hiding them.
        "valid_only_macro_coverage": _safe_rate(valid_coverage_sum, valid),
        "valid_only_macro_precision": _safe_rate(valid_precision_sum, valid),
        "valid_only_macro_f1": _safe_rate(valid_f1_sum, valid),
        "per_year": per_year,
    }


def classify_task2_record(row: dict[str, Any]) -> dict[str, Any]:
    gold = row.get("gold", row.get("정답", row.get("gold_components", "")))
    pred = row.get(
        "model_answer",
        row.get("prediction", row.get("예측", row.get("predicted_components", ""))),
    )
    score = score_task2_prediction(gold, pred)
    if not score["valid_prediction"]:
        outcome = "invalid_or_missing"
    elif score["exact_match"]:
        outcome = "exact_match"
    elif score["coverage"] == 1.0 and score["precision"] < 1.0:
        outcome = "over_generation"
    elif score["coverage"] < 1.0 and score["precision"] == 1.0:
        outcome = "under_generation"
    elif score["unordered_exact_match"]:
        outcome = "component_order_error"
    else:
        outcome = "partial_or_incorrect"

    return {
        "id": row.get("id"),
        "row_index": row.get("row_index"),
        "year": row.get("year"),
        "term": row.get("term", row.get("신어")),
        "gold": normalize_task2_answer(gold),
        "prediction": normalize_task2_answer(pred),
        "outcome": outcome,
        "coverage": score["coverage"],
        "precision": score["precision"],
        "f1": score["f1"],
        "status": row.get("status", "ok"),
        "raw_output": row.get("raw_output", row.get("raw_response", "")),
        "error": row.get("error", ""),
    }


def evaluate_task2_file(
    input_path: str | Path,
    *,
    metrics_path: str | Path | None = None,
    errors_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate integrated JSONL outputs or legacy Task 2 CSV result files."""
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".jsonl":
        records = list(read_jsonl(input_path) or [])
    elif input_path.suffix.lower() == ".csv":
        import pandas as pd

        records = pd.read_csv(input_path, encoding="utf-8-sig").to_dict("records")
    else:
        raise ValueError("Task 2 evaluation supports .jsonl or .csv prediction files")

    metrics = evaluate_task2_records(records)
    if metrics_path is not None:
        write_json(metrics_path, metrics)
    if errors_path is not None:
        from utils.io import append_jsonl

        errors_path = Path(errors_path)
        if errors_path.exists():
            errors_path.unlink()
        for row in records:
            classified = classify_task2_record(row)
            if classified["outcome"] != "exact_match":
                append_jsonl(errors_path, classified)
    return metrics


def parse_task3_type1_output(text: Any) -> tuple[str, str]:
    """Parse the JSON answer/reason format requested by Task 3 Type 1.

    The source notebooks attempted strict JSON parsing and then recovered an A-E label from
    malformed outputs. This implementation preserves that behavior while returning empty
    strings instead of ``None`` for stable JSONL serialization.
    """
    import json

    raw = clean_cell(text)
    if not raw:
        return "", ""

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        answer = extract_choice(obj.get("answer"))
        reason = clean_cell(obj.get("reason"))
        if answer in VALID_CHOICES:
            return answer, reason

    return extract_choice(cleaned), ""


def evaluate_task3_type1_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = len(rows)
    valid = 0
    correct = 0
    failures = 0
    reasons = 0
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    by_distractor: dict[str, Counter[str]] = defaultdict(Counter)
    by_gold_position: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        gold = extract_choice(row.get("gold"))
        pred = extract_choice(
            row.get("model_answer", row.get("pred", row.get("prediction", "")))
        )
        status = clean_cell(row.get("status")) or "ok"
        if status == "error" or clean_cell(row.get("error")):
            failures += 1
        if clean_cell(row.get("reason")):
            reasons += 1

        target = clean_cell(
            row.get("target_label", row.get("target_category", row.get("target_domain", "")))
        ) or "unknown"
        distractor = clean_cell(
            row.get(
                "distractor_label",
                row.get("distractor_category", row.get("distractor_domain", "")),
            )
        ) or "unknown"

        for bucket, key in (
            (by_target, target),
            (by_distractor, distractor),
            (by_gold_position, gold or "invalid_gold"),
        ):
            bucket[key]["total"] += 1

        if pred in VALID_CHOICES:
            valid += 1
            confusion[gold or "invalid_gold"][pred] += 1
            for bucket, key in (
                (by_target, target),
                (by_distractor, distractor),
                (by_gold_position, gold or "invalid_gold"),
            ):
                bucket[key]["valid"] += 1
            if pred == gold:
                correct += 1
                for bucket, key in (
                    (by_target, target),
                    (by_distractor, distractor),
                    (by_gold_position, gold or "invalid_gold"),
                ):
                    bucket[key]["correct"] += 1

    def summarize(groups: dict[str, Counter[str]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, counts in sorted(groups.items()):
            result[name] = {
                "total": counts["total"],
                "valid": counts["valid"],
                "correct": counts["correct"],
                "valid_accuracy": _safe_rate(counts["correct"], counts["valid"]),
                "strict_accuracy": _safe_rate(counts["correct"], counts["total"]),
            }
        return result

    return {
        "total_records": total,
        "valid_predictions": valid,
        "invalid_or_missing_predictions": total - valid,
        "generation_failures": failures,
        "correct": correct,
        "valid_accuracy": _safe_rate(correct, valid),
        "strict_accuracy": _safe_rate(correct, total),
        "prediction_coverage": _safe_rate(valid, total),
        "parse_fail_rate": _safe_rate(total - valid, total),
        "error_rate": _safe_rate(failures, total),
        "reason_coverage": _safe_rate(reasons, total),
        "by_target_label": summarize(by_target),
        "by_distractor_label": summarize(by_distractor),
        "by_gold_position": summarize(by_gold_position),
        "gold_prediction_confusion": {
            gold: dict(counts) for gold, counts in sorted(confusion.items())
        },
    }


def classify_task3_type1_record(row: dict[str, Any]) -> dict[str, Any]:
    gold = extract_choice(row.get("gold"))
    pred = extract_choice(
        row.get("model_answer", row.get("pred", row.get("prediction", "")))
    )
    if pred not in VALID_CHOICES:
        outcome = "invalid_or_missing"
    elif pred == gold:
        outcome = "correct"
    else:
        outcome = "incorrect"
    return {
        "id": row.get("id"),
        "row_index": row.get("row_index", row.get("source_index")),
        "task_type": row.get("task_type", row.get("type")),
        "category": row.get("category"),
        "gold": gold,
        "pred": pred,
        "gold_term": row.get("gold_term", row.get("target_term")),
        "target_label": row.get(
            "target_label", row.get("target_category", row.get("target_domain"))
        ),
        "distractor_label": row.get(
            "distractor_label",
            row.get("distractor_category", row.get("distractor_domain")),
        ),
        "outcome": outcome,
        "status": row.get("status", "ok"),
        "options": row.get("options"),
        "reason": row.get("reason", ""),
        "raw_output": row.get("raw_output", ""),
        "error": row.get("error", ""),
    }


def evaluate_task3_type1_file(
    input_path: str | Path,
    *,
    metrics_path: str | Path | None = None,
    errors_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    if input_path.suffix.lower() != ".jsonl":
        raise ValueError("Task 3 Type 1 evaluation currently supports JSONL files")
    records = list(read_jsonl(input_path) or [])
    metrics = evaluate_task3_type1_records(records)
    if metrics_path is not None:
        write_json(metrics_path, metrics)
    if errors_path is not None:
        from utils.io import append_jsonl

        errors_path = Path(errors_path)
        if errors_path.exists():
            errors_path.unlink()
        for row in records:
            classified = classify_task3_type1_record(row)
            if classified["outcome"] != "correct":
                append_jsonl(errors_path, classified)
    return metrics


def parse_task3_type2_output(
    text: Any,
    *,
    valid_indices: Iterable[str],
) -> str:
    """Parse a Task 3 Type 2 category index using one common rule for all models."""
    raw = clean_cell(text)
    if not raw:
        return ""

    valid = {clean_cell(index).upper() for index in valid_indices if clean_cell(index)}
    if not valid:
        return ""

    cleaned = re.sub(r"^```(?:text)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    upper = cleaned.upper()

    if upper in valid:
        return upper

    lines = [line.strip() for line in upper.splitlines() if line.strip()]
    for line in reversed(lines):
        exact = re.fullmatch(r"[\(\[]?([A-Z]|\d+)[\)\].]?", line)
        if exact and exact.group(1) in valid:
            return exact.group(1)
        labeled = re.search(
            r"(?:FINAL\s+ANSWER|ANSWER|OUTPUT|SELECTION|정답)\s*(?:IS|은|는)?\s*[:\-]?\s*([A-Z]|\d+)",
            line,
        )
        if labeled and labeled.group(1) in valid:
            return labeled.group(1)

    candidates: list[str]
    if all(index.isdigit() for index in valid):
        candidates = re.findall(r"\b\d+\b", upper)
    else:
        candidates = re.findall(r"\b[A-Z]\b", upper)
    valid_candidates = [candidate for candidate in candidates if candidate in valid]
    return valid_candidates[-1] if valid_candidates else ""


def evaluate_task3_type2_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = len(rows)
    valid = 0
    correct = 0
    failures = 0
    by_gold_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_year: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        gold = clean_cell(row.get("gold"))
        pred = clean_cell(row.get("pred", row.get("prediction", "")))
        pred_index = clean_cell(row.get("pred_index", row.get("model_answer", "")))
        status = clean_cell(row.get("status")) or "ok"
        year = clean_cell(row.get("year")) or "unknown"
        gold_key = gold or "unknown"

        by_gold_label[gold_key]["total"] += 1
        by_year[year]["total"] += 1
        if status == "error" or clean_cell(row.get("error")):
            failures += 1

        is_valid = bool(pred and pred_index)
        if is_valid:
            valid += 1
            by_gold_label[gold_key]["valid"] += 1
            by_year[year]["valid"] += 1
            confusion[gold_key][pred] += 1
            if pred == gold:
                correct += 1
                by_gold_label[gold_key]["correct"] += 1
                by_year[year]["correct"] += 1

    def summarize(groups: dict[str, Counter[str]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for name, counts in sorted(groups.items()):
            output[name] = {
                "total": counts["total"],
                "valid": counts["valid"],
                "correct": counts["correct"],
                "valid_accuracy": _safe_rate(counts["correct"], counts["valid"]),
                "strict_accuracy": _safe_rate(counts["correct"], counts["total"]),
            }
        return output

    return {
        "total_records": total,
        "valid_predictions": valid,
        "invalid_or_missing_predictions": total - valid,
        "generation_failures": failures,
        "correct": correct,
        "valid_accuracy": _safe_rate(correct, valid),
        "strict_accuracy": _safe_rate(correct, total),
        "prediction_coverage": _safe_rate(valid, total),
        "parse_fail_rate": _safe_rate(total - valid, total),
        "error_rate": _safe_rate(failures, total),
        "by_gold_label": summarize(by_gold_label),
        "by_year": summarize(by_year),
        "gold_prediction_confusion": {
            gold: dict(counts) for gold, counts in sorted(confusion.items())
        },
    }


def classify_task3_type2_record(row: dict[str, Any]) -> dict[str, Any]:
    gold = clean_cell(row.get("gold"))
    pred = clean_cell(row.get("pred", row.get("prediction", "")))
    pred_index = clean_cell(row.get("pred_index", row.get("model_answer", "")))
    if not pred or not pred_index:
        outcome = "invalid_or_missing"
    elif pred == gold:
        outcome = "correct"
    else:
        outcome = "incorrect"
    return {
        "id": row.get("id"),
        "row_index": row.get("row_index", row.get("source_index")),
        "task_type": row.get("task_type", "type2"),
        "category": row.get("category"),
        "context": row.get("context"),
        "headword": row.get("headword"),
        "year": row.get("year"),
        "gold": gold,
        "gold_index": row.get("gold_index"),
        "pred": pred,
        "pred_index": pred_index,
        "outcome": outcome,
        "status": row.get("status", "ok"),
        "raw_output": row.get("raw_output", row.get("raw", "")),
        "error": row.get("error", ""),
    }


def evaluate_task3_type2_file(
    input_path: str | Path,
    *,
    metrics_path: str | Path | None = None,
    errors_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    if input_path.suffix.lower() != ".jsonl":
        raise ValueError("Task 3 Type 2 evaluation currently supports JSONL files")
    records = list(read_jsonl(input_path) or [])
    metrics = evaluate_task3_type2_records(records)
    if metrics_path is not None:
        write_json(metrics_path, metrics)
    if errors_path is not None:
        from utils.io import append_jsonl

        errors_path = Path(errors_path)
        if errors_path.exists():
            errors_path.unlink()
        for row in records:
            classified = classify_task3_type2_record(row)
            if classified["outcome"] != "correct":
                append_jsonl(errors_path, classified)
    return metrics
