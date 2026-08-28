from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


class GoldContentWord(BaseModel):
    word: str
    in_pred: bool


class DefinitionJudgeRaw(BaseModel):
    semantic_equivalence: int = Field(..., ge=0, le=2)
    coverage: int = Field(..., ge=0, le=2)
    gold_content_words: list[GoldContentWord]

    gold_has_pragmatic: bool
    pred_has_pragmatic: bool
    polarity_match: bool
    pragmatic_equivalence: int = Field(..., ge=0, le=2)

    conciseness: int = Field(..., ge=0, le=1)
    non_circularity: int = Field(..., ge=0, le=1)
    lexicographic_convention: int = Field(..., ge=0, le=1)

    factuality: int = Field(..., ge=0, le=1)
    error_types: list[str] = Field(default_factory=list)
    comment: str
    confidence: float = Field(..., ge=0.0, le=1.0)




class MalformedJudgeOutputError(ValueError):
    def __init__(self, *, raw_output: str, cause: Exception):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.raw_output = raw_output
        self.cause = cause


@dataclass(frozen=True)
class JudgeRuntimeConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    thinking_budget: Optional[int] = 1024
    max_output_tokens: int = 4096
    api_retries: int = 4
    request_delay_seconds: float = 0.2
    api_key_env: str = "GEMINI_API_KEY"


@dataclass(frozen=True)
class JudgeResponse:
    scores: dict[str, Any]
    raw_output: str


def model_dump_compat(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()  # pragma: no cover - pydantic v1 fallback


def compute_scores(raw: DefinitionJudgeRaw | dict[str, Any]) -> dict[str, Any]:
    validated = raw if isinstance(raw, DefinitionJudgeRaw) else DefinitionJudgeRaw(**raw)
    data = model_dump_compat(validated)

    semantic_adequacy = (
        data["semantic_equivalence"]
        + data["coverage"]
        + data["pragmatic_equivalence"]
    )
    fluency = (
        data["conciseness"]
        + data["non_circularity"]
        + data["lexicographic_convention"]
    )
    total = semantic_adequacy + fluency + data["factuality"]

    data["semantic_adequacy"] = semantic_adequacy
    data["fluency"] = fluency
    data["total"] = total
    data["total_without_pragmatic"] = total - data["pragmatic_equivalence"]
    return data


def parse_judge_json(raw_text: str) -> dict[str, Any]:
    """Strictly parse one JSON object. Malformed judge output is not regenerated."""
    if not raw_text or not raw_text.strip():
        raise ValueError("empty_judge_output")
    data = json.loads(raw_text.strip())
    if not isinstance(data, dict):
        raise TypeError("judge_output_must_be_a_json_object")
    return compute_scores(data)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(row)
    return rows


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_table(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".jsonl":
        return pd.DataFrame(load_jsonl(input_path))
    raise ValueError(f"Unsupported table format: {input_path}")


def find_column(
    columns: Sequence[str], aliases: Sequence[str], *, required: bool = True
) -> Optional[str]:
    exact = {str(column): str(column) for column in columns}
    normalized = {re.sub(r"\s+", "", str(column)).lower(): str(column) for column in columns}

    for alias in aliases:
        if alias in exact:
            return exact[alias]
        key = re.sub(r"\s+", "", str(alias)).lower()
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"None of the column aliases were found: {list(aliases)}. "
            f"Existing columns: {list(columns)}"
        )
    return None


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _ensure_gold_identifiers(
    gold_df: pd.DataFrame,
    id_aliases: Sequence[str],
) -> tuple[pd.DataFrame, Optional[str]]:
    gold = gold_df.reset_index(drop=True).copy()
    gold["__row_index"] = gold.index.astype(int)
    id_column = find_column(gold.columns, id_aliases, required=False)
    if id_column:
        gold["__item_id"] = gold[id_column].map(clean_text)
    else:
        gold["__item_id"] = ""
    return gold, id_column


def merge_gold_predictions(
    *,
    gold_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    column_aliases: dict[str, list[str]],
) -> tuple[pd.DataFrame, str, dict[str, Optional[str]]]:
    """
    Merge by stable item ID when both sides have overlapping IDs; otherwise use row_index.
    Returns merged rows, selected merge key, and resolved source columns.
    """
    aliases = column_aliases
    gold, gold_id_column = _ensure_gold_identifiers(gold_df, aliases.get("id", []))
    pred = prediction_df.copy()

    term_column = find_column(gold.columns, aliases["term"])
    gold_definition_column = find_column(gold.columns, aliases["gold_definition"])
    example_column = find_column(gold.columns, aliases.get("example", []), required=False)
    gold["__term_gold"] = gold[term_column]
    gold["__gold_definition"] = gold[gold_definition_column]
    gold["__example_gold"] = gold[example_column] if example_column else None

    pred_id_column = find_column(pred.columns, aliases.get("prediction_id", ["id"]), required=False)
    pred_row_column = find_column(
        pred.columns, aliases.get("prediction_row_index", ["row_index"]), required=False
    )
    pred_definition_column = find_column(
        pred.columns, aliases.get("prediction_definition", ["definition"])
    )
    pred["__prediction_definition"] = pred[pred_definition_column]

    merge_key = "row_index"
    if gold_id_column and pred_id_column:
        gold_ids = set(gold["__item_id"].astype(str)) - {""}
        pred["__item_id"] = pred[pred_id_column].map(clean_text)
        pred_ids = set(pred["__item_id"].astype(str)) - {""}
        if gold_ids.intersection(pred_ids):
            merge_key = "id"

    if merge_key == "id":
        if gold["__item_id"].duplicated().any():
            raise ValueError("Gold item IDs must be unique.")
        if pred["__item_id"].duplicated().any():
            raise ValueError("Prediction item IDs must be unique.")
        merged = gold.merge(pred, on="__item_id", how="inner", suffixes=("_gold", "_pred"))
        merged["item_key"] = merged["__item_id"].astype(str)
    else:
        if pred_row_column is None:
            raise KeyError(
                "Could not merge by ID and prediction file has no row_index column."
            )
        pred["__row_index"] = pd.to_numeric(pred[pred_row_column], errors="raise").astype(int)
        if pred["__row_index"].duplicated().any():
            raise ValueError("Prediction row_index values must be unique.")
        merged = gold.merge(pred, on="__row_index", how="inner", suffixes=("_gold", "_pred"))
        merged["item_key"] = merged["__row_index"].astype(str)

    resolved = {
        "gold_id": gold_id_column,
        "term": "__term_gold",
        "gold_definition": "__gold_definition",
        "example": "__example_gold" if example_column else None,
        "prediction_id": pred_id_column,
        "prediction_row_index": pred_row_column,
        "prediction_definition": "__prediction_definition",
    }
    return merged, merge_key, resolved


class GeminiDefinitionJudge:
    def __init__(self, *, system_prompt: str, config: JudgeRuntimeConfig):
        from google import genai

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise EnvironmentError(f"{config.api_key_env} is not set.")
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = system_prompt
        self.config = config

    def _generate_once(self, user_payload: dict[str, Any]) -> str:
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "response_mime_type": "application/json",
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.config.thinking_budget
            )

        try:
            generation_config = types.GenerateContentConfig(**config_kwargs)
        except TypeError:
            config_kwargs.pop("thinking_config", None)
            generation_config = types.GenerateContentConfig(**config_kwargs)

        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": self.system_prompt},
                    {
                        "text": "\n\n입력(JSON):\n"
                        + json.dumps(user_payload, ensure_ascii=False)
                    },
                ],
            }
        ]

        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=generation_config,
            )
        except TypeError:
            # SDK/model compatibility fallback only; this is not output repair.
            config_kwargs.pop("thinking_config", None)
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        return response.text or ""

    def _call_with_api_backoff(self, user_payload: dict[str, Any]) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.config.api_retries):
            try:
                return self._generate_once(user_payload)
            except Exception as exc:
                last_error = exc
                if attempt == self.config.api_retries - 1:
                    raise
                time.sleep((2**attempt) + random.random())
        raise RuntimeError(last_error)

    def judge(
        self,
        *,
        term: str,
        gold_definition: str,
        predicted_definition: str,
        example: Optional[str],
    ) -> JudgeResponse:
        payload = {
            "term": term,
            "example": example,
            "gold_definition": gold_definition,
            "pred_definition": predicted_definition,
        }
        raw_output = self._call_with_api_backoff(payload)
        try:
            scores = parse_judge_json(raw_output)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise MalformedJudgeOutputError(raw_output=raw_output, cause=exc) from exc
        return JudgeResponse(scores=scores, raw_output=raw_output)


def failed_result(
    *,
    reason: str,
    comment: str,
) -> dict[str, Any]:
    return {
        "judge_status": "error",
        "judge_error": reason,
        "semantic_equivalence": None,
        "coverage": None,
        "gold_content_words": None,
        "gold_has_pragmatic": None,
        "pred_has_pragmatic": None,
        "polarity_match": None,
        "pragmatic_equivalence": None,
        "conciseness": None,
        "non_circularity": None,
        "lexicographic_convention": None,
        "factuality": None,
        "semantic_adequacy": None,
        "fluency": None,
        "total": None,
        "total_without_pragmatic": None,
        "error_types": [reason],
        "comment": comment,
        "confidence": 0.0,
        "judge_raw_output": "",
    }


def summarize_judgments(rows: pd.DataFrame | list[dict[str, Any]]) -> dict[str, Any]:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return {"n_total": 0, "n_scored": 0, "n_failed": 0}

    total_numeric = pd.to_numeric(frame.get("total"), errors="coerce")
    scored = frame.loc[total_numeric.notna()].copy()
    failed = frame.loc[total_numeric.isna()].copy()

    summary: dict[str, Any] = {
        "n_total": int(len(frame)),
        "n_scored": int(len(scored)),
        "n_failed": int(len(failed)),
    }

    score_columns = [
        "semantic_equivalence",
        "coverage",
        "pragmatic_equivalence",
        "semantic_adequacy",
        "conciseness",
        "non_circularity",
        "lexicographic_convention",
        "fluency",
        "factuality",
        "total",
        "total_without_pragmatic",
        "confidence",
    ]
    for column in score_columns:
        if column in scored.columns and len(scored):
            values = pd.to_numeric(scored[column], errors="coerce")
            summary[f"mean_{column}"] = (
                float(values.mean()) if values.notna().any() else None
            )

    if "total" in scored.columns and len(scored):
        distribution = (
            pd.to_numeric(scored["total"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        summary["total_distribution"] = {
            str(int(key)): int(value) for key, value in distribution.items()
        }

    if "error_types" in scored.columns and len(scored):
        errors = scored["error_types"].explode().dropna().astype(str)
        summary["top_error_types"] = {
            key: int(value) for key, value in errors.value_counts().head(10).items()
        }

    if "judge_error" in failed.columns and len(failed):
        failure_types = failed["judge_error"].fillna("unknown").astype(str).value_counts()
        summary["failure_types"] = {
            key: int(value) for key, value in failure_types.items()
        }
    return summary


def jsonify_dataframe_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        output[column] = output[column].map(
            lambda value: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (list, dict))
            else value
        )
    return output


def save_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
