from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_table(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        rows = list(read_jsonl(path))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported input format: {suffix}")


def resolve_column(
    columns: Iterable[str], aliases: Iterable[str], *, required: bool = True
) -> str | None:
    available = list(columns)
    for name in aliases:
        if name in available:
            return name
    if required:
        raise KeyError(
            f"None of the expected columns were found: {list(aliases)}. "
            f"Available columns: {available}"
        )
    return None


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if isinstance(obj, dict):
                yield obj


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def completed_ids(path: str | Path, *, retry_errors: bool = False) -> set[str]:
    done: set[str] = set()
    for record in read_jsonl(path) or []:
        record_id = clean_cell(record.get("id"))
        if not record_id:
            continue
        if retry_errors and record.get("status") != "ok":
            continue
        done.add(record_id)
    return done
