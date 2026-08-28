from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from utils.io import clean_cell

LETTERS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class MCQPrompt:
    system_prompt: str
    user_prompt: str
    options: dict[str, str]
    gold_letter: str
    gold_text: str
    confusion_letter: str
    confusion_text: str


def build_task1_mcq(
    *,
    question: str,
    correct: str,
    confusion: str,
    random_distractors: Iterable[str],
    system_template: str,
    user_template: str,
    seed: int,
) -> MCQPrompt:
    question = clean_cell(question)
    correct = clean_cell(correct)
    confusion = clean_cell(confusion)
    random_distractors = [clean_cell(x) for x in random_distractors]

    if not question:
        raise ValueError("Question is empty")
    if not correct:
        raise ValueError("Correct answer is empty")

    candidates = [correct, confusion, *random_distractors]
    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)

    if len(unique) != 5:
        raise ValueError(f"Task 1 requires exactly 5 unique options, got {len(unique)}: {unique}")

    rng = random.Random(seed)
    rng.shuffle(unique)
    options = dict(zip(LETTERS, unique, strict=True))
    gold_letter = next(letter for letter, text in options.items() if text == correct)
    confusion_letter = next(
        (letter for letter, text in options.items() if confusion and text == confusion), ""
    )
    format_values = dict(options)
    format_values.update({f"option_{letter}": text for letter, text in options.items()})
    user_prompt = user_template.format(question=question, **format_values)
    return MCQPrompt(
        system_prompt=system_template.strip(),
        user_prompt=user_prompt.strip(),
        options=options,
        gold_letter=gold_letter,
        gold_text=correct,
        confusion_letter=confusion_letter,
        confusion_text=confusion,
    )


@dataclass(frozen=True)
class Task2Prompt:
    system_prompt: str
    user_prompt: str
    example_terms: tuple[str, ...]


def build_task2_prompt(
    *,
    prompt_config: dict,
    term: str,
    shots: int,
) -> Task2Prompt:
    """Build the Task 2 CoT prompt reconstructed from the final V2 notebook."""
    term = clean_cell(term)
    if not term:
        raise ValueError("Task 2 term is empty")

    examples = list(prompt_config.get("examples", []))
    if shots < 0:
        raise ValueError("shots must be non-negative")
    if shots > len(examples):
        raise ValueError(
            f"Task 2 provides {len(examples)} fixed examples, but shots={shots} was requested"
        )

    system_parts = [clean_cell(prompt_config.get("system_base"))]
    selected = examples[:shots]
    if selected:
        examples_text = "\n\n".join(
            (
                f"입력: {clean_cell(example.get('term'))}\n"
                f"분석: {clean_cell(example.get('reasoning'))}\n"
                f"정답: {clean_cell(example.get('answer'))}"
            )
            for example in selected
        )
        system_parts.extend(["[예시]", examples_text, clean_cell(prompt_config.get("few_shot_tail"))])
    else:
        system_parts.append(clean_cell(prompt_config.get("zero_shot_tail")))

    user_template = clean_cell(prompt_config.get("user")) or "입력: {term}"
    user_prompt = user_template.format(term=term)
    example_terms = tuple(clean_cell(example.get("term")) for example in selected)
    return Task2Prompt(
        system_prompt="\n\n".join(part for part in system_parts if part).strip(),
        user_prompt=user_prompt.strip(),
        example_terms=example_terms,
    )


@dataclass(frozen=True)
class Task3Type1Prompt:
    system_prompt: str
    user_prompt: str
    options: dict[str, str]


def build_task3_type1_prompt(
    *,
    prompt_config: dict,
    options: dict[str, str],
    category_labels: Iterable[str],
) -> Task3Type1Prompt:
    """Build a Task 3 Type 1 odd-one-out prompt.

    The benchmark options are already constructed in the released task file. This function
    does not sample categories or create experimental data; it only formats the inference
    prompt reconstructed from the notebooks.
    """
    normalized_options = {letter: clean_cell(options.get(letter)) for letter in LETTERS}
    missing = [letter for letter, value in normalized_options.items() if not value]
    if missing:
        raise ValueError(f"Task 3 Type 1 has empty options: {missing}")

    labels = sorted({clean_cell(label) for label in category_labels if clean_cell(label)})
    if not labels:
        raise ValueError("Task 3 category/domain list is empty")

    system_template = clean_cell(prompt_config.get("system"))
    user_template = clean_cell(prompt_config.get("user"))
    if not system_template or not user_template:
        raise ValueError("Task 3 prompt configuration requires system and user templates")

    system_prompt = system_template.format(
        category_count=len(labels),
        category_list=", ".join(labels),
    )
    user_prompt = user_template.format(**normalized_options)
    return Task3Type1Prompt(
        system_prompt=system_prompt.strip(),
        user_prompt=user_prompt.strip(),
        options=normalized_options,
    )


@dataclass(frozen=True)
class Task3Type2Prompt:
    system_prompt: str
    user_prompt: str
    index_to_label: dict[str, str]


def build_task3_type2_prompt(
    *,
    prompt_config: dict,
    headword: str,
    example: str | None,
    index_to_label: dict[str, str],
) -> Task3Type2Prompt:
    """Build the direct label-classification prompt used by Task 3 Type 2.

    Category/domain labels and their indices are supplied by the finalized benchmark file.
    This function performs no sampling or benchmark-data construction.
    """
    normalized_headword = clean_cell(headword)
    if not normalized_headword:
        raise ValueError("Task 3 Type 2 headword is empty")

    normalized_mapping = {
        clean_cell(index): clean_cell(label)
        for index, label in index_to_label.items()
        if clean_cell(index) and clean_cell(label)
    }
    if not normalized_mapping:
        raise ValueError("Task 3 Type 2 category/domain mapping is empty")

    system_template = clean_cell(prompt_config.get("system"))
    user_template = clean_cell(prompt_config.get("user"))
    if not system_template or not user_template:
        raise ValueError("Task 3 Type 2 prompt requires system and user templates")

    label_list = "\n".join(
        f"{index}: {label}" for index, label in normalized_mapping.items()
    )
    example_text = clean_cell(example)
    example_part = f"\n[용례] {example_text}" if example_text else ""

    user_prompt = user_template.format(
        headword=normalized_headword,
        example_part=example_part,
        label_list=label_list,
    )
    return Task3Type2Prompt(
        system_prompt=system_template.strip(),
        user_prompt=user_prompt.strip(),
        index_to_label=normalized_mapping,
    )
