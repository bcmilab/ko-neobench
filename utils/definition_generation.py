from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError


class NeologismDefinition(BaseModel):
    term: str
    definition: str = Field(..., description="사전식 1문장 정의")


@dataclass(frozen=True)
class DefinitionResult:
    term: str
    definition: str
    raw_output: str


HANJA_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
KANA_RE = re.compile(r"[\u3040-\u30FF]")


def definition_schema() -> dict[str, Any]:
    if hasattr(NeologismDefinition, "model_json_schema"):
        return NeologismDefinition.model_json_schema()
    return NeologismDefinition.schema()  # pragma: no cover - pydantic v1 fallback


def _dump_model(obj: BaseModel) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()  # pragma: no cover - pydantic v1 fallback


def normalize_term_plain(term: str) -> str:
    if not term:
        return term
    normalized = str(term).strip()
    normalized = re.sub(r"[\^_\-]+", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_definition_text(text: str) -> str:
    normalized = str(text).strip().strip('"').strip("'")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return re.sub(
        r"^(정의|definition)\s*[:：]\s*", "", normalized, flags=re.IGNORECASE
    )


def find_forbidden_scripts(text: str) -> str:
    chars: list[str] = []
    seen: set[str] = set()
    for char in str(text or ""):
        if (HANJA_RE.search(char) or KANA_RE.search(char)) and char not in seen:
            chars.append(char)
            seen.add(char)
    return "".join(chars)


def validate_no_forbidden_scripts(result: dict[str, Any]) -> dict[str, Any]:
    validated = dict(result)
    validated["definition"] = normalize_definition_text(validated.get("definition", ""))
    bad_chars = find_forbidden_scripts(validated["definition"])
    if bad_chars:
        raise ValueError(
            "definition_contains_forbidden_script: "
            f"{bad_chars} | definition={validated['definition']}"
        )
    return validated


def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", str(text), flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return cleaned[start:]


def _clean_json_like_text(text: str) -> str:
    cleaned = str(text).strip().replace("\x00", "")
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("‘", "'").replace("’", "'")
    cleaned = cleaned.replace("：", ":")
    return re.sub(r",\s*}", "}", cleaned)


def _salvage_definition(raw_text: str) -> str:
    text = str(raw_text).strip().replace("\x00", "")
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    patterns = [
        r'"definition"\s*:\s*"([^"]*)"',
        r"'definition'\s*:\s*'([^']*)'",
        r'"definition"\s*:\s*([^,\n}]+)',
        r"definition\s*[:：]\s*([^\n}]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"').strip("'").strip()

    text = re.sub(
        r"^\s*(정의|definition|답변|answer)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = text.strip('"').strip("'").strip()
    return text.splitlines()[0].strip() if text else ""


def robust_load_generation_json(raw_text: str, term: str) -> dict[str, Any]:
    block = extract_first_json_object(raw_text)
    if block:
        cleaned = _clean_json_like_text(block)
        for candidate in (cleaned, cleaned + "}" if not cleaned.endswith("}") else cleaned):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

    definition = _salvage_definition(raw_text)
    if not definition:
        raise json.JSONDecodeError(
            "Could not parse JSON or salvage definition", str(raw_text), 0
        )
    return {"term": term, "definition": definition}


def build_structured_payload(term: str, example: Optional[str]) -> str:
    return json.dumps({"term": term, "example": example}, ensure_ascii=False)



def build_local_user_prompt(
    term: str,
    example: Optional[str],
    *,
    forbid_scripts: bool
) -> str:
    payload = {
        "term": term,
        "example": example,
        "output_schema": definition_schema(),
    }
    if not forbid_scripts:
        return json.dumps(payload, ensure_ascii=False)


class DefinitionBackend:
    def generate(
        self, *, term: str, example: Optional[str], system_prompt: str
    ) -> DefinitionResult:
        raise NotImplementedError


class StructuredChatDefinitionBackend(DefinitionBackend):
    def __init__(self, config: dict[str, Any]):
        from openai import OpenAI

        self.provider = str(config["provider"])
        self.model_id = str(config["model_id"])
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = config.get("max_tokens")
        self.max_retries = int(config.get("max_retries", 5))
        api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"{api_key_env} is not set")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if config.get("base_url"):
            client_kwargs["base_url"] = str(config["base_url"])
        self.client = OpenAI(**client_kwargs)

    def _request(self, system_prompt: str, user_prompt: str, *, schema: bool):
        request: dict[str, Any] = {
            "model": self.model_id,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.max_tokens is not None:
            request["max_tokens"] = int(self.max_tokens)
        if schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "NeologismDef",
                    "schema": definition_schema(),
                },
            }
        return self.client.chat.completions.create(**request)

    def _call_with_backoff(self, fn):
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    raise
                time.sleep((2**attempt) + random.random())
        raise RuntimeError(last_error)

    def generate(
        self, *, term: str, example: Optional[str], system_prompt: str
    ) -> DefinitionResult:
        payload = build_structured_payload(term, example)
        raw_schema: Optional[str] = None
        raw_fallback: Optional[str] = None

        try:
            response = self._call_with_backoff(
                lambda: self._request(system_prompt, payload, schema=True)
            )
            raw_schema = response.choices[0].message.content or ""
            data = json.loads(raw_schema)
            validated = NeologismDefinition(**data)
            result = _dump_model(validated)
            return DefinitionResult(
                term=str(result["term"]),
                definition=str(result["definition"]),
                raw_output=raw_schema,
            )
        except (json.JSONDecodeError, ValidationError, KeyError):
            pass

        json_only_system = (
            system_prompt
            + "\n\nIMPORTANT: Output MUST be a single valid JSON object ONLY. "
            "No extra text, no markdown, no code fences. "
            'Schema: {"term": string, "definition": string}.'
        )
        try:
            response = self._call_with_backoff(
                lambda: self._request(json_only_system, payload, schema=False)
            )
            raw_fallback = response.choices[0].message.content or ""
            candidate = extract_first_json_object(raw_fallback) or raw_fallback
            data = json.loads(candidate)
            validated = NeologismDefinition(**data)
            result = _dump_model(validated)
            return DefinitionResult(
                term=str(result["term"]),
                definition=str(result["definition"]),
                raw_output=raw_fallback,
            )
        except (json.JSONDecodeError, ValidationError, KeyError):
            pass

        for raw in (raw_fallback, raw_schema):
            candidate = extract_first_json_object(raw or "")
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                validated = NeologismDefinition(**data)
                result = _dump_model(validated)
                return DefinitionResult(
                    term=str(result["term"]),
                    definition=str(result["definition"]),
                    raw_output=raw or "",
                )
            except Exception:
                continue
        raise RuntimeError(
            "Failed to obtain valid definition JSON. "
            f"model={self.model_id}; raw_schema={str(raw_schema)[:300]}; "
            f"raw_fallback={str(raw_fallback)[:300]}"
        )


class TransformersDefinitionBackend(DefinitionBackend):
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.config = dict(config)
        self.model_id = str(config["model_id"])
        self.max_new_tokens = int(config.get("max_new_tokens", 192))
        self.forbid_scripts = bool(config.get("forbid_scripts", False))
        self.enable_token_ban = bool(config.get("enable_token_ban", False))
        self.max_script_retries = int(config.get("max_script_retries", 5))
        self.robust_json = bool(config.get("robust_json", False))
        self.disable_thinking = bool(config.get("disable_thinking", False))
        self.is_llama = bool(config.get("llama_eos_tokens", False))

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        auth_kwargs = {"token": hf_token} if hf_token else {}
        trust_remote_code = bool(config.get("trust_remote_code", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=trust_remote_code,
            **auth_kwargs,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_name = str(config.get("torch_dtype", "bfloat16")).lower()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)
        common_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
            **auth_kwargs,
        }
        if config.get("attn_implementation"):
            common_kwargs["attn_implementation"] = str(config["attn_implementation"])

        if bool(config.get("load_in_4bit", False)):
            from transformers import BitsAndBytesConfig

            common_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **common_kwargs
        )
        self.model.eval()
        self._forbidden_token_ids: Optional[list[int]] = None

    def _forbidden_ids(self) -> list[int]:
        if self._forbidden_token_ids is not None:
            return self._forbidden_token_ids
        bad_ids: list[int] = []
        vocab_size = len(self.tokenizer)
        for token_id in range(vocab_size):
            piece = self.tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if find_forbidden_scripts(piece):
                bad_ids.append(token_id)
        self._forbidden_token_ids = sorted(set(bad_ids))
        return self._forbidden_token_ids

    def _eos_token_ids(self):
        if not self.is_llama:
            return self.tokenizer.eos_token_id
        eos_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        for token in ("<|eot_id|>", "<|end_of_text|>"):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    eos_ids.append(token_id)
            except Exception:
                pass
        eos_ids = sorted(set(eos_ids))
        return eos_ids[0] if len(eos_ids) == 1 else eos_ids

    def _generate_once(
        self,
        *,
        term: str,
        example: Optional[str],
        system_prompt: str,
        retry_note: Optional[str],
    ) -> DefinitionResult:
        from transformers import LogitsProcessor, LogitsProcessorList

        class ForbiddenScriptLogitsProcessor(LogitsProcessor):
            def __init__(self, token_ids: list[int]):
                self.token_ids = token_ids
                self._tensor = None
                self._device = None

            def __call__(self, input_ids, scores):
                if not self.token_ids:
                    return scores
                if self._tensor is None or self._device != scores.device:
                    self._tensor = self_outer.torch.tensor(
                        self.token_ids, dtype=self_outer.torch.long, device=scores.device
                    )
                    self._device = scores.device
                scores[:, self._tensor] = -float("inf")
                return scores

        self_outer = self
        user_prompt = build_local_user_prompt(
            term,
            example,
            forbid_scripts=self.forbid_scripts,
            retry_note=retry_note,
        )
        if self.config.get("append_json_instruction", False):
            system_prompt = (
                system_prompt
                + "\n\n아래 사용자 입력은 JSON이다. 반드시 단일 JSON 객체만 출력하라. "
                "키는 term, definition 두 개만 사용하라."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.disable_thinking:
            template_kwargs["enable_thinking"] = False
        try:
            rendered = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            rendered = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": pad_token_id,
            "eos_token_id": self._eos_token_ids(),
        }
        if self.enable_token_ban:
            processor = ForbiddenScriptLogitsProcessor(self._forbidden_ids())
            generation_kwargs["logits_processor"] = LogitsProcessorList([processor])

        with self.torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        raw = self.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        ).strip()

        if self.robust_json:
            data = robust_load_generation_json(raw, term)
            data["term"] = term
        else:
            block = extract_first_json_object(raw)
            if not block:
                raise RuntimeError(f"parse_failed: no JSON object found | raw={raw[:500]}")
            data = json.loads(block)

        validated = NeologismDefinition(**data)
        parsed = _dump_model(validated)
        parsed["definition"] = normalize_definition_text(parsed.get("definition", ""))
        if self.forbid_scripts:
            parsed = validate_no_forbidden_scripts(parsed)
        return DefinitionResult(
            term=str(parsed["term"]),
            definition=str(parsed["definition"]),
            raw_output=raw,
        )

    def generate(
        self, *, term: str, example: Optional[str], system_prompt: str
    ) -> DefinitionResult:
        
        return self._generate_once(
                    term=term,
                    example=example,
                    system_prompt=system_prompt
                )



def create_definition_backend(config: dict[str, Any]) -> DefinitionBackend:
    provider = str(config.get("provider", ""))
    if provider in {"openai_structured", "upstage_structured"}:
        return StructuredChatDefinitionBackend(config)
    if provider == "transformers_definition":
        return TransformersDefinitionBackend(config)
    raise ValueError(f"Unsupported Task 4 provider: {provider}")
