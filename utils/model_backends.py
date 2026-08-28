from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from utils.scoring import extract_choice


@dataclass(frozen=True)
class GenerationResult:
    parsed_answer: str
    raw_output: str


class ModelBackend:
    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        raise NotImplementedError


class OpenAIResponsesBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        temperature: float | None = None,
        max_retries: int = 5,
    ):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        # The source notebooks used the Responses API with a single prompt string.
        prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                request: dict[str, Any] = {
                    "model": self.model_id,
                    "input": prompt,
                    "max_output_tokens": self.max_new_tokens,
                }
                if self.temperature is not None:
                    request["temperature"] = self.temperature
                response = self.client.responses.create(**request)
                raw = (response.output_text or "").strip()
                return GenerationResult(extract_choice(raw), raw)
            except Exception as exc:  # API/network failures
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


class UpstageChatBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        base_url: str,
        max_retries: int = 5,
    ):
        from openai import OpenAI

        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            raise EnvironmentError("UPSTAGE_API_KEY is not set")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.max_new_tokens,
                    temperature=0.0,
                )
                raw = (response.choices[0].message.content or "").strip()
                return GenerationResult(extract_choice(raw), raw)
            except Exception as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"Upstage request failed after retries: {last_error}")


class OpenAICompatibleChatBackend(ModelBackend):
    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        api_key_env: str,
        provider_name: str,
        base_url: str | None = None,
        temperature: float | None = 0.0,
        token_parameter: str = "max_tokens",
        disable_thinking: bool = False,
        max_retries: int = 5,
    ):
        from openai import OpenAI

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"{api_key_env} is not set")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.token_parameter = token_parameter
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries
        self.provider_name = provider_name

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        last_error: Exception | None = None
        token_parameters = [self.token_parameter]
        alternate = (
            "max_tokens"
            if self.token_parameter == "max_completion_tokens"
            else "max_completion_tokens"
        )
        if alternate not in token_parameters:
            token_parameters.append(alternate)

        # The source notebooks used parameter fallbacks because GPT and OpenAI-compatible
        # endpoints do not all accept the same token/temperature arguments.
        for attempt in range(self.max_retries):
            for token_parameter in token_parameters:
                for include_temperature in (True, False):
                    try:
                        request: dict[str, Any] = {
                            "model": self.model_id,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            token_parameter: self.max_new_tokens,
                        }
                        if include_temperature and self.temperature is not None:
                            request["temperature"] = self.temperature
                        if self.disable_thinking:
                            request["extra_body"] = {
                                "chat_template_kwargs": {"enable_thinking": False}
                            }

                        response = self.client.chat.completions.create(**request)
                        message = response.choices[0].message
                        raw = (message.content or "").strip()
                        if not raw:
                            dumped = (
                                message.model_dump()
                                if hasattr(message, "model_dump")
                                else {}
                            )
                            raw = str(dumped.get("reasoning", "") or "").strip()
                        return GenerationResult(extract_choice(raw), raw)
                    except Exception as exc:
                        last_error = exc
            time.sleep(2**attempt)
        raise RuntimeError(
            f"{self.provider_name} request failed after retries: {last_error}"
        )


class TransformersBackend(ModelBackend):
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = str(config["model_id"])
        self.max_new_tokens = int(config.get("max_new_tokens", 16))
        self.max_input_tokens = int(config.get("max_input_tokens", 4096))
        self.disable_thinking = bool(config.get("disable_thinking", False))
        self.load_in_4bit = bool(config.get("load_in_4bit", False))
        trust_remote_code = bool(config.get("trust_remote_code", False))
        hf_token = os.getenv("HF_TOKEN") or None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype_name = str(config.get("torch_dtype", "auto")).lower()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(dtype_name, "auto")
        if not torch.cuda.is_available() and torch_dtype != "auto":
            torch_dtype = torch.float32

        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "token": hf_token,
        }
        if self.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit Transformers inference requires a CUDA GPU")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "bitsandbytes/quantization support is required for load_in_4bit"
                ) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
                bnb_4bit_compute_dtype=(
                    torch_dtype if torch_dtype != "auto" else torch.bfloat16
                ),
                bnb_4bit_use_double_quant=bool(
                    config.get("bnb_4bit_use_double_quant", True)
                ),
            )
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.disable_thinking:
            kwargs["enable_thinking"] = False
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self._apply_chat_template(messages)
        else:
            text = f"{system_prompt}\n\n{user_prompt}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        generated = output[0][prompt_length:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GenerationResult(extract_choice(raw), raw)


def create_backend(model_config: dict[str, Any]) -> ModelBackend:
    provider = str(model_config.get("provider", ""))
    model_id = str(model_config.get("model_id", ""))
    max_new_tokens = int(model_config.get("max_new_tokens", 16))

    if provider == "openai_responses":
        temperature = model_config.get("temperature")
        return OpenAIResponsesBackend(
            model_id,
            max_new_tokens,
            float(temperature) if temperature is not None else None,
        )
    if provider == "openai_chat":
        temperature = model_config.get("temperature", 0.0)
        return OpenAICompatibleChatBackend(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            api_key_env="OPENAI_API_KEY",
            provider_name="OpenAI",
            temperature=float(temperature) if temperature is not None else None,
            token_parameter=str(model_config.get("token_parameter", "max_tokens")),
        )
    if provider == "upstage_chat":
        return UpstageChatBackend(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            base_url=str(model_config.get("base_url", "https://api.upstage.ai/v1")),
        )
    if provider == "huggingface_chat":
        temperature = model_config.get("temperature", 0.0)
        return OpenAICompatibleChatBackend(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            api_key_env="HF_TOKEN",
            provider_name="Hugging Face router",
            base_url=str(
                model_config.get("base_url", "https://router.huggingface.co/v1")
            ),
            temperature=float(temperature) if temperature is not None else None,
            token_parameter=str(model_config.get("token_parameter", "max_tokens")),
            disable_thinking=bool(model_config.get("disable_thinking", False)),
        )
    if provider == "transformers":
        return TransformersBackend(model_config)
    raise ValueError(f"Unsupported provider: {provider}")
