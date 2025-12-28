"""
Tutor Caller - Multi-model API wrapper with logit capture.

Wraps API/inference calls to tutor models with:
- Automatic model selection and load balancing
- Full request/response logging
- Logit capture for GKD distillation
- JSON parsing with error recovery
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .observability import (
    SparseLogits,
    SubagentLog,
    Timer,
    TutorType,
    hash_prompt,
)

# Model identifiers for API calls
MODEL_MAP = {
    # API models
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-3-5-haiku-20241022",
    # Open weight models (vLLM identifiers)
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "kimi-k2": "moonshotai/Kimi-K2-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "gemma-27b": "google/gemma-2-27b-it",
}

# Default model assignments per tutor type
TUTOR_MODELS = {
    TutorType.GPT_4O: "gpt-4o",
    TutorType.CLAUDE_SONNET: "claude-sonnet",
    TutorType.DEEPSEEK_R1: "deepseek-r1",
    TutorType.KIMI_K2: "kimi-k2",
    TutorType.QWEN_72B: "qwen-72b",
    TutorType.LLAMA_70B: "llama-70b",
    TutorType.GEMMA_27B: "gemma-27b",
    TutorType.MISTRAL_LARGE: "mistral-large-latest",
}


@dataclass
class TutorConfig:
    """Configuration for a tutor caller instance."""

    # API keys (loaded from env if not provided)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # vLLM settings for local inference
    vllm_base_url: Optional[str] = None
    vllm_api_key: Optional[str] = None

    # Logit storage settings
    top_p_storage: float = 0.95  # Store top 95% probability mass
    top_k_storage: int = 100  # Or top 100 tokens, whichever is fewer

    # Paths
    prompts_dir: Path = field(default_factory=lambda: Path("claudefiles/subagents"))
    traces_dir: Path = field(default_factory=lambda: Path("traces"))


# Lazy clients
_openai_client: Any = None
_anthropic_client: Any = None


def _get_openai_client(config: TutorConfig) -> Any:
    """Lazy-load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            import openai

            _openai_client = openai.OpenAI(api_key=config.openai_api_key)
        except ImportError as e:
            raise ImportError("openai package required. Install with: uv add openai") from e
    return _openai_client


def _get_anthropic_client(config: TutorConfig) -> Any:
    """Lazy-load Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic

            _anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: uv add anthropic"
            ) from e
    return _anthropic_client


class TutorCaller:
    """
    Orchestrator for calling tutor models with logit capture.

    Handles:
    - Multiple backend types (OpenAI, Anthropic, vLLM)
    - Logit extraction and sparsification
    - Logging all calls for observability
    - JSON response parsing
    """

    def __init__(self, config: Optional[TutorConfig] = None):
        self.config = config or TutorConfig()
        self.config.traces_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_cache: dict[str, str] = {}

    def _load_prompt(self, prompt_name: str) -> str:
        """Load system prompt from file."""
        if prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]

        prompt_path = self.config.prompts_dir / prompt_name / "CLAUDE.md"
        if not prompt_path.exists():
            # Try alternate locations
            alt_path = self.config.prompts_dir / f"{prompt_name}.md"
            if alt_path.exists():
                prompt_path = alt_path
            else:
                raise FileNotFoundError(f"System prompt not found: {prompt_path}")

        prompt = prompt_path.read_text()
        self._prompt_cache[prompt_name] = prompt
        return prompt

    def call(
        self,
        tutor_type: TutorType,
        user_message: str,
        system_prompt: Optional[str] = None,
        prompt_name: Optional[str] = None,
        input_payload: Optional[dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        capture_logits: bool = True,
    ) -> SubagentLog:
        """
        Call a tutor model and return a log entry with captured logits.

        Args:
            tutor_type: Which tutor to call
            user_message: The prompt to send
            system_prompt: Direct system prompt (or use prompt_name to load)
            prompt_name: Load system prompt from file
            input_payload: Structured input data (for logging)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            capture_logits: Whether to request logprobs

        Returns:
            SubagentLog with full request/response data and logits
        """
        # Resolve model
        model_key = TUTOR_MODELS[tutor_type]
        model = MODEL_MAP.get(model_key, model_key)

        # Resolve system prompt
        if system_prompt is None and prompt_name:
            system_prompt = self._load_prompt(prompt_name)

        # Initialize log
        log = SubagentLog(
            tutor_type=tutor_type,
            model=model,
            system_prompt_hash=hash_prompt(system_prompt or ""),
            input_payload=input_payload or {},
            user_message=user_message,
        )

        # Route to appropriate backend
        with Timer() as timer:
            try:
                if model.startswith("gpt-") or "openai" in model.lower():
                    self._call_openai(log, system_prompt, user_message, max_tokens, temperature, capture_logits)
                elif "claude" in model.lower():
                    self._call_anthropic(log, system_prompt, user_message, max_tokens, temperature)
                else:
                    # Assume vLLM for open-weight models
                    self._call_vllm(log, model, system_prompt, user_message, max_tokens, temperature, capture_logits)
            except Exception as e:
                log.parse_errors.append(f"API error: {type(e).__name__}: {e!s}")

        log.latency_ms = timer.elapsed_ms

        # Try to parse JSON from response
        if log.output_raw:
            log.output_parsed, log.parse_success, errors = self._parse_json_response(
                log.output_raw
            )
            log.parse_errors.extend(errors)

        return log

    def _call_openai(
        self,
        log: SubagentLog,
        system_prompt: Optional[str],
        user_message: str,
        max_tokens: int,
        temperature: float,
        capture_logits: bool,
    ) -> None:
        """Call OpenAI API."""
        client = _get_openai_client(self.config)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        # Request logprobs if capturing
        kwargs: dict[str, Any] = {
            "model": log.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if capture_logits:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = min(20, self.config.top_k_storage)  # OpenAI limit

        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        log.output_raw = choice.message.content or ""
        log.input_tokens = response.usage.prompt_tokens
        log.output_token_count = response.usage.completion_tokens

        # Extract logprobs
        if capture_logits and choice.logprobs and choice.logprobs.content:
            for token_logprob in choice.logprobs.content:
                if token_logprob.top_logprobs:
                    token_ids = []
                    logit_values = []
                    for tlp in token_logprob.top_logprobs:
                        # OpenAI returns tokens, not IDs - we'd need tokenizer to convert
                        # For now, store as placeholder
                        token_ids.append(hash(tlp.token) % 100000)
                        logit_values.append(tlp.logprob)
                    log.output_logits.append(
                        SparseLogits(
                            token_ids=token_ids,
                            logit_values=logit_values,
                            coverage=0.0,  # Unknown without full distribution
                            method="top_k",
                            threshold=len(token_ids),
                        )
                    )

    def _call_anthropic(
        self,
        log: SubagentLog,
        system_prompt: Optional[str],
        user_message: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        """Call Anthropic API."""
        client = _get_anthropic_client(self.config)

        kwargs: dict[str, Any] = {
            "model": log.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature != 1.0:
            kwargs["temperature"] = temperature

        response = client.messages.create(**kwargs)

        log.output_raw = response.content[0].text
        log.input_tokens = response.usage.input_tokens
        log.output_token_count = response.usage.output_tokens
        # Note: Anthropic doesn't expose logprobs via API

    def _call_vllm(
        self,
        log: SubagentLog,
        model: str,
        system_prompt: Optional[str],
        user_message: str,
        max_tokens: int,
        temperature: float,
        capture_logits: bool,
    ) -> None:
        """Call vLLM server or local inference."""
        if self.config.vllm_base_url:
            # Use OpenAI-compatible API
            import httpx

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_message})

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if capture_logits:
                payload["logprobs"] = True
                payload["top_logprobs"] = self.config.top_k_storage

            headers = {}
            if self.config.vllm_api_key:
                headers["Authorization"] = f"Bearer {self.config.vllm_api_key}"

            response = httpx.post(
                f"{self.config.vllm_base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            log.output_raw = choice["message"]["content"]
            log.input_tokens = data["usage"]["prompt_tokens"]
            log.output_token_count = data["usage"]["completion_tokens"]

            # Extract logprobs if available
            if capture_logits and "logprobs" in choice:
                for token_data in choice["logprobs"].get("content", []):
                    if "top_logprobs" in token_data:
                        token_ids = [int(tlp.get("token_id", 0)) for tlp in token_data["top_logprobs"]]
                        logit_values = [tlp["logprob"] for tlp in token_data["top_logprobs"]]
                        log.output_logits.append(
                            SparseLogits(
                                token_ids=token_ids,
                                logit_values=logit_values,
                                coverage=0.0,
                                method="top_k",
                                threshold=len(token_ids),
                            )
                        )
        else:
            raise NotImplementedError(
                "Local vLLM inference not implemented. "
                "Set vllm_base_url for server-based inference."
            )

    def _parse_json_response(
        self, raw: str
    ) -> tuple[Optional[dict[str, Any]], bool, list[str]]:
        """
        Try to extract JSON from response.

        Handles:
        - Pure JSON
        - JSON in markdown code blocks
        - JSON with surrounding text
        """
        errors = []

        # Try direct parse first
        try:
            return json.loads(raw), True, []
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        code_block = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", raw)
        if code_block:
            try:
                return json.loads(code_block.group(1)), True, []
            except json.JSONDecodeError as e:
                errors.append(f"JSON in code block invalid: {e}")

        # Try finding JSON object/array anywhere
        for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
            match = re.search(pattern, raw)
            if match:
                try:
                    return json.loads(match.group()), True, []
                except json.JSONDecodeError:
                    pass

        errors.append("Could not extract valid JSON from response")
        return None, False, errors


def sparsify_logits(
    logits: np.ndarray,
    method: str = "top_p",
    threshold: float = 0.95,
) -> SparseLogits:
    """
    Convert dense logits to sparse representation.

    Args:
        logits: Dense logit array of shape (vocab_size,)
        method: "top_p" or "top_k"
        threshold: p value for top_p, or k value for top_k

    Returns:
        SparseLogits with only high-probability tokens
    """

    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    if method == "top_p":
        probs = softmax(logits)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cumsum, threshold) + 1)

        top_indices = sorted_indices[:cutoff]
        top_logits = logits[top_indices]

        return SparseLogits(
            token_ids=top_indices.astype(np.int32).tolist(),
            logit_values=top_logits.astype(np.float32).tolist(),
            coverage=float(cumsum[cutoff - 1]) if cutoff > 0 else 0.0,
            method="top_p",
            threshold=threshold,
        )

    elif method == "top_k":
        k = int(threshold)
        top_indices = np.argpartition(logits, -k)[-k:]
        top_logits = logits[top_indices]

        probs = softmax(logits)
        coverage = float(probs[top_indices].sum())

        return SparseLogits(
            token_ids=top_indices.astype(np.int32).tolist(),
            logit_values=top_logits.astype(np.float32).tolist(),
            coverage=coverage,
            method="top_k",
            threshold=threshold,
        )

    else:
        raise ValueError(f"Unknown method: {method}")
