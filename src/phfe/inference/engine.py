"""
vLLM Inference Engine

Unified interface for local model inference via vLLM.
Handles both HuggingFace models and local GGUF files.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from . import InferenceConfig, GenerationResult

logger = logging.getLogger(__name__)

# Known local model paths (WSL mount points, etc.)
LOCAL_MODEL_PATHS = {
    "gemma-3-27b-it-q4": "/mnt/f/dox/ai/text/models/gemma-3-27b-it-q4_0.gguf",
    "gemma-3-27b-it-q4km": "/mnt/f/dox/ai/text/models/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf",
    "gemma-3-4b-it-q8": "/mnt/f/dox/ai/text/models/gemma-3-4b-it-Q8_0.gguf",
    "gemma-2-9b-it-q8": "/mnt/f/dox/ai/text/models/gemma-2-9b-it-Q8_0.gguf",
}


def resolve_model_path(model_id: str) -> str:
    """
    Resolve model identifier to actual path.

    Handles:
    - HuggingFace model IDs (passed through)
    - Local path shortcuts (from LOCAL_MODEL_PATHS)
    - Direct file paths
    """
    # Check shortcuts first
    if model_id in LOCAL_MODEL_PATHS:
        path = LOCAL_MODEL_PATHS[model_id]
        if Path(path).exists():
            logger.info(f"Using local model: {path}")
            return path
        else:
            logger.warning(f"Local model not found: {path}, falling back to HuggingFace")

    # Check if it's a direct path
    if Path(model_id).exists():
        return model_id

    # Assume it's a HuggingFace model ID
    return model_id


class VLLMEngine:
    """
    vLLM-based inference engine.

    Usage:
        engine = VLLMEngine.from_config(config)
        result = engine.generate("What is 2+2?")
        results = engine.generate_batch(["Q1", "Q2", "Q3"])
    """

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
    ):
        self.model_path = resolve_model_path(model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_model_len = max_model_len

        self._llm = None
        self._sampling_params = None

    @classmethod
    def from_config(cls, config: InferenceConfig) -> "VLLMEngine":
        """Create engine from config."""
        return cls(
            model_path=config.model_path,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
        )

    def _ensure_loaded(self):
        """Lazy-load the model."""
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM required for inference. Install with: pip install vllm"
            )

        logger.info(f"Loading model: {self.model_path}")
        start = time.perf_counter()

        llm_kwargs = {
            "model": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "trust_remote_code": True,
        }

        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**llm_kwargs)

        self._sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        elapsed = time.perf_counter() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> GenerationResult:
        """Generate completion for a single prompt."""
        results = self.generate_batch(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return results[0]

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> list[GenerationResult]:
        """Generate completions for multiple prompts."""
        self._ensure_loaded()

        from vllm import SamplingParams

        # Override sampling params if specified
        params = SamplingParams(
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p or self.top_p,
        )

        start = time.perf_counter()
        outputs = self._llm.generate(prompts, params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for prompt, output in zip(prompts, outputs):
            generated = output.outputs[0]
            results.append(GenerationResult(
                prompt=prompt,
                generated_text=generated.text,
                full_text=prompt + generated.text,
                finish_reason=generated.finish_reason or "unknown",
                tokens_generated=len(generated.token_ids),
                generation_time_ms=elapsed_ms / len(prompts),
            ))

        return results

    def evaluate_format_adherence(
        self,
        prompt: str,
        expected_format: str,
        verifier: callable,
    ) -> dict:
        """
        Generate and verify format adherence.

        Args:
            prompt: The generation prompt
            expected_format: Name of expected format
            verifier: Function to verify format compliance

        Returns:
            Dict with generation result and verification status
        """
        result = self.generate(prompt)

        try:
            adheres = verifier(result.generated_text)
        except Exception as e:
            adheres = False
            logger.warning(f"Verifier error: {e}")

        return {
            "prompt": prompt,
            "generated": result.generated_text,
            "expected_format": expected_format,
            "adheres_to_format": adheres,
            "finish_reason": result.finish_reason,
            "tokens": result.tokens_generated,
        }

    def judge_format(
        self,
        text: str,
        format_description: str,
    ) -> dict:
        """
        Use the model as a judge to evaluate format adherence.

        Args:
            text: The text to evaluate
            format_description: Description of the expected format

        Returns:
            Dict with judgment and reasoning
        """
        judge_prompt = f"""You are evaluating whether a piece of text follows a specific format.

## Format Specification
{format_description}

## Text to Evaluate
{text}

## Task
Does this text follow the specified format? Analyze step by step, then give a final verdict.

1. What are the key requirements of the format?
2. Does the text meet each requirement?
3. Final verdict: PASS or FAIL

## Analysis
"""

        result = self.generate(judge_prompt, max_tokens=1024, temperature=0.3)

        # Parse verdict from response
        response = result.generated_text.lower()
        if "final verdict: pass" in response or "verdict: pass" in response:
            verdict = "PASS"
        elif "final verdict: fail" in response or "verdict: fail" in response:
            verdict = "FAIL"
        else:
            verdict = "UNCLEAR"

        return {
            "text_evaluated": text[:200] + "..." if len(text) > 200 else text,
            "format_description": format_description[:200] + "...",
            "verdict": verdict,
            "reasoning": result.generated_text,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_generate(
    prompt: str,
    model: str = "google/gemma-3-27b-it-qat-q4_0-gguf",
    max_tokens: int = 512,
) -> str:
    """Quick one-off generation without engine management."""
    engine = VLLMEngine(
        model_path=model,
        max_tokens=max_tokens,
    )
    result = engine.generate(prompt)
    return result.generated_text


def test_format_adherence(
    format_name: str,
    model: str = "google/gemma-3-27b-it-qat-q4_0-gguf",
    n_samples: int = 3,
) -> list[dict]:
    """
    Test model's ability to follow a format specification.

    Generates examples and checks if they adhere to the format.
    """
    from phfe.icr_transform.format_specs import get_format_spec
    from phfe.icr_transform.template_generator import generate_format_adherence_test

    spec = get_format_spec(format_name)
    if not spec:
        raise ValueError(f"Unknown format: {format_name}")

    engine = VLLMEngine(model_path=model)
    results = []

    for i in range(n_samples):
        # Generate test prompt
        test_prompt = generate_format_adherence_test([format_name])

        # Generate completion
        gen_result = engine.generate(test_prompt)

        # Verify adherence
        adheres = False
        if spec.verifier:
            try:
                adheres = spec.verifier(gen_result.generated_text)
            except Exception:
                pass

        results.append({
            "sample": i + 1,
            "format": format_name,
            "adheres": adheres,
            "generated": gen_result.generated_text[:500],
        })

    return results
