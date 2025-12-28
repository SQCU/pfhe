"""
ICR Text Template Example Generator

Generates training examples that teach models to follow text formatting rules
described in-context. Each example includes:
1. Format specification (natural language)
2. Optional method library (domain knowledge)
3. Problem statement
4. Response following the format
"""

from dataclasses import dataclass, field
from typing import Optional
import random

from .format_specs import FORMAT_REGISTRY, FormatSpec, get_format_spec


@dataclass
class TemplateExample:
    """A single text template training example."""

    example_id: str
    format_name: str

    # Components
    format_spec: str  # Natural language format description
    method_library: Optional[str]  # Domain knowledge (e.g., math methods)
    problem: str  # The problem/question
    response: str  # Response following the format

    # Combined text for training
    full_text: str

    # Verification
    format_verified: bool = False

    # Metadata
    source_benchmark: Optional[str] = None
    source_problem_id: Optional[str] = None


class TemplateExampleGenerator:
    """Generate ICR examples with text template specifications."""

    def __init__(
        self,
        formats: Optional[list[str]] = None,
        include_format_spec: bool = True,
        include_method_library: bool = True,
    ):
        """
        Initialize generator.

        Args:
            formats: List of format names to use (default: all)
            include_format_spec: Whether to prepend format specifications
            include_method_library: Whether to include domain method libraries
        """
        self.formats = formats or list(FORMAT_REGISTRY.keys())
        self.include_format_spec = include_format_spec
        self.include_method_library = include_method_library

    def generate(
        self,
        problem_id: str,
        problem_text: str,
        answer_text: str,
        format_name: Optional[str] = None,
        method_library: Optional[str] = None,
        benchmark: Optional[str] = None,
    ) -> TemplateExample:
        """
        Generate a single template example.

        Args:
            problem_id: Unique identifier for the problem
            problem_text: The problem/question text
            answer_text: The answer/response text
            format_name: Specific format to use (default: random)
            method_library: Optional domain knowledge to include
            benchmark: Source benchmark name

        Returns:
            TemplateExample with format-wrapped content
        """
        # Select format
        if format_name is None:
            format_name = random.choice(self.formats)

        spec = get_format_spec(format_name)
        if spec is None:
            raise ValueError(f"Unknown format: {format_name}")

        # Format the problem and response according to the template
        formatted_problem, formatted_response = self._apply_format(
            spec, problem_text, answer_text
        )

        # Build full text
        components = []

        if self.include_format_spec:
            components.append(f"# Format Specification\n\n{spec.explanation}")

        if self.include_method_library and method_library:
            components.append(f"# Method Library\n\n{method_library}")

        components.append(formatted_problem)
        components.append(formatted_response)

        full_text = "\n\n---\n\n".join(components)

        # Verify format
        verified = False
        if spec.verifier:
            try:
                verified = spec.verifier(formatted_response)
            except Exception:
                verified = False

        return TemplateExample(
            example_id=f"{problem_id}_{format_name}",
            format_name=format_name,
            format_spec=spec.explanation if self.include_format_spec else "",
            method_library=method_library,
            problem=formatted_problem,
            response=formatted_response,
            full_text=full_text,
            format_verified=verified,
            source_benchmark=benchmark,
            source_problem_id=problem_id,
        )

    def generate_variants(
        self,
        problem_id: str,
        problem_text: str,
        answer_text: str,
        method_library: Optional[str] = None,
        benchmark: Optional[str] = None,
        formats: Optional[list[str]] = None,
    ) -> list[TemplateExample]:
        """
        Generate examples in multiple formats for the same problem.

        Args:
            problem_id: Unique identifier
            problem_text: The problem text
            answer_text: The answer text
            method_library: Optional domain knowledge
            benchmark: Source benchmark
            formats: Specific formats to use (default: all configured)

        Returns:
            List of TemplateExamples, one per format
        """
        formats = formats or self.formats
        examples = []

        for fmt in formats:
            try:
                example = self.generate(
                    problem_id=problem_id,
                    problem_text=problem_text,
                    answer_text=answer_text,
                    format_name=fmt,
                    method_library=method_library,
                    benchmark=benchmark,
                )
                examples.append(example)
            except Exception as e:
                # Log but continue with other formats
                print(f"Warning: Failed to generate {fmt} variant: {e}")

        return examples

    def _apply_format(
        self,
        spec: FormatSpec,
        problem: str,
        answer: str,
    ) -> tuple[str, str]:
        """
        Apply a format specification to problem and answer.

        Returns:
            (formatted_problem, formatted_response)
        """
        name = spec.name

        if name == "qa_simple":
            return self._format_qa_simple(problem, answer)
        elif name == "bracket_speaker":
            return self._format_bracket_speaker(problem, answer)
        elif name == "markerdown":
            return self._format_markerdown(problem, answer)
        elif name == "indentation_scope":
            return self._format_indentation_scope(problem, answer)
        elif name == "markdown":
            return self._format_markdown(problem, answer)
        elif name == "json":
            return self._format_json(problem, answer)
        elif name == "yaml":
            return self._format_yaml(problem, answer)
        else:
            # Default: simple concatenation
            return f"Problem:\n{problem}", f"Answer:\n{answer}"

    def _format_qa_simple(self, problem: str, answer: str) -> tuple[str, str]:
        """Format as Q: / A: with calculation notation."""
        formatted_problem = f"Q: {problem}"

        # Try to extract final answer for #### notation
        lines = answer.strip().split('\n')
        formatted_answer = f"A: {answer}"

        # Add #### if there's a clear numeric answer
        import re
        numbers = re.findall(r'[\d,]+(?:\.\d+)?', answer)
        if numbers:
            last_num = numbers[-1].replace(',', '')
            if not answer.strip().endswith(f"#### {last_num}"):
                formatted_answer += f"\n\n#### {last_num}"

        return formatted_problem, formatted_answer

    def _format_bracket_speaker(self, problem: str, answer: str) -> tuple[str, str]:
        """Format as [SPEAKER: content] blocks."""
        formatted_problem = f"[STUDENT: {problem}]"
        formatted_answer = f"[TUTOR: {answer}]"
        return formatted_problem, formatted_answer

    def _format_markerdown(self, problem: str, answer: str) -> tuple[str, str]:
        """Format with Markerdown chromatic markers."""
        formatted_problem = f"## Problem Statement 🖍🔵\n\n{problem}"

        # Split answer into sections if it has steps
        lines = answer.strip().split('\n')
        if len(lines) > 3:
            # Try to identify solution vs final answer
            formatted_answer = f"## Solution 🖍🟢\n\n{answer}"
        else:
            formatted_answer = f"## Answer 🖍🟢\n\n{answer}"

        return formatted_problem, formatted_answer

    def _format_indentation_scope(self, problem: str, answer: str) -> tuple[str, str]:
        """Format with indentation-based scoping."""
        formatted_problem = problem  # Level 0

        # Indent the answer as level 1 content
        indented_lines = []
        for line in answer.split('\n'):
            if line.strip():
                indented_lines.append(f"  {line}")
            else:
                indented_lines.append("")

        formatted_answer = '\n'.join(indented_lines)
        return formatted_problem, formatted_answer

    def _format_markdown(self, problem: str, answer: str) -> tuple[str, str]:
        """Format as standard Markdown."""
        formatted_problem = f"### Problem\n\n{problem}"
        formatted_answer = f"### Solution\n\n{answer}"
        return formatted_problem, formatted_answer

    def _format_json(self, problem: str, answer: str) -> tuple[str, str]:
        """Format as JSON objects."""
        import json

        problem_obj = {"type": "problem", "text": problem}
        answer_obj = {"type": "answer", "text": answer}

        formatted_problem = json.dumps(problem_obj, indent=2)
        formatted_answer = json.dumps(answer_obj, indent=2)
        return formatted_problem, formatted_answer

    def _format_yaml(self, problem: str, answer: str) -> tuple[str, str]:
        """Format as YAML."""
        # Escape any YAML-problematic characters
        problem_escaped = problem.replace('\n', '\n  ')
        answer_escaped = answer.replace('\n', '\n  ')

        formatted_problem = f"problem: |\n  {problem_escaped}"
        formatted_answer = f"answer: |\n  {answer_escaped}"
        return formatted_problem, formatted_answer


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_template_example(
    problem_id: str,
    problem_text: str,
    answer_text: str,
    format_name: str = "qa_simple",
    method_library: Optional[str] = None,
) -> TemplateExample:
    """Convenience function to generate a single template example."""
    generator = TemplateExampleGenerator()
    return generator.generate(
        problem_id=problem_id,
        problem_text=problem_text,
        answer_text=answer_text,
        format_name=format_name,
        method_library=method_library,
    )


def generate_format_adherence_test(
    formats: Optional[list[str]] = None,
) -> str:
    """
    Generate a test prompt for format adherence evaluation.

    The prompt describes multiple formats and asks the model to continue
    a document using one of them. Used to test if models can follow
    in-context format specifications.

    Returns:
        Test prompt string
    """
    formats = formats or ["qa_simple", "markerdown", "bracket_speaker", "indentation_scope"]

    # Build format descriptions
    descriptions = []
    for i, fmt_name in enumerate(formats, 1):
        spec = get_format_spec(fmt_name)
        if spec:
            descriptions.append(f"### Format {i}: {fmt_name}\n\n{spec.explanation}")

    # Select one format to actually use
    active_format = random.choice(formats)
    active_spec = get_format_spec(active_format)

    prompt = f"""# Document Format Specifications

This document describes {len(formats)} different text formatting approaches.
Only ONE of these formats is used in the actual content below.

{chr(10).join(descriptions)}

---

# Content Section

The following content uses the **{active_format}** format described above.

{active_spec.example if active_spec else ""}

---

# Your Task

Continue this document with a new section about calculating compound interest.
The principal is $1000, the rate is 5% per year, and the time is 3 years.
Follow the {active_format} format exactly as demonstrated above.

# New Section

"""

    return prompt
