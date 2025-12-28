"""
PHFE - Posthumanity's First Exam

A benchmark suite and training infrastructure for measuring in-context learning
vs memorization in language models.

Core components:
- benchmark: PFE benchmark suite with canonical evals and ICR variants
- icr_transform: Transform benchmarks to In-Context Retrieval format
- answer_key_corpus: Multi-tutor answer key collection with contamination firewall
- distillation: Offline GKD training with cached tutor logits
- arxiv_pipeline: LaTeX to multimodal training data pipeline
- curriculum_generator: Synthetic problem generation with difficulty scaling
- language_evals: Language competence evaluation suite
- orchestrator: Subagent dispatch and observability infrastructure
"""

__version__ = "0.1.0"
