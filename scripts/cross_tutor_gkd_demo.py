#!/usr/bin/env python3
"""
Cross-Tutor GKD Demo

Demonstrates cross-tokenizer distillation with real models and real GSM1K problems.
No training - just computes JSD losses and dumps detailed per-position logs showing:
- Top-10 tokens from teacher
- Top-10 tokens from student
- Logit deltas
- Per-position JSD contribution

Usage:
    # Quick tokenizer-only test (no GPU needed)
    python scripts/cross_tutor_gkd_demo.py --tokenizers-only

    # Full demo with small models (needs GPU, ~2GB VRAM)
    python scripts/cross_tutor_gkd_demo.py

    # Custom models
    python scripts/cross_tutor_gkd_demo.py \
        --student google/gemma-3-1b-it \
        --teacher Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TokenLogit:
    """A single token with its logit/probability."""
    token_id: int
    token_str: str
    log_prob: float
    prob: float


@dataclass
class PositionComparison:
    """Comparison at a single sequence position."""
    position: int
    teacher_pos: Optional[int]  # Aligned teacher position (may differ due to tokenization)

    # Ground truth token at this position
    student_token_id: int
    student_token_str: str
    teacher_token_id: Optional[int]
    teacher_token_str: Optional[str]

    # Top-k from each model
    teacher_top_k: list[TokenLogit]
    student_top_k: list[TokenLogit]

    # Losses at this position
    jsd: float
    forward_kl: float
    reverse_kl: float

    # Delta analysis
    top_token_agrees: bool  # Does argmax match?
    teacher_rank_of_student_top: int  # Where does teacher rank student's top token?
    student_rank_of_teacher_top: int  # Where does student rank teacher's top token?


@dataclass
class SequenceAnalysis:
    """Full analysis of a query-response pair."""
    query: str
    response: str
    full_text: str

    # Tokenization
    student_tokens: int
    teacher_tokens: int

    # Aggregate losses
    mean_jsd: float
    mean_forward_kl: float
    mean_reverse_kl: float

    # Per-position breakdown
    positions: list[PositionComparison] = field(default_factory=list)

    # Agreement stats
    top_token_agreement_rate: float = 0.0
    mean_teacher_rank_of_student_top: float = 0.0


def load_models(student_name: str, teacher_name: str, device: str = "auto"):
    """Load student and teacher models."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        logger.error("Please install: pip install transformers torch")
        sys.exit(1)

    logger.info(f"Loading student tokenizer: {student_name}")
    student_tok = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)

    logger.info(f"Loading teacher tokenizer: {teacher_name}")
    teacher_tok = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)

    logger.info(f"Loading student model: {student_name}")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    logger.info(f"Loading teacher model: {teacher_name}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    return student_tok, teacher_tok, student_model, teacher_model


def get_logits(model, tokenizer, text: str, device: str = None, use_chat_template: bool = False):
    """Get full logits from a model for given text."""
    import torch

    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        # Format as chat for instruct models
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt")
    else:
        inputs = tokenizer(text, return_tensors="pt")

    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        # Use model's device
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Return logits on CPU for easier manipulation
    return outputs.logits[0].cpu()  # [seq_len, vocab_size]


def compute_jsd(p_logits, q_logits, eps: float = 1e-10):
    """
    Compute Jensen-Shannon Divergence between two distributions.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        p_logits: Log probabilities from distribution P
        q_logits: Log probabilities from distribution Q

    Returns:
        JSD value (in nats)
    """
    import torch
    import torch.nn.functional as F

    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)

    m = 0.5 * (p + q)

    # KL(P || M) = sum(p * log(p/m))
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum()
    # KL(Q || M) = sum(q * log(q/m))
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum()

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return jsd.item()


def compute_kl(p_logits, q_logits, eps: float = 1e-10):
    """Compute KL(P || Q) - how well Q approximates P."""
    import torch
    import torch.nn.functional as F

    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)

    kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum()
    return kl.item()


def get_top_k_tokens(logits, tokenizer, k: int = 10) -> list[TokenLogit]:
    """Get top-k tokens from logits."""
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)

    result = []
    for i in range(k):
        tid = top_ids[i].item()
        try:
            token_str = tokenizer.decode([tid])
        except:
            token_str = f"<{tid}>"

        result.append(TokenLogit(
            token_id=tid,
            token_str=token_str,
            log_prob=torch.log(top_probs[i]).item(),
            prob=top_probs[i].item(),
        ))

    return result


def find_rank(logits, token_id: int) -> int:
    """Find the rank of a token in the distribution (0-indexed)."""
    import torch

    sorted_ids = torch.argsort(logits, descending=True)
    ranks = (sorted_ids == token_id).nonzero(as_tuple=True)[0]
    if len(ranks) > 0:
        return ranks[0].item()
    return -1


def align_positions(teacher_tok, student_tok, text: str):
    """
    Simple character-based position alignment.
    Returns mapping from student position -> teacher position.
    """
    teacher_ids = teacher_tok.encode(text)
    student_ids = student_tok.encode(text)

    # Decode each token to get character spans
    def get_spans(tokenizer, token_ids):
        spans = []
        decoded_so_far = ""
        for tid in token_ids:
            token_str = tokenizer.decode([tid])
            start = len(decoded_so_far)
            decoded_so_far += token_str
            end = len(decoded_so_far)
            spans.append((start, end))
        return spans

    teacher_spans = get_spans(teacher_tok, teacher_ids)
    student_spans = get_spans(student_tok, student_ids)

    # Map student positions to teacher positions based on overlap
    alignment = {}
    for s_idx, (s_start, s_end) in enumerate(student_spans):
        # Find teacher token with most overlap
        best_t_idx = None
        best_overlap = 0
        for t_idx, (t_start, t_end) in enumerate(teacher_spans):
            overlap_start = max(s_start, t_start)
            overlap_end = min(s_end, t_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_t_idx = t_idx
        alignment[s_idx] = best_t_idx

    return alignment, teacher_ids, student_ids


def analyze_sequence(
    text: str,
    query: str,
    response: str,
    student_tok,
    teacher_tok,
    student_model,
    teacher_model,
    top_k: int = 10,
) -> SequenceAnalysis:
    """Analyze a single sequence with detailed per-position breakdown."""
    import torch

    # Get logits from both models
    logger.info(f"  Getting student logits...")
    student_logits = get_logits(student_model, student_tok, text)

    logger.info(f"  Getting teacher logits...")
    teacher_logits = get_logits(teacher_model, teacher_tok, text)

    # Align positions
    alignment, teacher_ids, student_ids = align_positions(teacher_tok, student_tok, text)

    positions = []
    jsds = []
    forward_kls = []
    reverse_kls = []
    agreements = []
    teacher_ranks = []

    # Analyze each position (skip last since we predict next token)
    for s_pos in range(len(student_ids) - 1):
        t_pos = alignment.get(s_pos)

        # Get the ground truth next tokens
        s_next_id = student_ids[s_pos + 1]
        s_next_str = student_tok.decode([s_next_id])

        t_next_id = None
        t_next_str = None
        if t_pos is not None and t_pos + 1 < len(teacher_ids):
            t_next_id = teacher_ids[t_pos + 1]
            t_next_str = teacher_tok.decode([t_next_id])

        # Get logits at this position (predicting next token)
        s_logits = student_logits[s_pos]

        if t_pos is not None and t_pos < teacher_logits.shape[0]:
            t_logits = teacher_logits[t_pos]

            # For cross-tokenizer comparison, we need to map vocabularies
            # Simplified: just compare the top-k analysis, compute divergences on aligned vocab
            # For true JSD, we'd need vocabulary mapping - here we use a proxy

            # Compute divergences (using student vocab space as reference)
            # This is approximate since vocabularies differ
            jsd = compute_jsd(s_logits, s_logits)  # Self-JSD = 0, placeholder
            fwd_kl = 0.0
            rev_kl = 0.0

            # For actual cross-vocab JSD, we compare distributions over shared tokens
            # Simplified proxy: compare entropy and top-token agreement
            student_top_k = get_top_k_tokens(s_logits, student_tok, top_k)
            teacher_top_k = get_top_k_tokens(t_logits, teacher_tok, top_k)

            # Check if top tokens agree (by string, not ID)
            top_agrees = student_top_k[0].token_str.strip() == teacher_top_k[0].token_str.strip()

            # Find where teacher ranks student's top token (by string match)
            student_top_str = student_top_k[0].token_str
            t_rank = -1
            for i, tt in enumerate(teacher_top_k):
                if tt.token_str.strip() == student_top_str.strip():
                    t_rank = i
                    break

            # Find where student ranks teacher's top token
            teacher_top_str = teacher_top_k[0].token_str
            s_rank = -1
            for i, st in enumerate(student_top_k):
                if st.token_str.strip() == teacher_top_str.strip():
                    s_rank = i
                    break

            # Compute a proxy JSD based on probability mass overlap
            # (tokens that appear in both top-k)
            shared_mass_s = 0.0
            shared_mass_t = 0.0
            teacher_strs = {t.token_str.strip(): t.prob for t in teacher_top_k}
            for st in student_top_k:
                if st.token_str.strip() in teacher_strs:
                    shared_mass_s += st.prob
                    shared_mass_t += teacher_strs[st.token_str.strip()]

            # Proxy JSD: 1 - geometric mean of shared masses
            if shared_mass_s > 0 and shared_mass_t > 0:
                jsd = 1.0 - (shared_mass_s * shared_mass_t) ** 0.5
            else:
                jsd = 1.0  # No overlap = max divergence

        else:
            # No alignment
            student_top_k = get_top_k_tokens(s_logits, student_tok, top_k)
            teacher_top_k = []
            jsd = float('nan')
            fwd_kl = float('nan')
            rev_kl = float('nan')
            top_agrees = False
            t_rank = -1
            s_rank = -1

        pos_comparison = PositionComparison(
            position=s_pos,
            teacher_pos=t_pos,
            student_token_id=s_next_id,
            student_token_str=s_next_str,
            teacher_token_id=t_next_id,
            teacher_token_str=t_next_str,
            teacher_top_k=teacher_top_k,
            student_top_k=student_top_k,
            jsd=jsd,
            forward_kl=fwd_kl,
            reverse_kl=rev_kl,
            top_token_agrees=top_agrees,
            teacher_rank_of_student_top=t_rank,
            student_rank_of_teacher_top=s_rank,
        )
        positions.append(pos_comparison)

        if not (jsd != jsd):  # not NaN
            jsds.append(jsd)
            agreements.append(1 if top_agrees else 0)
            if t_rank >= 0:
                teacher_ranks.append(t_rank)

    return SequenceAnalysis(
        query=query,
        response=response,
        full_text=text,
        student_tokens=len(student_ids),
        teacher_tokens=len(teacher_ids),
        mean_jsd=sum(jsds) / len(jsds) if jsds else float('nan'),
        mean_forward_kl=sum(forward_kls) / len(forward_kls) if forward_kls else float('nan'),
        mean_reverse_kl=sum(reverse_kls) / len(reverse_kls) if reverse_kls else float('nan'),
        positions=positions,
        top_token_agreement_rate=sum(agreements) / len(agreements) if agreements else 0,
        mean_teacher_rank_of_student_top=sum(teacher_ranks) / len(teacher_ranks) if teacher_ranks else float('nan'),
    )


def format_position_log(pos: PositionComparison, show_top_k: int = 10) -> str:
    """Format a single position comparison for logging."""
    lines = []
    lines.append(f"Position {pos.position} (teacher: {pos.teacher_pos})")
    lines.append(f"  Next token: student={repr(pos.student_token_str)} teacher={repr(pos.teacher_token_str)}")
    lines.append(f"  JSD: {pos.jsd:.4f} | Top agrees: {pos.top_token_agrees}")
    lines.append(f"  Teacher rank of student top: {pos.teacher_rank_of_student_top}")
    lines.append(f"  Student rank of teacher top: {pos.student_rank_of_teacher_top}")

    lines.append(f"  Teacher top-{show_top_k}:")
    for i, t in enumerate(pos.teacher_top_k[:show_top_k]):
        lines.append(f"    {i}: {repr(t.token_str):20s} p={t.prob:.4f}")

    lines.append(f"  Student top-{show_top_k}:")
    for i, t in enumerate(pos.student_top_k[:show_top_k]):
        lines.append(f"    {i}: {repr(t.token_str):20s} p={t.prob:.4f}")

    return "\n".join(lines)


STYLE_GUIDE = """You are a math tutor. Solve word problems step by step.
Show your arithmetic clearly. End your answer with #### followed by the final number.

Example:
Q: Tom has 3 apples. He buys 2 more. How many apples does Tom have?
A: Tom starts with 3 apples. He buys 2 more. 3 + 2 = 5. Tom has 5 apples.
#### 5

Now solve this problem:"""


def get_gsm1k_samples(n: int = 3, use_style_guide: bool = True):
    """Get real GSM1K samples using the session fixture approach."""
    from phfe.benchmark import BenchmarkLoader, BenchmarkType

    loader = BenchmarkLoader()
    samples = loader.load(BenchmarkType.GSM1K, limit=n)

    # Format as query-response pairs
    results = []
    for sample in samples:
        # GSM1K format: question -> answer with #### marker
        query = sample.text
        response = sample.answer
        if use_style_guide:
            query = f"{STYLE_GUIDE}\nQ: {query}"
            response = f"A: {response}"
        results.append((query, response))

    return results


def run_demo(
    student_name: str,
    teacher_name: str,
    n_samples: int = 3,
    output_path: Optional[str] = None,
    top_k: int = 10,
):
    """Run the full demo."""
    logger.info("=" * 70)
    logger.info("CROSS-TUTOR GKD DEMO")
    logger.info("=" * 70)
    logger.info(f"Student: {student_name}")
    logger.info(f"Teacher: {teacher_name}")
    logger.info(f"Samples: {n_samples} GSM1K problems")
    logger.info("")

    # Load models
    student_tok, teacher_tok, student_model, teacher_model = load_models(
        student_name, teacher_name
    )

    logger.info(f"\nStudent vocab size: {student_tok.vocab_size:,}")
    logger.info(f"Teacher vocab size: {teacher_tok.vocab_size:,}")

    # Get GSM1K samples
    logger.info(f"\nLoading {n_samples} GSM1K samples...")
    samples = get_gsm1k_samples(n_samples)

    analyses = []

    for i, (query, response) in enumerate(samples):
        logger.info(f"\n{'='*70}")
        logger.info(f"SAMPLE {i+1}/{len(samples)}")
        logger.info(f"{'='*70}")
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"Response: {response[:100]}...")

        # Combine for full sequence (query already has style guide if enabled)
        full_text = f"{query}\n{response}"

        # Analyze
        analysis = analyze_sequence(
            full_text, query, response,
            student_tok, teacher_tok,
            student_model, teacher_model,
            top_k=top_k,
        )
        analyses.append(analysis)

        # Log summary
        logger.info(f"\nSummary:")
        logger.info(f"  Student tokens: {analysis.student_tokens}")
        logger.info(f"  Teacher tokens: {analysis.teacher_tokens}")
        logger.info(f"  Mean JSD (proxy): {analysis.mean_jsd:.4f}")
        logger.info(f"  Top-token agreement: {analysis.top_token_agreement_rate:.1%}")
        logger.info(f"  Mean teacher rank of student top: {analysis.mean_teacher_rank_of_student_top:.2f}")

        # Log a few interesting positions
        logger.info(f"\nSample positions (first 5):")
        for pos in analysis.positions[:5]:
            logger.info(format_position_log(pos, show_top_k=5))
            logger.info("")

    # Overall summary
    logger.info(f"\n{'='*70}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*70}")

    mean_jsd = sum(a.mean_jsd for a in analyses) / len(analyses)
    mean_agreement = sum(a.top_token_agreement_rate for a in analyses) / len(analyses)

    logger.info(f"Mean JSD across samples: {mean_jsd:.4f}")
    logger.info(f"Mean top-token agreement: {mean_agreement:.1%}")

    # Optionally save detailed output
    if output_path:
        logger.info(f"\nSaving detailed output to {output_path}")

        # Convert to JSON-serializable format
        output_data = []
        for analysis in analyses:
            d = {
                "query": analysis.query,
                "response": analysis.response,
                "student_tokens": analysis.student_tokens,
                "teacher_tokens": analysis.teacher_tokens,
                "mean_jsd": analysis.mean_jsd,
                "top_token_agreement_rate": analysis.top_token_agreement_rate,
                "positions": [
                    {
                        "position": p.position,
                        "teacher_pos": p.teacher_pos,
                        "student_next": p.student_token_str,
                        "teacher_next": p.teacher_token_str,
                        "jsd": p.jsd,
                        "top_agrees": p.top_token_agrees,
                        "teacher_top_5": [
                            {"token": t.token_str, "prob": t.prob}
                            for t in p.teacher_top_k[:5]
                        ],
                        "student_top_5": [
                            {"token": t.token_str, "prob": t.prob}
                            for t in p.student_top_k[:5]
                        ],
                    }
                    for p in analysis.positions
                ],
            }
            output_data.append(d)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("DEMO COMPLETE")
    logger.info(f"{'='*70}")

    return analyses


def run_tokenizers_only(student_name: str, teacher_name: str):
    """Quick test with tokenizers only, no model loading."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("Please install: pip install transformers")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("TOKENIZER-ONLY TEST")
    logger.info("=" * 70)

    logger.info(f"Loading student tokenizer: {student_name}")
    student_tok = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)

    logger.info(f"Loading teacher tokenizer: {teacher_name}")
    teacher_tok = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)

    logger.info(f"\nStudent vocab size: {student_tok.vocab_size:,}")
    logger.info(f"Teacher vocab size: {teacher_tok.vocab_size:,}")

    # Test alignment on a sample
    sample_text = "Alice has 5 apples. Bob gives her 3 more. How many apples does Alice have now? Answer: 8"

    alignment, teacher_ids, student_ids = align_positions(teacher_tok, student_tok, sample_text)

    logger.info(f"\nSample: {sample_text}")
    logger.info(f"Student tokens: {len(student_ids)}")
    logger.info(f"Teacher tokens: {len(teacher_ids)}")
    logger.info(f"Aligned positions: {len([v for v in alignment.values() if v is not None])}")

    # Show tokenization difference
    logger.info(f"\nStudent tokenization:")
    for i, tid in enumerate(student_ids[:20]):
        logger.info(f"  {i}: {repr(student_tok.decode([tid]))}")

    logger.info(f"\nTeacher tokenization:")
    for i, tid in enumerate(teacher_ids[:20]):
        logger.info(f"  {i}: {repr(teacher_tok.decode([tid]))}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-tutor GKD demo with real GSM1K problems"
    )
    parser.add_argument(
        "--student",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Student model (default: Qwen2-0.5B-Instruct)"
    )
    parser.add_argument(
        "--teacher",
        default="google/gemma-2-2b-it",
        help="Teacher model (default: gemma-2-2b-it)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of GSM1K samples (default: 3)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path for detailed logs"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top tokens to compare (default: 10)"
    )
    parser.add_argument(
        "--tokenizers-only",
        action="store_true",
        help="Only test tokenizers, no model loading"
    )

    args = parser.parse_args()

    if args.tokenizers_only:
        run_tokenizers_only(args.student, args.teacher)
    else:
        run_demo(
            args.student,
            args.teacher,
            n_samples=args.n_samples,
            output_path=args.output,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
