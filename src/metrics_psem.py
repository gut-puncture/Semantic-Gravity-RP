"""
metrics_psem.py - Semantic Pressure Computation

This module provides:
- Surface variant enumeration for target words
- Token sequence deduplication
- Teacher-forced probability computation with KV cache
- P_sem (semantic pressure) as sum of sequence probabilities

Per specification Sections 6 and 7.
"""

from typing import List, Tuple, Set, Any, Optional

# ============================================================================
# IMPORT UTILITIES
# ============================================================================

try:
    from .utils import normalize_for_match
    from .config import CONFIG
except ImportError:
    from utils import normalize_for_match
    try:
        from config import CONFIG
    except ImportError:
        CONFIG = {}


def normalize_for_word_match(text: str) -> str:
    """Alias for normalize_for_match."""
    return normalize_for_match(text)


# ============================================================================
# SURFACE VARIANT ENUMERATION
# ============================================================================


def enumerate_surface_variants(word: str) -> List[str]:
    """
    Generate surface variants of a word for tokenization.

    Per spec Section 6.1:
    - Case variants: orig, lower, title, upper
    - Whitespace variants: none, leading, trailing, both
    - Punctuation variants: at minimum trailing period

    Args:
        word: The target word X

    Returns:
        List of all surface variant strings
    """
    # Case variants (deduplicated)
    case_variants = list({word, word.lower(), word.title(), word.upper()})

    variants = []
    for v in case_variants:
        # Whitespace variants
        whitespace_forms = [
            v,           # no extra whitespace
            f" {v}",     # leading space
            f"{v} ",     # trailing space
            f" {v} ",    # both
        ]

        for ws_form in whitespace_forms:
            # Add without punctuation
            variants.append(ws_form)
            # Add with trailing period (stripping trailing space first)
            variants.append(ws_form.rstrip() + ".")
            # Add with trailing period plus space
            variants.append(ws_form.rstrip() + ". ")

    return variants


def token_sequences_for_variants(
    word: str,
    tokenizer: Any,
) -> List[Tuple[int, ...]]:
    """
    Enumerate and deduplicate token sequences for all surface variants.

    Per spec Section 6.2:
    - Tokenize each variant (no special tokens)
    - Decode back and normalize
    - Keep only sequences whose normalized decode equals X_norm
    - Deduplicate identical token sequences

    Args:
        word: The target word X
        tokenizer: Tokenizer with encode/decode methods

    Returns:
        List of unique token sequences (as tuples)
    """
    x_norm = normalize_for_word_match(word)
    variants = enumerate_surface_variants(word)

    seen: Set[Tuple[int, ...]] = set()
    result: List[Tuple[int, ...]] = []

    for variant in variants:
        # Tokenize without special tokens
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if not token_ids:
            continue

        seq = tuple(token_ids)

        # Skip if already seen
        if seq in seen:
            continue

        # Decode back and normalize (preserve whitespace)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_norm = normalize_for_word_match(decoded)

        # Keep only if normalized decode matches X_norm
        if decoded_norm == x_norm:
            seen.add(seq)
            result.append(seq)

    return result


# ============================================================================
# SEMANTIC PRESSURE COMPUTATION
# ============================================================================


def compute_p_sem(
    model: Any,
    tokenizer: Any,
    context_ids: List[int],
    token_sequences: List[Tuple[int, ...]],
    strict: bool = True,
) -> float:
    """
    Compute semantic pressure P_sem as sum of sequence probabilities.

    Per spec Section 7.2-7.3:
    - For each sequence, compute teacher-forced probability with KV cache
    - P_sem = sum over all sequences

    Args:
        model: Language model with forward method
        tokenizer: Tokenizer (for device placement if needed)
        context_ids: Context token IDs
        token_sequences: List of target token sequences
        strict: If True (default), raise errors on invalid inputs or high failure rates.
                Per spec, P_sem unavailability should halt processing.

    Returns:
        P_sem in [0, 1] (clamped for numerical stability)

    Raises:
        ValueError: If inputs are invalid and strict=True
        RuntimeError: If PyTorch unavailable or too many sequences fail and strict=True
    """
    import logging
    logger = logging.getLogger(__name__)

    if not token_sequences:
        if strict:
            raise ValueError(
                "compute_p_sem: empty token_sequences. Cannot compute P_sem "
                "without target token sequences. Check tokenization of target word."
            )
        return 0.0

    if not context_ids:
        if strict:
            raise ValueError(
                "compute_p_sem: empty context_ids. Cannot compute P_sem "
                "without context. Check prompt tokenization."
            )
        return 0.0

    try:
        import torch
    except ImportError:
        if strict:
            raise RuntimeError(
                "compute_p_sem: PyTorch unavailable. Cannot compute P_sem."
            )
        return 0.0

    # Get device from model
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")

    p_total = 0.0
    failed_sequences = []

    for seq in token_sequences:
        if not seq:
            continue

        try:
            p_seq = _compute_sequence_prob(model, context_ids, seq, device)
            p_total += p_seq
        except Exception as e:
            failed_sequences.append((seq[:3] if len(seq) > 3 else seq, str(e)))
            logger.warning("P_sem sequence computation failed for seq=%s: %s", seq[:3], e)
            continue

    # Check if too many sequences failed (>50% is a hard fail in strict mode)
    if failed_sequences and strict:
        failure_rate = len(failed_sequences) / len(token_sequences)
        if failure_rate > 0.5:
            raise RuntimeError(
                f"compute_p_sem: {len(failed_sequences)}/{len(token_sequences)} "
                f"sequences failed ({failure_rate:.1%}). Sample errors: "
                f"{failed_sequences[:3]}"
            )

    # Clamp to [0, 1] for numerical stability
    return min(max(p_total, 0.0), 1.0)


def _compute_sequence_prob(
    model: Any,
    context_ids: List[int],
    seq: Tuple[int, ...],
    device: Any,
) -> float:
    """
    Compute P(seq|context) using teacher forcing with KV cache.

    Args:
        model: Language model
        context_ids: Context token IDs
        seq: Target sequence token IDs
        device: Torch device

    Returns:
        Probability of sequence given context
    """
    import torch

    # Initial forward pass on context
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

    # Get probability of first token in sequence
    # logits shape: [1, seq_len, vocab_size]
    # We want the logits at the last context position
    last_logits = logits[0, -1, :]  # [vocab_size]
    probs = torch.softmax(last_logits, dim=-1)

    log_prob = torch.log(probs[seq[0]] + 1e-12).item()

    # Teacher-forced continuation for remaining tokens
    for i in range(1, len(seq)):
        prev_token = torch.tensor([[seq[i - 1]]], dtype=torch.long, device=device)

        with torch.inference_mode():
            outputs = model(
                input_ids=prev_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        next_logits = logits[0, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        log_prob += torch.log(probs[seq[i]] + 1e-12).item()

    import math
    return math.exp(log_prob)


# ============================================================================
# HIGH-LEVEL API
# ============================================================================


def compute_p_sem_for_prompt(
    prompt_text: str,
    target_word: str,
    model: Any,
    tokenizer: Any,
) -> float:
    """
    Compute semantic pressure for a prompt and target word.

    Args:
        prompt_text: Full prompt text
        target_word: Target word X
        model: Language model
        tokenizer: Tokenizer

    Returns:
        P_sem value in [0, 1]
    """
    add_special_tokens = CONFIG.get("model", {}).get("add_special_tokens", False)

    # Build context IDs (consistent with generation)
    context_ids = tokenizer.encode(prompt_text, add_special_tokens=add_special_tokens)

    # Get token sequences for target
    token_sequences = token_sequences_for_variants(target_word, tokenizer)

    # Compute P_sem
    return compute_p_sem(model, tokenizer, context_ids, token_sequences)


def compute_suppression_metrics(p0: float, p1: float) -> dict:
    """
    Compute suppression metrics from baseline and negative pressures.

    Per spec Section 7.4:
    - Absolute suppression: Delta = P0 - P1
    - Relative suppression: R = (P0 - P1) / max(P0, 1e-9)
    - Log suppression: L = log(max(P0, 1e-12)) - log(max(P1, 1e-12))

    Args:
        p0: Baseline pressure
        p1: Negative instruction pressure

    Returns:
        Dict with delta, relative, log_suppression
    """
    import math

    delta = p0 - p1
    relative = delta / max(p0, 1e-9)
    log_supp = math.log(max(p0, 1e-12)) - math.log(max(p1, 1e-12))

    return {
        "delta": delta,
        "relative": relative,
        "log_suppression": log_supp,
    }


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("METRICS_PSEM SELF-TEST")
    print("=" * 60)

    # DummyTokenizer for testing
    class DummyTokenizer:
        """Minimal tokenizer for testing."""

        def __init__(self):
            # Simple mapping: each character to its ord
            pass

        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            # Simple: one token per character
            return [ord(c) for c in text]

        def decode(
            self,
            ids: List[int],
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True,
        ) -> str:
            return "".join(chr(i) for i in ids if 32 <= i < 127)

    tok = DummyTokenizer()

    # Test 1: Variant enumeration
    print("\n1. Testing enumerate_surface_variants:")
    variants = enumerate_surface_variants("space")
    print(f"   Generated {len(variants)} variants for 'space'")
    assert len(variants) > 0, "Expected at least one variant"
    assert "space" in variants, "Expected 'space' in variants"
    assert " space" in variants, "Expected ' space' in variants"
    assert "space." in variants, "Expected 'space.' in variants"
    assert "space. " in variants, "Expected 'space. ' in variants"
    assert " space. " in variants, "Expected ' space. ' in variants"
    print("   PASS: Variant enumeration works")

    # Test 2: Token sequence dedup
    print("\n2. Testing token_sequences_for_variants:")
    sequences = token_sequences_for_variants("space", tok)
    print(f"   Generated {len(sequences)} unique sequences")
    assert len(sequences) > 0, "Expected at least one sequence"

    # Check dedup: all sequences should be unique
    as_set = set(sequences)
    assert len(as_set) == len(sequences), "Sequences should be unique"
    print("   PASS: Token sequence deduplication works")

    # Test 3: compute_p_sem with empty sequences (strict=False for backward compat)
    print("\n3. Testing compute_p_sem with empty sequences:")

    class DummyModel:
        """Minimal model stub for testing."""

        def parameters(self):
            return iter([])

    dummy_model = DummyModel()
    result = compute_p_sem(dummy_model, tok, [1, 2, 3], [], strict=False)
    assert result == 0.0, f"Expected 0.0 for empty sequences, got {result}"
    print("   PASS: compute_p_sem returns 0 for empty sequences (strict=False)")

    # Test 3b: compute_p_sem with empty sequences (strict=True raises)
    print("\n3b. Testing compute_p_sem raises on empty sequences (strict=True):")
    try:
        compute_p_sem(dummy_model, tok, [1, 2, 3], [], strict=True)
        assert False, "Expected ValueError for empty sequences"
    except ValueError as e:
        print(f"   PASS: Raised ValueError: {str(e)[:60]}...")

    # Test 4: compute_p_sem with empty context (strict=False for backward compat)
    print("\n4. Testing compute_p_sem with empty context:")
    result = compute_p_sem(dummy_model, tok, [], [(1, 2, 3)], strict=False)
    assert result == 0.0, f"Expected 0.0 for empty context, got {result}"
    print("   PASS: compute_p_sem returns 0 for empty context (strict=False)")

    # Test 4b: compute_p_sem with empty context (strict=True raises)
    print("\n4b. Testing compute_p_sem raises on empty context (strict=True):")
    try:
        compute_p_sem(dummy_model, tok, [], [(1, 2, 3)], strict=True)
        assert False, "Expected ValueError for empty context"
    except ValueError as e:
        print(f"   PASS: Raised ValueError: {str(e)[:60]}...")

    # Test 5: Suppression metrics
    print("\n5. Testing suppression metrics:")
    metrics = compute_suppression_metrics(0.8, 0.2)
    assert abs(metrics["delta"] - 0.6) < 1e-9, "Delta should be 0.6"
    assert abs(metrics["relative"] - 0.75) < 1e-9, "Relative should be 0.75"
    print(f"   Metrics: {metrics}")
    print("   PASS: Suppression metrics computed correctly")

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
