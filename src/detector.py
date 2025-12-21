"""
detector.py - Correctness Engine for Semantic Gravity Experiment

This module provides:
- Word-level detection of target word in completion
- Token-level mapping using prefix incremental decode
- Error auditing for mapping failures

Per specification Section 5 and execution-plan Section 4.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

# ============================================================================
# IMPORT UTILITIES
# ============================================================================

try:
    from .utils import normalize_for_match, find_word_occurrences, compute_token_char_spans
except ImportError:
    from utils import normalize_for_match, find_word_occurrences, compute_token_char_spans


# ============================================================================
# NORMALIZATION (alias for spec compliance)
# ============================================================================


def normalize_for_word_match(text: str) -> str:
    """
    Normalize text for word-level matching.

    Alias for utils.normalize_for_match per spec Section 5.2.
    """
    return normalize_for_match(text)


# ============================================================================
# WORD DETECTION
# ============================================================================


def word_present(target: str, completion: str) -> bool:
    """
    Check if target word appears in completion as a complete word.

    Uses regex word-boundary matching (letters only) to align with
    token-span mapping.

    Args:
        target: Target word to detect
        completion: Completion text to search

    Returns:
        True if target appears as complete word
    """
    return bool(find_word_spans(completion, target))


# ============================================================================
# TOKEN CHARACTER SPANS (prefix incremental decode)
# ============================================================================


def token_char_spans(tokenizer: Any, ids: List[int]) -> List[Tuple[int, int]]:
    """
    Compute character spans for each token using prefix incremental decode.

    Per spec Section 5.3:
    - For i=1..N: decode ids[:i]
    - Token i span = (len(decode[:i-1]), len(decode[:i]))

    Args:
        tokenizer: Tokenizer with decode method
        ids: List of token IDs

    Returns:
        List of (start, end) character indices for each token
    """
    return compute_token_char_spans(ids, tokenizer)


# ============================================================================
# FIND WORD SPANS IN DECODED TEXT
# ============================================================================


def find_word_spans(decoded: str, target: str) -> List[Tuple[int, int]]:
    """
    Find all character-level occurrences of target word in decoded text.

    Per spec Section 5.4:
    - Pattern: (?i)(?<![A-Za-z])X(?![A-Za-z]) with re.escape(X)
    - Returns list of (start, end) character indices

    Args:
        decoded: Decoded text to search
        target: Target word to find

    Returns:
        List of (start, end) character indices for each match
    """
    return find_word_occurrences(target, decoded)


# ============================================================================
# MAP WORD TO TOKENS
# ============================================================================


def _is_non_letter(char: str) -> bool:
    """Check if character is not a letter."""
    return not char.isalpha()


def _verify_boundaries(decoded: str, char_start: int, char_end: int) -> bool:
    """
    Verify that word boundaries are non-letters.

    Args:
        decoded: Full decoded text
        char_start: Start character index
        char_end: End character index

    Returns:
        True if both boundaries are non-letters (or at string edge)
    """
    left_ok = char_start == 0 or _is_non_letter(decoded[char_start - 1])
    right_ok = char_end >= len(decoded) or _is_non_letter(decoded[char_end])
    return left_ok and right_ok


def _decode_token_range(tokenizer: Any, ids: List[int], start: int, end: int) -> str:
    """Decode a range of tokens."""
    return tokenizer.decode(
        ids[start:end + 1],
        clean_up_tokenization_spaces=False,
        skip_special_tokens=True,
    )


def _span_covers_match(
    tok_spans: List[Tuple[int, int]],
    t_start: int,
    t_end: int,
    char_start: int,
    char_end: int,
) -> bool:
    """Check if token span covers the character match interval."""
    if t_start < 0 or t_end >= len(tok_spans) or t_start > t_end:
        return False
    span_start = tok_spans[t_start][0]
    span_end = tok_spans[t_end][1]
    return span_start <= char_start and span_end >= char_end


def _verify_span(
    tokenizer: Any,
    token_ids: List[int],
    tok_spans: List[Tuple[int, int]],
    decoded: str,
    target: str,
    char_start: int,
    char_end: int,
    t_start: int,
    t_end: int,
) -> bool:
    """Verify a candidate token span maps to the intended character match."""
    if not _span_covers_match(tok_spans, t_start, t_end, char_start, char_end):
        return False
    if not _verify_boundaries(decoded, char_start, char_end):
        return False
    match_text = decoded[char_start:char_end]
    if match_text.lower() != target.lower():
        return False
    span_text = _decode_token_range(tokenizer, token_ids, t_start, t_end)
    if match_text.lower() not in span_text.lower():
        return False
    return True


def _map_char_matches_to_tokens(
    target: str,
    token_ids: List[int],
    tokenizer: Any,
    decoded: str,
    char_matches: List[Tuple[int, int]],
    max_span_len: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Map each character-level match to a token span; return spans + failures."""
    tok_spans = token_char_spans(tokenizer, token_ids)
    if not tok_spans:
        return [], list(char_matches)

    spans: List[Tuple[int, int]] = []
    failures: List[Tuple[int, int]] = []

    for char_start, char_end in char_matches:
        # Find minimal token range covering the match
        token_start = None
        token_end = None
        for tok_idx, (ts, te) in enumerate(tok_spans):
            if te > char_start and ts < char_end:
                if token_start is None:
                    token_start = tok_idx
                token_end = tok_idx

        if token_start is None or token_end is None:
            failures.append((char_start, char_end))
            continue

        if _verify_span(
            tokenizer,
            token_ids,
            tok_spans,
            decoded,
            target,
            char_start,
            char_end,
            token_start,
            token_end,
        ):
            spans.append((token_start, token_end))
            continue

        matched_span = None

        # Expand window by +/-1 token
        for ds in (-1, 0):
            for de in (0, 1):
                t_start = token_start + ds
                t_end = token_end + de
                if _verify_span(
                    tokenizer,
                    token_ids,
                    tok_spans,
                    decoded,
                    target,
                    char_start,
                    char_end,
                    t_start,
                    t_end,
                ):
                    matched_span = (t_start, t_end)
                    break
            if matched_span:
                break

        # Brute-force search over spans up to max_span_len
        if matched_span is None:
            max_len = min(max_span_len, len(token_ids))
            for span_len in range(1, max_len + 1):
                for start in range(0, len(token_ids) - span_len + 1):
                    end = start + span_len - 1
                    if not _span_covers_match(tok_spans, start, end, char_start, char_end):
                        continue
                    if _verify_span(
                        tokenizer,
                        token_ids,
                        tok_spans,
                        decoded,
                        target,
                        char_start,
                        char_end,
                        start,
                        end,
                    ):
                        matched_span = (start, end)
                        break
                if matched_span:
                    break

        if matched_span:
            spans.append(matched_span)
        else:
            failures.append((char_start, char_end))

    return spans, failures


def map_word_to_tokens(
    target: str,
    token_ids: List[int],
    tokenizer: Any,
    decoded: str,
    max_span_len: int = 8,
) -> List[Tuple[int, int]]:
    """
    Map word occurrences to token spans.

    Per spec Section 5.4:
    1) Find matches in decoded string
    2) Find minimal contiguous token range covering each match
    3) Verify decoded token span contains the matched substring
    4) Verify left/right boundaries are non-letters
    5) If fail, expand window by +/-1 token, retry
    6) If still fail, brute-force all contiguous spans up to length max_span_len

    Args:
        target: Target word to find
        token_ids: List of token IDs
        tokenizer: Tokenizer instance
        decoded: Full decoded text
        max_span_len: Maximum span length for brute-force search

    Returns:
        List of (token_start, token_end) for each successfully mapped occurrence
    """
    if not token_ids:
        return []

    # Find character-level matches
    char_matches = find_word_spans(decoded, target)
    if not char_matches:
        return []

    spans, _ = _map_char_matches_to_tokens(
        target=target,
        token_ids=token_ids,
        tokenizer=tokenizer,
        decoded=decoded,
        char_matches=char_matches,
        max_span_len=max_span_len,
    )
    return spans


# ============================================================================
# DETECT AND MAP (main entry point)
# ============================================================================


def detect_and_map(
    target: str,
    completion_text: str,
    token_ids: List[int],
    tokenizer: Any,
    errors_path: Optional[str] = None,
    prompt_id: Optional[str] = None,
    condition: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect target word and map to token spans.

    Per spec Section 5:
    - Returns word_present, token_spans, mapping_error
    - If mapping fails but word_present == True, set mapping_error True
    - Append error record to errors_path if provided

    Args:
        target: Target word to detect
        completion_text: Completion text to search
        token_ids: Token IDs for the completion
        tokenizer: Tokenizer instance
        errors_path: Optional path to JSONL error log

    Returns:
        Dict with:
        - word_present: bool
        - token_spans: list of (start, end) token indices
        - mapping_error: bool
    """
    # Decode full text for verification
    decoded = tokenizer.decode(
        token_ids,
        clean_up_tokenization_spaces=False,
        skip_special_tokens=True,
    )

    # Word detection and matches (on decoded to preserve offsets)
    char_matches = find_word_spans(decoded, target)
    present = bool(char_matches)

    # Token mapping
    token_spans: List[Tuple[int, int]] = []
    mapping_error = False
    failures: List[Tuple[int, int]] = []

    if present:
        token_spans, failures = _map_char_matches_to_tokens(
            target=target,
            token_ids=token_ids,
            tokenizer=tokenizer,
            decoded=decoded,
            char_matches=char_matches,
            max_span_len=8,
        )

        if failures or len(token_spans) != len(char_matches):
            mapping_error = True

    if mapping_error and errors_path:
        error_record = {
            "prompt_id": prompt_id,
            "condition": condition,
            "completion_text": completion_text,
            "target_word": target,
            "generated_token_ids": token_ids,
            "decoded_text": decoded,
            "error_reason": "mapping_failed_for_some_or_all_matches",
        }
        try:
            path = Path(errors_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error_record) + "\n")
        except Exception:
            pass  # Do not fail on logging errors

    return {
        "word_present": present,
        "token_spans": token_spans,
        "mapping_error": mapping_error,
    }


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DETECTOR SELF-TEST")
    print("=" * 60)

    # DummyTokenizer for testing
    class DummyTokenizer:
        """Minimal tokenizer for testing."""

        def __init__(self, mapping: Dict[int, str]):
            """
            Args:
                mapping: dict from token ID to string fragment
            """
            self._mapping = mapping

        def decode(
            self,
            ids: List[int],
            clean_up_tokenization_spaces: bool = False,
            skip_special_tokens: bool = True,
        ) -> str:
            return "".join(self._mapping.get(i, "") for i in ids)

    # Test 1: "space" not found in "spacetime"
    print("\n1. Testing 'space' not in 'spacetime':")
    result = word_present("space", "spacetime")
    assert result is False, f"Expected False, got {result}"
    print("   PASS: 'space' correctly not detected in 'spacetime'")

    # Test 2: "space." is detected
    print("\n2. Testing 'space.' detection:")
    result = word_present("space", "The answer is space.")
    assert result is True, f"Expected True, got {result}"
    print("   PASS: 'space' detected in 'The answer is space.'")

    # Test 3: Multi-token mapping
    print("\n3. Testing multi-token mapping:")
    # Setup: ids [1,2,3] decode to "sp" + "ace" + "."
    tok = DummyTokenizer({1: "sp", 2: "ace", 3: "."})
    decoded = tok.decode([1, 2, 3])
    assert decoded == "space.", f"Decoded should be 'space.', got {decoded}"

    # Find spans for "space"
    spans = map_word_to_tokens("space", [1, 2, 3], tok, decoded)
    # Expect token span covering tokens 0 and 1 (which form "space")
    assert len(spans) > 0, "Expected at least one span"
    span = spans[0]
    assert span == (0, 1), f"Expected (0, 1), got {span}"
    print(f"   PASS: Multi-token 'sp'+'ace' mapped to span {span}")

    # Test 4: detect_and_map integration
    print("\n4. Testing detect_and_map integration:")
    result = detect_and_map("space", "space.", [1, 2, 3], tok)
    assert result["word_present"] is True
    assert len(result["token_spans"]) > 0
    assert result["mapping_error"] is False
    print(f"   PASS: detect_and_map returned {result}")

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
