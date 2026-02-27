"""
break_loader.py

Load the HotpotQA 2-step subset of Break (QDMR-high-level) and convert
each example into a ToQ compatible with the operadic_consistency core.

Each Break 2-step decomposition has the form:
    step1: "return <entity or concept>"
    step2: "return <something about> #1"

We naturalize steps by:
  1. Stripping a leading "return " prefix (case-insensitive).
  2. Replacing "#1" with "[A1]" (core placeholder convention).
  3. Capitalizing the first letter and appending "?" if absent.

The resulting ToQ has:
    Node 1 (leaf, parent=2): naturalized step 1
    Node 2 (root, parent=None): naturalized step 2 with [A1] placeholder

The original question string is preserved in BreakToQExample for use
by the collapser (full-collapse plan) and for evaluation.
"""

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from operadic_consistency.core.toq_types import ToQ, ToQNode


@dataclass(frozen=True)
class BreakToQExample:
    question_id: str
    original_question: str
    toq: ToQ
    operators: tuple[str, ...]
    # The original HotpotQA question is the "total collapse" of the ToQ.
    # Used by HotpotCollapser for the empty-cut plan.


def _naturalize(step: str) -> str:
    """
    Convert a QDMR step string to a natural question.

    "return the president of France"  ->  "What is the president of France?"
    "return who managed #1"           ->  "Who managed [A1]?"
    "return #1 that won an Oscar"     ->  "[A1] that won an Oscar?"
    """
    s = step.strip()

    # Strip leading "return " (case-insensitive)
    s = re.sub(r"(?i)^return\s+", "", s)

    # Replace Break's #N references with core [AN] placeholders
    s = re.sub(r"#(\d+)", r"[A\1]", s)

    # Capitalize first character
    if s:
        s = s[0].upper() + s[1:]

    # Ensure it ends with "?"
    if s and not s.endswith("?"):
        s = s + "?"

    return s


def _parse_operators(operators_str: str) -> tuple[str, ...]:
    """Parse the operators field, which is a Python list literal stored as a string."""
    try:
        ops = ast.literal_eval(operators_str)
        return tuple(ops)
    except Exception:
        return ()


def _is_valid_2step(steps: Sequence[str]) -> bool:
    """
    Accept only decompositions where:
      - There are exactly 2 steps.
      - Step 2 references #1 (so there is an actual dependency).
    """
    if len(steps) != 2:
        return False
    if "#1" not in steps[1]:
        return False
    return True


def _to_toq(steps: Sequence[str]) -> ToQ:
    """Build a 2-node ToQ from two naturalized step strings."""
    leaf_text = _naturalize(steps[0])
    root_text = _naturalize(steps[1])

    nodes = {
        1: ToQNode(id=1, text=leaf_text, parent=2),
        2: ToQNode(id=2, text=root_text, parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=2)
    toq.validate()
    return toq


def load_hotpot_2step(
    split: str = "train",
    max_examples: Optional[int] = None,
    operators: Optional[Sequence[str]] = ("select", "project"),
) -> list[BreakToQExample]:
    """
    Load the HotpotQA 2-step subset from Break (QDMR-high-level).

    Args:
        split: "train" or "validation"
        max_examples: if set, cap the number of returned examples.
        operators: if set, only include examples whose operator list
            exactly matches this sequence. Defaults to ("select", "project"),
            which are the well-formed bridge questions. Pass None to include
            all operator types (including the noisier "select", "filter" ones).

    Returns:
        List of BreakToQExample, each containing the original question,
        its 2-node ToQ, and the operator tuple.
    """
    # Import here so the rest of the module is usable without `datasets` installed
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("allenai/break_data", "QDMR-high-level", split=split)

    examples: list[BreakToQExample] = []

    for row in ds:
        if not row["question_id"].startswith("HOTPOT"):
            continue

        raw_steps = [s.strip() for s in row["decomposition"].split(";")]

        if not _is_valid_2step(raw_steps):
            continue

        ops = _parse_operators(row["operators"])

        if operators is not None and ops != tuple(operators):
            continue

        toq = _to_toq(raw_steps)

        examples.append(
            BreakToQExample(
                question_id=row["question_id"],
                original_question=row["question_text"],
                toq=toq,
                operators=ops,
            )
        )

        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples
