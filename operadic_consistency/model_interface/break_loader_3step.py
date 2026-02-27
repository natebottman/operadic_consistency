"""
break_loader_3step.py

Load 3-step HotpotQA decompositions from Break (QDMR-high-level) and convert
them into ToQs for the operadic consistency check.

Two structural classes are supported:

  Chain:   node1 -> node2 -> node3 (root)
           step1 has no refs, step2 refs #1, step3 refs #2
           (purely sequential)

  Tree:    node1 \
                  -> node3 (root)   [fan-in / binary product]
           node2 /
           step1 and step2 have no refs, step3 refs both #1 and #2

The resulting ToQ node ids follow the Break step numbering:
  Chain: nodes {1, 2, 3}, root=3, edges 1->2->3
  Tree:  nodes {1, 2, 3}, root=3, edges 1->3 and 2->3

Usage:
    from operadic_consistency.model_interface.break_loader_3step import (
        load_hotpot_3step_chains,
        load_hotpot_3step_trees,
        BreakToQExample3,
    )
"""

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from operadic_consistency.core.toq_types import ToQ, ToQNode


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _naturalize(step: str) -> str:
    """
    Convert a QDMR high-level step string to a natural question.

    "return the president of France"  ->  "The president of France?"
    "return who managed #1"           ->  "Who managed [A1]?"
    "return which is true of #1, #2"  ->  "Which is true of [A1], [A2]?"
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


def _parse_operators(operators_str: str) -> tuple:
    try:
        ops = ast.literal_eval(operators_str)
        return tuple(ops)
    except Exception:
        return ()


def _get_refs(step_text: str) -> Set[int]:
    """Return the set of step-indices (1-based, as in Break) referenced by this step."""
    return set(int(r) for r in re.findall(r"#(\d+)", step_text))


def _split_steps(decomp: str) -> List[str]:
    return [s.strip() for s in decomp.split(";") if s.strip()]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BreakToQExample3:
    question_id: str
    original_question: str
    toq: ToQ
    operators: tuple
    structure: str   # "chain" or "tree"


# ---------------------------------------------------------------------------
# Chain loader  (1 -> 2 -> 3)
# ---------------------------------------------------------------------------

def _is_valid_chain(steps: List[str]) -> bool:
    """
    Chain: exactly 3 steps where step2 refs only #1 and step3 refs only #2.
    """
    if len(steps) != 3:
        return False
    refs1 = _get_refs(steps[0])
    refs2 = _get_refs(steps[1])
    refs3 = _get_refs(steps[2])
    return refs1 == set() and refs2 == {1} and refs3 == {2}


def _chain_to_toq(steps: List[str]) -> ToQ:
    """
    Build a 3-node chain ToQ:
      node 1 (leaf) -> node 2 -> node 3 (root)
    """
    nodes = {
        1: ToQNode(id=1, text=_naturalize(steps[0]), parent=2),
        2: ToQNode(id=2, text=_naturalize(steps[1]), parent=3),
        3: ToQNode(id=3, text=_naturalize(steps[2]), parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=3)
    toq.validate()
    return toq


def load_hotpot_3step_chains(
    split: str = "validation",
    max_examples: Optional[int] = None,
) -> List[BreakToQExample3]:
    """
    Load HotpotQA 3-step chain decompositions from Break (QDMR-high-level).

    A chain has the dependency pattern:
        step1 (no refs) -> step2 (refs #1) -> step3 (refs #2)

    Args:
        split: "train" or "validation"
        max_examples: cap on returned examples.

    Returns:
        List of BreakToQExample3 with structure="chain".
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("allenai/break_data", "QDMR-high-level", split=split)
    examples: List[BreakToQExample3] = []

    for row in ds:
        if not row["question_id"].startswith("HOTPOT"):
            continue

        steps = _split_steps(row["decomposition"])
        if not _is_valid_chain(steps):
            continue

        ops = _parse_operators(row["operators"])
        toq = _chain_to_toq(steps)

        examples.append(
            BreakToQExample3(
                question_id=row["question_id"],
                original_question=row["question_text"],
                toq=toq,
                operators=ops,
                structure="chain",
            )
        )

        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples


# ---------------------------------------------------------------------------
# Tree (fan-in) loader  (1 \-> 3, 2 \-> 3)
# ---------------------------------------------------------------------------

def _is_valid_tree(steps: List[str]) -> bool:
    """
    Fan-in tree: exactly 3 steps where step1 and step2 have no refs,
    and step3 refs both #1 and #2.
    """
    if len(steps) != 3:
        return False
    refs1 = _get_refs(steps[0])
    refs2 = _get_refs(steps[1])
    refs3 = _get_refs(steps[2])
    return refs1 == set() and refs2 == set() and {1, 2}.issubset(refs3)


def _tree_to_toq(steps: List[str]) -> ToQ:
    """
    Build a 3-node fan-in ToQ:
      node 1 (leaf) \
                     -> node 3 (root)
      node 2 (leaf) /
    """
    nodes = {
        1: ToQNode(id=1, text=_naturalize(steps[0]), parent=3),
        2: ToQNode(id=2, text=_naturalize(steps[1]), parent=3),
        3: ToQNode(id=3, text=_naturalize(steps[2]), parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=3)
    toq.validate()
    return toq


def load_hotpot_3step_trees(
    split: str = "validation",
    max_examples: Optional[int] = None,
) -> List[BreakToQExample3]:
    """
    Load HotpotQA 3-step fan-in tree decompositions from Break (QDMR-high-level).

    A fan-in tree has the dependency pattern:
        step1 (no refs) \
                          -> step3 (refs #1 and #2)
        step2 (no refs) /

    Args:
        split: "train" or "validation"
        max_examples: cap on returned examples.

    Returns:
        List of BreakToQExample3 with structure="tree".
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("allenai/break_data", "QDMR-high-level", split=split)
    examples: List[BreakToQExample3] = []

    for row in ds:
        if not row["question_id"].startswith("HOTPOT"):
            continue

        steps = _split_steps(row["decomposition"])
        if not _is_valid_tree(steps):
            continue

        ops = _parse_operators(row["operators"])
        toq = _tree_to_toq(steps)

        examples.append(
            BreakToQExample3(
                question_id=row["question_id"],
                original_question=row["question_text"],
                toq=toq,
                operators=ops,
                structure="tree",
            )
        )

        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples
