"""
hotpot_collapser.py

Collapser for 2-step HotpotQA ToQs loaded from Break.

For a 2-node tree (leaf=1, root=2) there are exactly 2 collapse plans:

  Plan A  cut_edges=()   -- full collapse (empty cut)
    The entire tree is one component rooted at node 2.
    inputs=() because there are no external inputs.
    -> Return the original HotpotQA question (the "total collapse").

  Plan B  cut_edges=(1,) -- no collapse (cut the only edge)
    Two components: {node 1} and {node 2}.
    The collapser is called once per component:
      - Component rooted at node 1: inputs=(), closed leaf.
        -> Return node 1's text as-is (already a natural question).
      - Component rooted at node 2: inputs=(1,), open (has external input).
        -> Return node 2's text as-is (contains [A1] placeholder).
    Neither call does any "collapsing"; the structure is preserved exactly.
    The substituter fills in [A1] at evaluation time.

Usage:
    collapser = HotpotCollapser(original_question="Who managed ...?")
    # Then pass to run_consistency_check(..., collapser=collapser)
"""

from typing import Optional

from operadic_consistency.core.toq_types import OpenToQ


class HotpotCollapser:
    """
    Collapser for a single 2-step HotpotQA example.

    Args:
        original_question: The original HotpotQA question string.
            Used as the collapsed question when the full tree is fused
            into a single node (empty-cut plan).
    """

    def __init__(self, original_question: str) -> None:
        self._original_question = original_question

    def __call__(self, open_toq: OpenToQ, *, context: Optional[str] = None) -> str:
        if not open_toq.inputs:
            # Closed component (no external inputs).
            # For the 2-step case this happens in two situations:
            #   1. Full collapse (all nodes in one component, root=2):
            #      return the original question.
            #   2. The leaf component (root=1, no children, no inputs):
            #      return its text as-is.
            root_node = open_toq.toq.nodes[open_toq.root_id]
            if len(open_toq.toq.nodes) > 1:
                # Multi-node closed component = full collapse
                return self._original_question
            else:
                # Single-node closed component = leaf, no collapsing needed
                return root_node.text
        else:
            # Open component: has external inputs ([A1] placeholder present).
            # Return the root node's text unchanged; substituter handles [A1].
            return open_toq.toq.nodes[open_toq.root_id].text
