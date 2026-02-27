"""
hotpot_collapser_3step.py

Collapser for 3-step HotpotQA ToQs (both chain and fan-in tree).

For a 3-node ToQ there are 2^(n_edges) collapse plans.  The collapser is
called once per connected component of each plan.  Its job is to return a
single question string for that component.

Strategy
--------
For a *closed* component (no external inputs):
  - Single node:   return the node's naturalized text as-is.
  - Multi-node:    return the original HotpotQA question (full collapse).
    (We don't have a general oracle collapser, so we approximate multi-node
    closed components the same way as the 2-step case: use the original question.
    In practice, for 3-step Break examples the only fully-closed multi-node
    component is the whole tree, which maps exactly to the original question.)

For an *open* component (has external inputs, i.e. [AN] placeholders):
  - Single node:   return the node's text unchanged (substituter fills [AN]).
  - Multi-node (chain sub-tree with placeholder):
    We do a simple template substitution: compose the node texts
    by propagating placeholder names upward through the chain.
    For example, if node2 text is "Who managed [A1]?" and node3 text is
    "When was [A2] born?" and node1 is cut (external input A1):
      -> The component {2,3} has root 3, input={1}.
      -> We inline node2's text into node3's [A2] slot, yielding:
         "When was who managed [A1] born?"

    For a fan-in tree component where one leaf is cut (e.g. {2,3}, input={1}):
      -> Node3 text "Which is true of [A1], [A2]?" with [A2] coming from node2.
      -> Node2 is internal: substitute node2's question for [A2].
      -> Result: "Which is true of [A1], <node2_question>?"
         where <node2_question> has its trailing "?" stripped.

The substitution is purely textual (no LLM call) and is an oracle collapse:
it exactly mirrors how a perfectly compositional model would concatenate steps.
"""

import re
from typing import Optional

from operadic_consistency.core.toq_types import OpenToQ, ToQ


def _strip_q(text: str) -> str:
    """Strip trailing '?' for inline embedding."""
    return text.rstrip("?").strip()


def _compose_open_toq(open_toq: OpenToQ, original_question: str) -> str:
    """
    Compose the nodes of an OpenToQ into a single question string by
    topologically resolving internal [AN] references.

    Nodes whose id appears in open_toq.inputs are external: their [AN]
    placeholder is left as-is (to be filled by the substituter later).

    Internal nodes' answers are substituted textually in bottom-up order.
    """
    toq = open_toq.toq
    external_ids = set(open_toq.inputs)

    # Build children map
    children: dict = {nid: [] for nid in toq.nodes}
    for nid, node in toq.nodes.items():
        if node.parent is not None and node.parent in children:
            children[node.parent].append(nid)

    # Topological sort (leaves first)
    order = []
    visited = set()

    def dfs(u):
        if u in visited:
            return
        visited.add(u)
        for v in children.get(u, []):
            dfs(v)
        order.append(u)

    dfs(open_toq.root_id)
    order.reverse()  # root first; we want leaves first for substitution
    order.reverse()  # back to leaves first
    # Actually we want leaves first so we substitute bottom-up.
    # dfs appends in post-order (leaves first), so `order` is already leaves-first.
    # Let's redo clearly:
    topo = []
    visited2 = set()

    def post_order(u):
        visited2.add(u)
        for v in children.get(u, []):
            if v not in visited2:
                post_order(v)
        topo.append(u)

    post_order(open_toq.root_id)
    # topo is now leaves-first (post-order), root last

    # resolved[nid] = the collapsed text for that node
    resolved: dict = {}
    for nid in topo:
        node = toq.nodes[nid]
        text = node.text
        # Substitute internal children's resolved text into this node's placeholders
        for ch_id in children.get(nid, []):
            if ch_id not in external_ids:
                # Internal: inline its resolved text
                placeholder = f"[A{ch_id}]"
                inline = _strip_q(resolved[ch_id])
                text = text.replace(placeholder, inline)
        resolved[nid] = text

    return resolved[open_toq.root_id]


class HotpotCollapser3Step:
    """
    Collapser for 3-step HotpotQA ToQs (chain or fan-in tree).

    Args:
        original_question: The full original HotpotQA question string.
            Used when the entire tree is collapsed into one component.
    """

    def __init__(self, original_question: str) -> None:
        self._original_question = original_question

    def __call__(self, open_toq: OpenToQ, *, context: Optional[str] = None) -> str:
        toq = open_toq.toq
        n_nodes = len(toq.nodes)

        if not open_toq.inputs:
            # Closed component (no external inputs)
            if n_nodes == 1:
                # Single closed node: return its text as-is
                return toq.nodes[open_toq.root_id].text
            elif n_nodes == 3:
                # Full tree collapse (all 3 nodes in one component) = original question
                return self._original_question
            else:
                # Partial closed sub-tree (e.g. chain prefix {1,2}): compose textually
                return _compose_open_toq(open_toq, self._original_question)
        else:
            # Open component: has external placeholders
            if n_nodes == 1:
                # Single open node: return its text unchanged (substituter fills placeholders)
                return toq.nodes[open_toq.root_id].text
            else:
                # Multi-node open component: compose internal nodes textually
                return _compose_open_toq(open_toq, self._original_question)
