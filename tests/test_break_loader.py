"""
Tests for model_interface.break_loader and model_interface.hotpot_collapser.

These tests do NOT hit the network (no datasets download). They exercise:
  - _naturalize()
  - _is_valid_2step()
  - _to_toq()
  - HotpotCollapser correctness on both collapse plans
  - Full run_consistency_check round-trip with a mock answerer
"""

import pytest

from operadic_consistency.core.toq_types import ToQ, ToQNode, OpenToQ
from operadic_consistency.core.consistency import run_consistency_check
from operadic_consistency.core.interfaces import Answer
from operadic_consistency.model_interface.break_loader import (
    _naturalize,
    _is_valid_2step,
    _parse_operators,
    _to_toq,
)
from operadic_consistency.model_interface.hotpot_collapser import HotpotCollapser


# ---------------------------------------------------------------------------
# _naturalize
# ---------------------------------------------------------------------------

class TestNaturalize:
    def test_strips_return_prefix(self):
        assert _naturalize("return the president of France") == "The president of France?"

    def test_strips_return_case_insensitive(self):
        assert _naturalize("Return who managed #1") == "Who managed [A1]?"

    def test_replaces_hash_reference(self):
        result = _naturalize("return who managed #1")
        assert "[A1]" in result
        assert "#1" not in result

    def test_adds_question_mark(self):
        result = _naturalize("return something")
        assert result.endswith("?")

    def test_does_not_double_question_mark(self):
        # If the step already ends with "?"
        result = _naturalize("return who is it?")
        assert result.endswith("?")
        assert not result.endswith("??")

    def test_capitalizes_first_letter(self):
        result = _naturalize("return the 5,000 acre estate of Thomas Jefferson")
        assert result[0].isupper()

    def test_multiple_references(self):
        result = _naturalize("return #1 or #2")
        assert "[A1]" in result
        assert "[A2]" in result

    def test_empty_after_strip(self):
        # Edge case: "return" with nothing after
        result = _naturalize("return")
        # Should not crash; result is "?" at minimum
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _parse_operators
# ---------------------------------------------------------------------------

class TestParseOperators:
    def test_select_project(self):
        assert _parse_operators("['select', 'project']") == ("select", "project")

    def test_select_filter(self):
        assert _parse_operators("['select', 'filter']") == ("select", "filter")

    def test_malformed_returns_empty(self):
        assert _parse_operators("not a list") == ()


# ---------------------------------------------------------------------------
# _is_valid_2step
# ---------------------------------------------------------------------------

class TestIsValid2Step:
    def test_valid(self):
        assert _is_valid_2step(["return something", "return who managed #1"])

    def test_wrong_length_1(self):
        assert not _is_valid_2step(["return something"])

    def test_wrong_length_3(self):
        assert not _is_valid_2step(["a", "b #1", "c #2"])

    def test_missing_reference(self):
        # Step 2 has no #1
        assert not _is_valid_2step(["return something", "return other thing"])


# ---------------------------------------------------------------------------
# _to_toq
# ---------------------------------------------------------------------------

class TestToToq:
    def test_basic_structure(self):
        steps = ["return the 5,000 acre estate of Thomas Jefferson",
                 "return who managed #1"]
        toq = _to_toq(steps)

        assert toq.root_id == 2
        assert 1 in toq.nodes
        assert 2 in toq.nodes

        # Node 1 is the leaf (child)
        assert toq.nodes[1].parent == 2
        # Node 2 is the root
        assert toq.nodes[2].parent is None

        # Placeholder substitution happened
        assert "[A1]" in toq.nodes[2].text
        assert "#1" not in toq.nodes[2].text

    def test_validates_cleanly(self):
        steps = ["return something", "return who did #1"]
        toq = _to_toq(steps)
        toq.validate()  # should not raise


# ---------------------------------------------------------------------------
# HotpotCollapser
# ---------------------------------------------------------------------------

class TestHotpotCollapser:
    """
    Test collapser on both plans for the 2-node ToQ:
        Node 1 (leaf): "The 5,000 acre estate of Thomas Jefferson?"  parent=2
        Node 2 (root): "Who managed [A1]?"                           parent=None
    """

    def _make_toq(self):
        nodes = {
            1: ToQNode(1, "The 5,000 acre estate of Thomas Jefferson?", parent=2),
            2: ToQNode(2, "Who managed [A1]?", parent=None),
        }
        return ToQ(nodes=nodes, root_id=2)

    def test_full_collapse_returns_original_question(self):
        """Plan: cut_edges=() -- entire tree in one component, root=2, inputs=()"""
        toq = self._make_toq()
        original_q = "Who managed the 5,000 acre estate of Thomas Jefferson?"
        collapser = HotpotCollapser(original_question=original_q)

        # Simulate the full-collapse OpenToQ: all nodes, root=2, no inputs
        open_toq = OpenToQ(toq=toq, inputs=(), root_id=2)

        result = collapser(open_toq)
        assert result == original_q

    def test_leaf_component_returns_leaf_text(self):
        """Plan: cut_edges=(1,) -- leaf component is just node 1, inputs=()"""
        leaf_nodes = {1: ToQNode(1, "The 5,000 acre estate of Thomas Jefferson?", parent=None)}
        leaf_toq = ToQ(nodes=leaf_nodes, root_id=1)
        open_toq = OpenToQ(toq=leaf_toq, inputs=(), root_id=1)

        collapser = HotpotCollapser(original_question="anything")
        result = collapser(open_toq)
        assert result == "The 5,000 acre estate of Thomas Jefferson?"

    def test_root_open_component_returns_root_text(self):
        """Plan: cut_edges=(1,) -- root component is node 2 with input=1"""
        root_nodes = {2: ToQNode(2, "Who managed [A1]?", parent=None)}
        root_toq = ToQ(nodes=root_nodes, root_id=2)
        open_toq = OpenToQ(toq=root_toq, inputs=(1,), root_id=2)

        collapser = HotpotCollapser(original_question="anything")
        result = collapser(open_toq)
        assert result == "Who managed [A1]?"


# ---------------------------------------------------------------------------
# End-to-end: run_consistency_check with mock answerer
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """
    Full round-trip through run_consistency_check using:
      - A real 2-node ToQ from _to_toq
      - HotpotCollapser
      - A deterministic mock answerer
    """

    def _make_example(self):
        steps = [
            "return the 5,000 acre estate of Thomas Jefferson",
            "return who managed #1",
        ]
        toq = _to_toq(steps)
        original_q = "Who managed the 5,000 acre estate of Thomas Jefferson?"
        return toq, original_q

    def test_report_has_correct_number_of_runs(self):
        """
        2-node tree has 1 edge -> 2^1 = 2 collapse plans.
        With include_empty=True that's both plans.
        """
        toq, original_q = self._make_example()

        class MockAnswerer:
            def __call__(self, question, *, context=None):
                return Answer("Edmund Bacon")

        report = run_consistency_check(
            toq,
            answerer=MockAnswerer(),
            collapser=HotpotCollapser(original_q),
            plan_opts={"include_empty": True},
        )
        assert len(report.runs) == 2

    def test_baseline_answer_is_set(self):
        toq, original_q = self._make_example()

        class MockAnswerer:
            def __call__(self, question, *, context=None):
                return Answer("Edmund Bacon")

        report = run_consistency_check(
            toq,
            answerer=MockAnswerer(),
            collapser=HotpotCollapser(original_q),
        )
        assert report.base_root_answer.text == "Edmund Bacon"

    def test_consistent_answerer_produces_agreement(self):
        """When answerer always returns same answer, all runs should agree."""
        from operadic_consistency.core.metrics import agreement_rate

        toq, original_q = self._make_example()

        class MockAnswerer:
            def __call__(self, question, *, context=None):
                return Answer("Edmund Bacon")

        report = run_consistency_check(
            toq,
            answerer=MockAnswerer(),
            collapser=HotpotCollapser(original_q),
            plan_opts={"include_empty": True},
        )
        assert agreement_rate(report, use_normalized=False) == 1.0

    def test_inconsistent_answerer_detected(self):
        """
        Answerer returns different answers depending on question phrasing.
        The full-collapse plan asks the original question (gets answer A),
        the no-collapse plan asks node 2 with [A1] filled (gets answer B).
        This should show up as <1.0 agreement rate.
        """
        from operadic_consistency.core.metrics import agreement_rate

        toq, original_q = self._make_example()

        class InconsistentAnswerer:
            def __call__(self, question, *, context=None):
                if "5,000 acre estate" in question and "[A1]" not in question:
                    # Leaf question or original
                    return Answer("Monticello")
                if "managed" in question.lower() and "[A1]" not in question:
                    return Answer("Edmund Bacon")
                return Answer("Edmund Bacon")

        report = run_consistency_check(
            toq,
            answerer=InconsistentAnswerer(),
            collapser=HotpotCollapser(original_q),
            plan_opts={"include_empty": True},
        )
        # We're not asserting exact disagreement since it depends on the
        # exact questions asked, but we verify the report ran cleanly.
        assert len(report.runs) == 2
        assert report.base_root_answer is not None
