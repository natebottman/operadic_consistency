"""
run_eval_3step.py

Run the operadic consistency check on 3-step HotpotQA/Break examples
(chains and/or fan-in trees) for a single model and save results to JSON.

Usage:
    python run_eval_3step.py <model_id> <output_file> <api_key> [structure] [max_tokens]

    structure: "chain", "tree", or "both" (default: "both")
    max_tokens: default 64

Example:
    python run_eval_3step.py meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \\
        results/llama8b_3step.json $TOGETHER_API_KEY both 64
"""

import json
import sys
import traceback

from operadic_consistency.model_interface.break_loader_3step import (
    load_hotpot_3step_chains,
    load_hotpot_3step_trees,
)
from operadic_consistency.model_interface.hotpot_collapser_3step import HotpotCollapser3Step
from operadic_consistency.model_interface.llm_answerer import TogetherAnswerer
from operadic_consistency.core.consistency import run_consistency_check
from operadic_consistency.core.transforms import CollapsePlan
from operadic_consistency.core.metrics import agreement_rate


def run(
    model_id: str,
    output_file: str,
    api_key: str,
    structure: str = "both",
    max_tokens: int = 64,
):
    # Load examples
    examples = []
    if structure in ("chain", "both"):
        chains = load_hotpot_3step_chains("validation")
        print(f"[{model_id}] Loaded {len(chains)} 3-step chain examples", flush=True)
        examples.extend(chains)
    if structure in ("tree", "both"):
        trees = load_hotpot_3step_trees("validation")
        print(f"[{model_id}] Loaded {len(trees)} 3-step tree (fan-in) examples", flush=True)
        examples.extend(trees)

    print(f"[{model_id}] Total: {len(examples)} examples", flush=True)

    answerer = TogetherAnswerer(model=model_id, api_key=api_key, max_tokens=max_tokens)

    results = []
    for i, ex in enumerate(examples):
        try:
            # Only use two plans:
            #   plan_full = cut_edges=()       -> full collapse (original question)
            #   plan_none = cut_edges=(1,2,...) -> no collapse (all steps separate)
            non_root_ids = tuple(sorted(nid for nid in ex.toq.nodes if nid != ex.toq.root_id))
            plan_full = CollapsePlan(cut_edges=())
            plan_none = CollapsePlan(cut_edges=non_root_ids)

            report = run_consistency_check(
                ex.toq,
                answerer=answerer,
                collapser=HotpotCollapser3Step(ex.original_question),
                plan_opts={"include_empty": True, "plans_override": [plan_full, plan_none]},
            )
            rate = agreement_rate(report, use_normalized=False)
            run_answers = [r.root_answer.text for r in report.runs]
            plan_descs = [str(list(r.plan.cut_edges)) for r in report.runs]

            results.append({
                "question_id": ex.question_id,
                "structure": ex.structure,
                "question": ex.original_question,
                "steps": {
                    str(nid): node.text
                    for nid, node in ex.toq.nodes.items()
                },
                "operators": list(ex.operators),
                "baseline": report.base_root_answer.text,
                "plan_answers": dict(zip(plan_descs, run_answers)),
                "consistent": rate == 1.0,
                "agreement_rate": rate,
            })
        except Exception as e:
            results.append({
                "question_id": ex.question_id,
                "structure": ex.structure,
                "question": ex.original_question,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "consistent": None,
            })

        if (i + 1) % 50 == 0:
            n_done = i + 1
            n_consistent = sum(1 for r in results if r.get("consistent") is True)
            n_valid = sum(1 for r in results if r.get("consistent") is not None)
            print(
                f"[{model_id}] {n_done}/{len(examples)}  "
                f"consistent: {n_consistent}/{n_valid} ({100*n_consistent/max(n_valid,1):.1f}%)",
                flush=True,
            )

    with open(output_file, "w") as f:
        json.dump({"model": model_id, "structure": structure, "results": results}, f, indent=2)

    n_valid = sum(1 for r in results if r.get("consistent") is not None)
    n_consistent = sum(1 for r in results if r.get("consistent") is True)

    # Per-structure breakdown
    for s in ("chain", "tree"):
        sub = [r for r in results if r.get("structure") == s and r.get("consistent") is not None]
        sub_c = sum(1 for r in sub if r.get("consistent") is True)
        if sub:
            print(
                f"[{model_id}] {s}: {sub_c}/{len(sub)} consistent ({100*sub_c/len(sub):.1f}%)",
                flush=True,
            )

    print(
        f"[{model_id}] DONE. Overall consistent: {n_consistent}/{n_valid} ({100*n_consistent/max(n_valid,1):.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    model_id = sys.argv[1]
    output_file = sys.argv[2]
    api_key = sys.argv[3]
    structure = sys.argv[4] if len(sys.argv) > 4 else "both"
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    run(model_id, output_file, api_key, structure=structure, max_tokens=max_tokens)
