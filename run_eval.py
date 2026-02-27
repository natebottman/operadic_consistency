"""
run_eval.py

Run the operadic consistency check on the HotpotQA/Break validation set
for a single model and save results to a JSON file.

Usage:
    python run_eval.py <model_id> <output_file> <api_key> [max_tokens]
"""

import json
import sys
import traceback

from operadic_consistency.model_interface import load_hotpot_2step, HotpotCollapser, TogetherAnswerer
from operadic_consistency.core.consistency import run_consistency_check
from operadic_consistency.core.metrics import agreement_rate

def run(model_id: str, output_file: str, api_key: str, max_tokens: int = 64):
    examples = load_hotpot_2step("validation")
    print(f"[{model_id}] Loaded {len(examples)} examples", flush=True)

    answerer = TogetherAnswerer(model=model_id, api_key=api_key, max_tokens=max_tokens)

    results = []
    for i, ex in enumerate(examples):
        try:
            report = run_consistency_check(
                ex.toq,
                answerer=answerer,
                collapser=HotpotCollapser(ex.original_question),
                plan_opts={"include_empty": True},
            )
            rate = agreement_rate(report, use_normalized=False)
            run_answers = [r.root_answer.text for r in report.runs]
            results.append({
                "question_id": ex.question_id,
                "question": ex.original_question,
                "step1": ex.toq.nodes[1].text,
                "step2": ex.toq.nodes[2].text,
                "baseline": report.base_root_answer.text,
                "direct_answer": run_answers[0],   # cut_edges=() plan
                "stepbystep_answer": run_answers[1], # cut_edges=(1,) plan
                "consistent": rate == 1.0,
            })
        except Exception as e:
            results.append({
                "question_id": ex.question_id,
                "question": ex.original_question,
                "error": str(e),
                "consistent": None,
            })

        if (i + 1) % 50 == 0:
            n_done = i + 1
            n_consistent = sum(1 for r in results if r.get("consistent") is True)
            print(f"[{model_id}] {n_done}/{len(examples)}  consistent so far: {n_consistent}/{n_done} ({100*n_consistent/n_done:.1f}%)", flush=True)

    with open(output_file, "w") as f:
        json.dump({"model": model_id, "results": results}, f, indent=2)

    n_valid = sum(1 for r in results if r.get("consistent") is not None)
    n_consistent = sum(1 for r in results if r.get("consistent") is True)
    print(f"[{model_id}] DONE. Consistent: {n_consistent}/{n_valid} ({100*n_consistent/n_valid:.1f}%)", flush=True)

if __name__ == "__main__":
    model_id, output_file, api_key = sys.argv[1], sys.argv[2], sys.argv[3]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    run(model_id, output_file, api_key, max_tokens=max_tokens)
