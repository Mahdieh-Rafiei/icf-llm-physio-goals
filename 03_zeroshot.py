"""
Option A: Zero-shot LLM linking.

Input:  test concept (Konzept)
Prompt: ICF linking task + full BfArM 2005 code list
Output: 1–4 predicted ICF codes

Writes predictions to output/results_option_a.jsonl
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, load_icf_reference, icf_list_text, OUTPUT_DIR
from utils.llm_utils import call_llm, build_zeroshot_prompt
from utils.eval_utils import score_prediction


def run_option_a(limit: int | None = None, resume: bool = True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "results_option_a.jsonl"

    print("Loading data...")
    _, test_df = load_splits()
    icf_df = load_icf_reference()
    code_list = icf_list_text(icf_df)  # stable across all calls → good for caching

    if limit:
        test_df = test_df.head(limit)

    # Resume support: skip already-processed rows
    done_ids: set[int] = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["row_id"])
        print(f"Resuming: {len(done_ids)} rows already done.")

    rows_to_run = test_df[~test_df["idx"].isin(done_ids)] if done_ids else test_df

    print(f"Running Option A (zero-shot) on {len(rows_to_run)} test rows...")

    with open(out_path, "a" if resume else "w") as f_out:
        for _, row in tqdm(rows_to_run.iterrows(), total=len(rows_to_run)):
            konzept = row["konzept"]
            gold_codes = list(row["gold_codes"])
            cohort = row["cohort"]
            row_id = int(row["idx"])

            prompt = build_zeroshot_prompt(konzept, code_list)
            result = call_llm(prompt)

            predicted = result.get("predicted_codes", [])
            scores = score_prediction(predicted, gold_codes)

            record = {
                "row_id": row_id,
                "cohort": cohort,
                "konzept": konzept,
                "gold_codes": gold_codes,
                "predicted_codes": predicted,
                "reasoning": result.get("reasoning", ""),
                **scores,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"\nDone. Results saved to {out_path}")
    _print_summary(out_path)


def _print_summary(path: Path):
    from utils.eval_utils import aggregate_scores
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    summary = aggregate_scores(results)
    print("\n=== Option A: Zero-shot Results ===")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N rows (for testing)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Overwrite existing results instead of resuming")
    args = parser.parse_args()
    run_option_a(limit=args.limit, resume=not args.no_resume)
