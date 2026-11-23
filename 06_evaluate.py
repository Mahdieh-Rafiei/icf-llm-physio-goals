"""
Step 6: Compare Options A, B, C — print and save the full results table.

Reads:
  output/results_option_a.jsonl
  output/results_option_b.jsonl
  output/results_option_c.jsonl

Outputs:
  output/comparison_table.csv   (per-cohort × per-option accuracy)
  Prints formatted table to stdout
"""

import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import OUTPUT_DIR
from utils.eval_utils import aggregate_scores, score_prediction


OPTION_FILES = {
    "A_ZeroShot": OUTPUT_DIR / "results_option_a.jsonl",
    "B_FewShot": OUTPUT_DIR / "results_option_b.jsonl",
    "C_RAG": OUTPUT_DIR / "results_option_c.jsonl",
}


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            # Recompute scores in case eval logic changed
            scores = score_prediction(rec["predicted_codes"], rec["gold_codes"])
            rec.update(scores)
            results.append(rec)
    return results


def build_comparison_table() -> pd.DataFrame:
    """
    Builds a multi-level table:
      rows   = cohort (Akut / Geriat / PostAkut / Vertigo / ALL)
      cols   = (option, metric) pairs
    """
    all_summaries = {}

    for option_name, path in OPTION_FILES.items():
        if not path.exists():
            print(f"  Warning: {path.name} not found, skipping {option_name}")
            continue
        results = load_results(path)
        summary = aggregate_scores(results)
        all_summaries[option_name] = summary
        print(f"\n=== {option_name} ===")
        print(summary.to_string())

    if not all_summaries:
        print("No result files found. Run options A, B, C first.")
        return pd.DataFrame()

    # Build combined table
    frames = []
    for opt, df in all_summaries.items():
        df = df.copy()
        df.columns = [f"{opt}_{col}" for col in df.columns]
        frames.append(df)

    combined = pd.concat(frames, axis=1)
    return combined


def main():
    print("Loading and comparing results...")
    table = build_comparison_table()

    if table.empty:
        return

    out_path = OUTPUT_DIR / "comparison_table.csv"
    table.to_csv(out_path)
    print(f"\nComparison table saved to {out_path}")

    print("\n" + "=" * 70)
    print("FULL COMPARISON TABLE")
    print("=" * 70)
    print(table.to_string())

    # Also print a clean accuracy-only summary
    print("\n" + "=" * 70)
    print("EXACT MATCH ACCURACY SUMMARY")
    print("=" * 70)
    exact_cols = [c for c in table.columns if "exact_acc" in c]
    if exact_cols:
        print(table[exact_cols].to_string())

    print("\nONE-LEVEL-UP ACCURACY SUMMARY")
    print("=" * 70)
    olu_cols = [c for c in table.columns if "one_level_up_acc" in c]
    if olu_cols:
        print(table[olu_cols].to_string())

    print("\nCOMPONENT-LEVEL ACCURACY SUMMARY")
    print("=" * 70)
    comp_cols = [c for c in table.columns if "component_acc" in c]
    if comp_cols:
        print(table[comp_cols].to_string())


if __name__ == "__main__":
    main()
