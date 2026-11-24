"""
Step 17: Breakdown of exact-match misses of the selected model (llama4 + RAG)
on the 4ReusD test set (Discussion and Supplementary Table S16).

Each goal statement without an exact match is assigned to one category:
  more_specific  - a predicted code is a descendant of a gold code (e.g. d4502 for d450)
  less_specific  - a predicted code is an ancestor of a gold code (e.g. d450 for d4502)
  different_code - any other miss (other category, or gold code non-standard)

Reads:  output/results_rag_llama4_latest.jsonl  (run 08_llm_comparison.py first)
Writes: output/exact_miss_breakdown.csv

Usage:
  python scripts/17_exact_miss_breakdown.py
"""
import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
RESULTS = OUTPUT_DIR / "results_rag_llama4_latest.jsonl"


def is_standard(code: str) -> bool:
    return len(code) >= 2 and code[0] in "bsde" and code[1].isdigit()


def classify(gold: list[str], pred: list[str]) -> str:
    g = [c for c in gold if is_standard(c)]
    p = [c for c in pred if is_standard(c)]
    if any(pc != gc and pc.startswith(gc) for pc in p for gc in g):
        return "more_specific"
    if any(pc != gc and gc.startswith(pc) for pc in p for gc in g):
        return "less_specific"
    return "different_code"


def main():
    rows = [json.loads(line) for line in open(RESULTS, encoding="utf-8")]
    misses = [r for r in rows if not r["exact"]]
    cats = pd.Series([classify(r["gold_codes"], r["predicted_codes"]) for r in misses])

    order = ["more_specific", "less_specific", "different_code"]
    table = cats.value_counts().reindex(order, fill_value=0).rename("n").to_frame()
    table["pct_of_misses"] = (table["n"] / len(misses) * 100).round(1)
    table.loc["total"] = [len(misses), 100.0]
    table["n"] = table["n"].astype(int)

    print(f"Test statements: {len(rows)} | exact-match misses: {len(misses)}\n")
    print(table.to_string())
    table.to_csv(OUTPUT_DIR / "exact_miss_breakdown.csv")
    print(f"\nSaved: {OUTPUT_DIR / 'exact_miss_breakdown.csv'}")


if __name__ == "__main__":
    main()
