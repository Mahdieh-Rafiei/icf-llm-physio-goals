"""
Step 12: Cohen's Kappa at the component level (b / s / d / e).

Computes Cohen's Kappa between model predictions and gold-standard ICF
component labels on the LMU held-out test set (n=470).

Reads (run 08_llm_comparison.py first):
  output/results_rag_llama4_latest.jsonl   — primary analysis (best model)

Also reads for prompting comparison (run 03-05 first):
  output/results_option_a.jsonl            — zero-shot
  output/results_option_b.jsonl            — few-shot
  output/results_option_c.jsonl            — RAG (qwen3.6:35b)

Outputs:
  output/kappa_results.csv                 — kappa by model / condition / cohort
  Prints formatted table to stdout

Usage:
  python scripts/12_kappa.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import OUTPUT_DIR
from utils.eval_utils import compute_kappa, score_prediction


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "exact" not in rec:
                scores = score_prediction(rec["predicted_codes"], rec["gold_codes"])
                rec.update(scores)
            rows.append(rec)
    return rows


TARGETS = {
    "llama4 (RAG)":        OUTPUT_DIR / "results_rag_llama4_latest.jsonl",
    "qwen3.6 (Zero-shot)": OUTPUT_DIR / "results_option_a.jsonl",
    "qwen3.6 (Few-shot)":  OUTPUT_DIR / "results_option_b.jsonl",
    "qwen3.6 (RAG)":       OUTPUT_DIR / "results_option_c.jsonl",
}


def main():
    rows = []

    for label, path in TARGETS.items():
        if not path.exists():
            print(f"  Skipping {label}: {path.name} not found")
            continue

        results = load_jsonl(path)
        kappa_data = compute_kappa(results)
        overall = kappa_data["overall"]
        n = kappa_data["n"]

        print(f"\n{'='*55}")
        print(f"  {label}  (n={n} standard-code rows)")
        print(f"{'='*55}")
        print(f"  Overall Cohen's Kappa (component): {overall:.4f}")
        print(f"  Per-cohort:")
        for cohort, k in sorted(kappa_data["per_cohort"].items()):
            k_str = f"{k:.4f}" if k is not None else "n/a"
            print(f"    {cohort:<12} κ = {k_str}")

        rows.append({"model_condition": label, "n": n, "kappa_overall": overall})
        for cohort, k in kappa_data["per_cohort"].items():
            rows.append({
                "model_condition": label,
                "n": None,
                "cohort": cohort,
                "kappa_cohort": k,
            })

    if not rows:
        print("\nNo result files found. Run 08_llm_comparison.py and/or 03-05 first.")
        return

    out_path = OUTPUT_DIR / "kappa_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nKappa results saved to {out_path}")

    # Summary table for manuscript
    summary = [r for r in rows if "cohort" not in r and r["n"] is not None]
    if summary:
        print("\n" + "="*55)
        print("  MANUSCRIPT TABLE — Cohen's Kappa (component level)")
        print("="*55)
        df = pd.DataFrame(summary)[["model_condition", "n", "kappa_overall"]]
        df.columns = ["Model / Condition", "n (standard)", "κ (component)"]
        df["κ (component)"] = df["κ (component)"].map(lambda x: f"{x:.3f}" if x is not None else "n/a")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
