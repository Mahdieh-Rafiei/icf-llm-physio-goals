"""
Step 13: Assign reliability tiers to EHR Uni annotations (output of 10_uke_annotation.py).

Uses the same tier rule as 12_4reusd_tier_validation.py (manuscript §2.4), so the tiers
reported for EHR Uni and the per-tier accuracy measured on 4ReusD are directly comparable.

Usage:
  python scripts/13_uke_reliability_tiers.py
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import OUTPUT_DIR

_spec = importlib.util.spec_from_file_location(
    "reusd_tier_validation", Path(__file__).parent / "12_4reusd_tier_validation.py"
)
val = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(val)


def main():
    df = pd.read_csv(OUTPUT_DIR / "uke_annotations.csv")
    df["tier"] = df.apply(val.assign_tier, axis=1)
    df.to_csv(OUTPUT_DIR / "uke_reliability_tiers.csv", index=False)

    counts = df["tier"].value_counts().reindex(val.TIER_ORDER, fill_value=0)
    summary = pd.DataFrame({
        "tier": counts.index,
        "n": counts.values,
        "pct": (counts.values / len(df) * 100).round(1),
    })
    summary.to_csv(OUTPUT_DIR / "uke_reliability_tiers_summary.csv", index=False)

    valid = df[df["comp_rag"] != "?"]
    print(f"EHR Uni statements: {len(df)} (valid RAG component: {len(valid)})")
    print(summary.to_string(index=False))
    print("\nComponent distribution (RAG, valid):")
    print(valid["comp_rag"].value_counts(normalize=True).mul(100).round(1).to_string())
    print("\nTop chapters (RAG, valid):")
    print(valid["chapter_rag"].value_counts(normalize=True).mul(100).round(1).head(8).to_string())


if __name__ == "__main__":
    main()
