"""
Step 16: Sensitivity analysis — repeat Phase 2 with gpt-oss:120b instead of llama4.

Runs the EHR Uni annotation (zero-shot + RAG) and the 4ReusD reliability-tier validation
with gpt-oss, using the identical prompts, retrieval index and tier rules as the main
analysis (10, 12, 13). Results are written only to output/sensitivity_gptoss/; the main
llama4 results in output/ are only read for comparison.

Usage:
  python scripts/16_gptoss_sensitivity.py            # run + compare
  python scripts/16_gptoss_sensitivity.py --compare  # compare only
"""

from __future__ import annotations

import sys
import argparse
import importlib.util
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, OUTPUT_DIR
from utils.embed_utils import ConceptIndex


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = _load_module("repeat_runs", "14_repeat_runs.py")
val = rep.val

MODEL = "gpt-oss:120b"
OUT_DIR = OUTPUT_DIR / "sensitivity_gptoss"
PATHS = {name: OUT_DIR / f"{name}.jsonl" for name in rep.RUN1}


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_splits()
    concept_index = ConceptIndex.load(OUTPUT_DIR / "concept_index.pkl")
    if len(concept_index.metadata) != len(train_df):
        sys.exit("ERROR: concept index size differs from training set.")
    uke_df = rep.load_uke_inputs()
    for name, df in [("reusd", test_df), ("uke", uke_df)]:
        for cond in ["zeroshot", "rag"]:
            print(f"\n{MODEL} — {name.upper()} {cond}")
            val.run_condition(df, MODEL, cond, concept_index, PATHS[f"{name}_{cond}"])


def summarize(paths: dict, test_df: pd.DataFrame) -> dict:
    original = rep.run_paths
    rep.run_paths = lambda k: paths
    try:
        return rep.summarize_run(0, test_df)
    finally:
        rep.run_paths = original


def compare():
    _, test_df = load_splits()
    llama = summarize(rep.RUN1, test_df)
    gpt = summarize(PATHS, test_df)
    table = pd.DataFrame({"llama4 (main)": llama, "gpt-oss (sensitivity)": gpt}).drop(index="run")

    u_l = rep._load(rep.RUN1["uke_rag"], "row_id")
    u_g = rep._load(PATHS["uke_rag"], "idx")
    m = u_l[["idx", "component", "chapter"]].merge(
        u_g[["idx", "component", "chapter"]], on="idx", suffixes=("_llama4", "_gptoss"))
    agree = pd.Series({
        "EHR Uni RAG component agreement llama4 vs gpt-oss (%)":
            round((m["component_llama4"] == m["component_gptoss"]).mean() * 100, 1),
        "EHR Uni RAG chapter agreement llama4 vs gpt-oss (%)":
            round((m["chapter_llama4"] == m["chapter_gptoss"]).mean() * 100, 1),
    })

    table.to_csv(OUT_DIR / "comparison_llama4_vs_gptoss.csv")
    agree.to_csv(OUT_DIR / "model_agreement_ehr_uni.csv", header=["value"])
    pd.set_option("display.width", 200)
    print(table.to_string())
    print()
    print(agree.to_string())
    print(f"\nSaved → {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="gpt-oss sensitivity analysis for Phase 2.")
    parser.add_argument("--compare", action="store_true", help="Skip LLM calls, compare saved runs")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)

    if not args.compare:
        run()
    compare()


if __name__ == "__main__":
    main()
