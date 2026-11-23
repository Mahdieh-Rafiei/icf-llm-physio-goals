"""
Step 12: Validate the EHR Uni reliability tiers on the annotated 4ReusD test set.

Runs the exact Phase 2 pipeline (llama4, zero-shot + RAG, component/chapter/confidence
prompts imported from 10_uke_annotation.py) on the 470 4ReusD test statements, assigns
High/Medium/Low tiers with the manuscript §2.4 rule, and measures accuracy per tier
against the expert gold codes.

Writes only new files (prefix reusd_tier_validation_*) — no existing output is touched.

Usage:
  python scripts/12_4reusd_tier_validation.py --limit 10    # smoke test
  python scripts/12_4reusd_tier_validation.py
  python scripts/12_4reusd_tier_validation.py --eval-only   # recompute tables from saved runs
"""

from __future__ import annotations

import sys
import json
import argparse
import importlib.util
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, OUTPUT_DIR
from utils.llm_utils import call_llm
from utils.embed_utils import ConceptIndex, query_concept_index

_spec = importlib.util.spec_from_file_location(
    "uke_annotation", Path(__file__).parent / "10_uke_annotation.py"
)
uke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uke)

PREFIX = "reusd_tier_validation"
ZERO_PATH = OUTPUT_DIR / f"{PREFIX}_zeroshot.jsonl"
RAG_PATH = OUTPUT_DIR / f"{PREFIX}_rag.jsonl"
ROWS_PATH = OUTPUT_DIR / f"{PREFIX}_rows.csv"
SUMMARY_PATH = OUTPUT_DIR / f"{PREFIX}_summary.csv"
TIER_ORDER = ["High", "Medium", "Low", "Invalid"]


def run_condition(test_df, model, condition, concept_index, out_path):
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            done_ids = {json.loads(line)["idx"] for line in f if line.strip()}
        print(f"  Resuming: {len(done_ids)} rows already done.")

    todo = test_df[~test_df["idx"].isin(done_ids)]
    print(f"  Running {len(todo)} rows...")

    with open(out_path, "a") as f_out:
        for _, row in tqdm(todo.iterrows(), total=len(todo)):
            konzept = row["konzept"]
            if condition == "zeroshot":
                prompt = uke.build_zeroshot_prompt(konzept)
            else:
                examples = query_concept_index(konzept, concept_index, top_k=uke.TOP_K_EXAMPLES)
                prompt = uke.build_rag_prompt(konzept, examples)
            result = call_llm(prompt, model=model, think=False)
            comp, chap, conf = uke.parse_annotation(result)
            f_out.write(json.dumps({
                "idx": int(row["idx"]),
                "konzept": konzept,
                "component": comp,
                "chapter": chap,
                "confidence": conf,
                "llm_error": bool(result.get("error") or result.get("parse_error")),
            }, ensure_ascii=False) + "\n")
            f_out.flush()

    return pd.read_json(out_path, lines=True)


def gold_sets(codes):
    std = [c.lower() for c in codes if c and c[0].lower() in uke.VALID_COMPONENTS and len(c) > 1 and c[1].isdigit()]
    return {c[0] for c in std}, {c[:2] for c in std}


def assign_tier(r):
    if "?" in (r.comp_zeroshot, r.comp_rag, r.chapter_rag):
        return "Invalid"
    if r.comp_zeroshot != r.comp_rag:
        return "Low"
    if r.chapter_zeroshot == r.chapter_rag and r.conf_rag == "high":
        return "High"
    return "Medium"


def evaluate(test_df, df_zero, df_rag):
    df = test_df[["idx", "cohort", "konzept", "gold_codes"]].merge(
        df_zero[["idx", "component", "chapter", "confidence"]].rename(columns={
            "component": "comp_zeroshot", "chapter": "chapter_zeroshot", "confidence": "conf_zeroshot"}),
        on="idx",
    ).merge(
        df_rag[["idx", "component", "chapter", "confidence"]].rename(columns={
            "component": "comp_rag", "chapter": "chapter_rag", "confidence": "conf_rag"}),
        on="idx",
    )

    golds = df["gold_codes"].apply(gold_sets)
    df["gold_components"] = golds.apply(lambda g: ",".join(sorted(g[0])))
    df["gold_chapters"] = golds.apply(lambda g: ",".join(sorted(g[1])))
    df["has_standard_gold"] = golds.apply(lambda g: bool(g[0]))
    df["rag_component_correct"] = [r in g[0] for r, g in zip(df["comp_rag"], golds)]
    df["rag_chapter_correct"] = [r in g[1] for r, g in zip(df["chapter_rag"], golds)]
    df["tier"] = df.apply(assign_tier, axis=1)
    df.to_csv(ROWS_PATH, index=False)

    scored = df[df["has_standard_gold"]]
    rows = []
    for tier in TIER_ORDER + ["All"]:
        all_t = df if tier == "All" else df[df["tier"] == tier]
        sc = scored if tier == "All" else scored[scored["tier"] == tier]
        rows.append({
            "tier": tier,
            "n": len(all_t),
            "pct_of_total": round(len(all_t) / len(df) * 100, 1),
            "n_scored": len(sc),
            "component_acc": round(sc["rag_component_correct"].mean() * 100, 1) if len(sc) else None,
            "chapter_acc": round(sc["rag_chapter_correct"].mean() * 100, 1) if len(sc) else None,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n=== Reliability tiers on 4ReusD test set (accuracy vs expert gold) ===")
    print(summary.to_string(index=False))
    print(f"\nExcluded from accuracy (no standard ICF gold code, e.g. nd/nc only): "
          f"{(~df['has_standard_gold']).sum()}")
    print(f"LLM call/parse errors: zero-shot {int(df_zero['llm_error'].sum())}, "
          f"RAG {int(df_rag['llm_error'].sum())}")
    print(f"\nSaved → {ROWS_PATH}\nSaved → {SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Validate reliability tiers on 4ReusD test set.")
    parser.add_argument("--model", default="llama4:latest")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--eval-only", action="store_true", help="Skip LLM calls, evaluate saved runs")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)

    train_df, test_df = load_splits()
    if args.limit:
        test_df = test_df.head(args.limit)
    print(f"4ReusD test statements: {len(test_df)}")

    if args.eval_only:
        df_zero = pd.read_json(ZERO_PATH, lines=True)
        df_rag = pd.read_json(RAG_PATH, lines=True)
    else:
        concept_index = ConceptIndex.load(OUTPUT_DIR / "concept_index.pkl")
        n_index = len(concept_index.metadata)
        print(f"Concept index: {n_index} examples (train set: {len(train_df)})")
        if n_index != len(train_df):
            sys.exit("ERROR: concept index size differs from training set — possible test leakage.")

        print(f"\nZERO-SHOT — {args.model}")
        df_zero = run_condition(test_df, args.model, "zeroshot", concept_index, ZERO_PATH)
        print(f"\nRAG — {args.model}")
        df_rag = run_condition(test_df, args.model, "rag", concept_index, RAG_PATH)

    evaluate(test_df, df_zero, df_rag)


if __name__ == "__main__":
    main()
