"""
Step 14: Run-to-run stability of the reliability analysis.

Repeats the EHR Uni annotation (10) and the 4ReusD tier validation (12) with identical
prompts and rules, writing each repeat to output/repeats/run<k>/. Run 1 is the main
analysis in output/ and is only read, never written.

Usage:
  python scripts/14_repeat_runs.py --run 2
  python scripts/14_repeat_runs.py --run 3
  python scripts/14_repeat_runs.py --summarize
"""

from __future__ import annotations

import sys
import json
import argparse
import importlib.util
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, OUTPUT_DIR
from utils.embed_utils import ConceptIndex

_spec = importlib.util.spec_from_file_location(
    "reusd_tier_validation", Path(__file__).parent / "12_4reusd_tier_validation.py"
)
val = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(val)
uke = val.uke

REPEAT_DIR = OUTPUT_DIR / "repeats"
RUN1 = {
    "uke_zeroshot": OUTPUT_DIR / "uke_zeroshot.jsonl",
    "uke_rag": OUTPUT_DIR / "uke_rag.jsonl",
    "reusd_zeroshot": OUTPUT_DIR / "reusd_tier_validation_zeroshot.jsonl",
    "reusd_rag": OUTPUT_DIR / "reusd_tier_validation_rag.jsonl",
}


def run_paths(k: int) -> dict:
    if k == 1:
        return RUN1
    d = REPEAT_DIR / f"run{k}"
    return {name: d / f"{name}.jsonl" for name in RUN1}


def load_uke_inputs() -> pd.DataFrame:
    df = pd.read_csv(uke.DATASET_PATH).rename(columns={"Unnamed: 0": "row_id"})
    df = df.dropna(subset=["set_goal"])
    return df.rename(columns={"row_id": "idx", "set_goal": "konzept"})


def run_repeat(k: int, model: str):
    if k == 1:
        sys.exit("Run 1 is the main analysis and is not re-run here.")
    paths = run_paths(k)
    paths["uke_rag"].parent.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_splits()
    concept_index = ConceptIndex.load(OUTPUT_DIR / "concept_index.pkl")
    if len(concept_index.metadata) != len(train_df):
        sys.exit("ERROR: concept index size differs from training set.")
    uke_df = load_uke_inputs()

    for name, df in [("reusd", test_df), ("uke", uke_df)]:
        for cond in ["zeroshot", "rag"]:
            print(f"\nRun {k} — {name.upper()} {cond}")
            val.run_condition(df, model, cond, concept_index, paths[f"{name}_{cond}"])


def _load(path: Path, id_col: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    df = pd.DataFrame(rows)
    if id_col != "idx":
        df = df.rename(columns={id_col: "idx"})
    return df


def _tiers(zs: pd.DataFrame, rag: pd.DataFrame) -> pd.DataFrame:
    df = zs[["idx", "component", "chapter"]].rename(
        columns={"component": "comp_zeroshot", "chapter": "chapter_zeroshot"}
    ).merge(
        rag[["idx", "component", "chapter", "confidence"]].rename(
            columns={"component": "comp_rag", "chapter": "chapter_rag", "confidence": "conf_rag"}),
        on="idx",
    )
    df["tier"] = df.apply(val.assign_tier, axis=1).replace({"Invalid": "Low"})
    return df


def summarize_run(k: int, test_df: pd.DataFrame) -> dict:
    p = run_paths(k)
    out = {"run": k}

    u = _tiers(_load(p["uke_zeroshot"], "row_id"), _load(p["uke_rag"], "row_id"))
    n = len(u)
    valid = u[(u["comp_rag"] != "?") & (u["chapter_rag"] != "?")]
    for t in ["High", "Medium", "Low"]:
        out[f"uke_{t}_n"] = int((u["tier"] == t).sum())
        out[f"uke_{t}_pct"] = round((u["tier"] == t).mean() * 100, 1)
    out["uke_valid_n"] = len(valid)
    out["uke_d_pct"] = round((valid["comp_rag"] == "d").mean() * 100, 1)
    out["uke_b_pct"] = round((valid["comp_rag"] == "b").mean() * 100, 1)
    out["uke_d4_pct"] = round((valid["chapter_rag"] == "d4").mean() * 100, 1)
    out["uke_comp_agree_pct"] = round((u["comp_zeroshot"] == u["comp_rag"]).mean() * 100, 1)
    out["uke_conf_high_pct"] = round((u["conf_rag"] == "high").mean() * 100, 1)

    g = _tiers(_load(p["reusd_zeroshot"], "idx"), _load(p["reusd_rag"], "idx"))
    g = g.merge(test_df[["idx", "gold_codes"]], on="idx")
    golds = g["gold_codes"].apply(val.gold_sets)
    g["scored"] = golds.apply(lambda s: bool(s[0]))
    g["comp_ok"] = [c in s[0] for c, s in zip(g["comp_rag"], golds)]
    for t in ["High", "Medium", "Low"]:
        sc = g[(g["tier"] == t) & g["scored"]]
        out[f"reusd_{t}_n"] = int((g["tier"] == t).sum())
        out[f"reusd_{t}_comp_acc"] = round(sc["comp_ok"].mean() * 100, 1) if len(sc) else None
    return out


def summarize():
    _, test_df = load_splits()
    runs = [1] + sorted(
        int(d.name[3:]) for d in REPEAT_DIR.glob("run*")
        if all(path.exists() for path in run_paths(int(d.name[3:])).values())
    )
    per_run = pd.DataFrame([summarize_run(k, test_df) for k in runs]).set_index("run")
    REPEAT_DIR.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(REPEAT_DIR / "stability_per_run.csv")

    stats = per_run.agg(["mean", "min", "max"]).T.round(1)
    stats.to_csv(REPEAT_DIR / "stability_summary.csv")

    pd.set_option("display.width", 200)
    print(f"Runs included: {runs}  (Invalid counted as Low)\n")
    print(per_run.T.to_string())
    print("\n=== Mean [min–max] ===")
    print(stats.to_string())
    print(f"\nSaved → {REPEAT_DIR / 'stability_per_run.csv'}\nSaved → {REPEAT_DIR / 'stability_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Run-to-run stability of reliability analysis.")
    parser.add_argument("--run", type=int, help="Repeat number to execute (>= 2)")
    parser.add_argument("--model", default="llama4:latest")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)

    if args.summarize:
        summarize()
    elif args.run:
        run_repeat(args.run, args.model)
    else:
        parser.error("use --run K or --summarize")


if __name__ == "__main__":
    main()
