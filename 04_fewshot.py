"""
Option B: Few-shot LLM linking.

Input:  test concept (Konzept)
Prompt: ICF linking task + N fixed random examples + full ICF code list
Output: 1–4 predicted ICF codes

Fixed examples are sampled once from the pooled training set (all cohorts),
stratified to cover diverse code types (standard ICF + non-standard).

Writes predictions to output/results_option_b.jsonl
"""

from __future__ import annotations

import sys
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, load_icf_reference, icf_list_text, OUTPUT_DIR
from utils.llm_utils import call_llm, build_fewshot_prompt
from utils.eval_utils import score_prediction, NON_STANDARD

N_EXAMPLES = 8
RANDOM_SEED = 42


def select_fixed_examples(train_df, n: int = N_EXAMPLES, seed: int = RANDOM_SEED) -> list[dict]:
    """
    Select N diverse examples from the training pool:
    - 75% with standard ICF codes (diverse components b/s/d/e)
    - 25% with non-standard codes (nd/nc/pf/hc)
    Ensures cross-cohort coverage.
    """
    rng = random.Random(seed)

    # Separate standard and non-standard
    std_rows = train_df[~train_df["primary_code"].isin(NON_STANDARD)]
    nonstd_rows = train_df[train_df["primary_code"].isin(NON_STANDARD)]

    n_nonstd = max(1, n // 4)
    n_std = n - n_nonstd

    selected = []

    # Pick diverse standard examples: one per component where possible
    components = ["b", "d", "s", "e"]
    per_component = []
    for comp in components:
        comp_rows = std_rows[std_rows["primary_code"].str.startswith(comp)]
        if len(comp_rows) > 0:
            per_component.append(comp_rows.sample(1, random_state=seed))

    if per_component:
        import pandas as pd
        comp_sample = pd.concat(per_component)
        remaining_std = std_rows[~std_rows.index.isin(comp_sample.index)]
        extra_n = max(0, n_std - len(comp_sample))
        if extra_n > 0 and len(remaining_std) > 0:
            extra = remaining_std.sample(min(extra_n, len(remaining_std)), random_state=seed)
            std_sample = pd.concat([comp_sample, extra])
        else:
            std_sample = comp_sample.head(n_std)
    else:
        std_sample = std_rows.sample(min(n_std, len(std_rows)), random_state=seed)

    for _, row in std_sample.iterrows():
        selected.append({
            "konzept": row["konzept"],
            "gold_codes": list(row["gold_codes"]),
            "cohort": row["cohort"],
        })

    # Add non-standard examples
    if len(nonstd_rows) > 0:
        ns_sample = nonstd_rows.sample(
            min(n_nonstd, len(nonstd_rows)), random_state=seed
        )
        for _, row in ns_sample.iterrows():
            selected.append({
                "konzept": row["konzept"],
                "gold_codes": list(row["gold_codes"]),
                "cohort": row["cohort"],
            })

    rng.shuffle(selected)
    return selected[:n]


def run_option_b(limit: int | None = None, resume: bool = True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "results_option_b.jsonl"

    print("Loading data...")
    train_df, test_df = load_splits()
    icf_df = load_icf_reference()
    code_list = icf_list_text(icf_df)

    print(f"Selecting {N_EXAMPLES} fixed few-shot examples...")
    examples = select_fixed_examples(train_df, n=N_EXAMPLES)
    print("  Selected examples:")
    for i, ex in enumerate(examples, 1):
        print(f"    {i}. [{ex['cohort']}] \"{ex['konzept']}\" → {ex['gold_codes']}")

    # Save fixed examples for reproducibility
    ex_path = OUTPUT_DIR / "option_b_fixed_examples.json"
    with open(ex_path, "w") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"  Fixed examples saved to {ex_path}")

    if limit:
        test_df = test_df.head(limit)

    done_ids: set[int] = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["row_id"])
        print(f"Resuming: {len(done_ids)} rows already done.")

    rows_to_run = test_df[~test_df["idx"].isin(done_ids)] if done_ids else test_df

    print(f"Running Option B (few-shot, N={N_EXAMPLES}) on {len(rows_to_run)} test rows...")

    with open(out_path, "a" if resume else "w") as f_out:
        for _, row in tqdm(rows_to_run.iterrows(), total=len(rows_to_run)):
            konzept = row["konzept"]
            gold_codes = list(row["gold_codes"])
            cohort = row["cohort"]
            row_id = int(row["idx"])

            prompt = build_fewshot_prompt(konzept, code_list, examples)
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
    print(f"\n=== Option B: Few-shot (N={N_EXAMPLES}) Results ===")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_option_b(limit=args.limit, resume=not args.no_resume)
