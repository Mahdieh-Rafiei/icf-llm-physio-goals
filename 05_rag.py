"""
Option C: RAG-based LLM linking (Retrieval-Augmented Generation).

Two-stage retrieval using multilingual-e5-large-instruct embeddings:
  1. Retrieve top-k semantically similar ICF codes → shortened candidate list
  2. Retrieve top-k similar annotated concepts → dynamic few-shot examples

Input:  test concept (Konzept)
Prompt: retrieved candidate ICF codes + retrieved similar examples + concept
Output: 1–4 predicted ICF codes

This eliminates the 1454-code list bottleneck and provides
task-specific, context-aware examples for each test concept.

Requires: output/concept_index.pkl and output/icf_code_index.pkl
           (run 02_build_index.py first)

Writes predictions to output/results_option_c.jsonl
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, OUTPUT_DIR
from utils.llm_utils import call_llm, build_rag_prompt
from utils.eval_utils import score_prediction
from utils.embed_utils import (
    ConceptIndex,
    ICFCodeIndex,
    query_concept_index,
    query_icf_code_index,
)

TOP_K_CODES = 30      # candidate ICF codes to retrieve
TOP_K_EXAMPLES = 6   # similar annotated concepts to retrieve


def run_option_c(limit: int | None = None, resume: bool = True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "results_option_c.jsonl"

    # Check index files exist
    concept_idx_path = OUTPUT_DIR / "concept_index.pkl"
    code_idx_path = OUTPUT_DIR / "icf_code_index.pkl"
    if not concept_idx_path.exists() or not code_idx_path.exists():
        print("ERROR: Index files not found. Run 02_build_index.py first.")
        sys.exit(1)

    print("Loading data...")
    _, test_df = load_splits()

    print("Loading indices...")
    concept_index = ConceptIndex.load(concept_idx_path)
    code_index = ICFCodeIndex.load(code_idx_path)
    print(f"  Concept index: {len(concept_index.metadata)} training examples")
    print(f"  Code index:    {len(code_index.metadata)} ICF codes")

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

    print(
        f"Running Option C (RAG: top-{TOP_K_CODES} codes + top-{TOP_K_EXAMPLES} examples) "
        f"on {len(rows_to_run)} test rows..."
    )

    with open(out_path, "a" if resume else "w") as f_out:
        for _, row in tqdm(rows_to_run.iterrows(), total=len(rows_to_run)):
            konzept = row["konzept"]
            gold_codes = list(row["gold_codes"])
            cohort = row["cohort"]
            row_id = int(row["idx"])

            # Stage 1: retrieve candidate ICF codes
            candidate_codes = query_icf_code_index(
                konzept, code_index, top_k=TOP_K_CODES
            )

            # Stage 2: retrieve similar annotated examples
            similar_examples = query_concept_index(
                konzept, concept_index, top_k=TOP_K_EXAMPLES,
            )

            prompt = build_rag_prompt(konzept, candidate_codes, similar_examples)
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
                "retrieved_code_candidates": [
                    {"code": c["code"], "sim": round(c["similarity"], 3)}
                    for c in candidate_codes[:5]  # log top-5 for analysis
                ],
                "retrieved_examples": [
                    {"konzept": ex["konzept"], "codes": ex["gold_codes"],
                     "sim": round(ex["similarity"], 3)}
                    for ex in similar_examples[:3]
                ],
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
    print("\n=== Option C: RAG Results ===")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--top-k-codes", type=int, default=TOP_K_CODES,
        help=f"Number of candidate codes to retrieve (default: {TOP_K_CODES})"
    )
    parser.add_argument(
        "--top-k-examples", type=int, default=TOP_K_EXAMPLES,
        help=f"Number of similar examples to retrieve (default: {TOP_K_EXAMPLES})"
    )
    args = parser.parse_args()
    TOP_K_CODES = args.top_k_codes
    TOP_K_EXAMPLES = args.top_k_examples
    run_option_c(limit=args.limit, resume=not args.no_resume)
