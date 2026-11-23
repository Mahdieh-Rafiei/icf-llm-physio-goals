"""
Step 9: Embedding model comparison.

Rebuilds the retrieval index with a different embedding model and re-runs
RAG (Option C) with the best LLM to isolate the contribution of the
embedding model from the LLM.

Default comparison:
  Baseline : qwen3-embedding:8b  → results_option_c.jsonl       (53.2% exact)
  Challenger: bge-m3:latest      → results_option_c_bge-m3.jsonl

The two runs use identical prompts and the same LLM (qwen3.6:35b by default).
The only difference is the embedding model used for retrieval.

Usage:
  python scripts/09_embedding_comparison.py
  python scripts/09_embedding_comparison.py --embed-model bge-m3:latest --llm qwen2.5:72b
"""

from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, load_icf_reference, OUTPUT_DIR
from utils.eval_utils import score_prediction, aggregate_scores
from utils.llm_utils import call_llm, build_rag_prompt

TOP_K_CODES    = 30
TOP_K_EXAMPLES = 6


def build_index_for_model(embed_model: str, slug: str):
    """Build concept + code index for a given embedding model."""
    import importlib
    import utils.embed_utils as eu

    # Override the embedding model at runtime; use EMBED_API_URL if set
    eu.EMBED_MODEL    = embed_model
    eu.OLLAMA_API_URL = os.getenv("EMBED_API_URL", os.environ["OLLAMA_API_URL"]).rstrip("/")
    eu.API_KEY        = os.getenv("EMBED_API_KEY", os.environ["COSYBIO_API_KEY"])

    concept_path = OUTPUT_DIR / f"concept_index_{slug}.pkl"
    code_path    = OUTPUT_DIR / f"icf_code_index_{slug}.pkl"

    train_df, _ = load_splits()
    icf_df      = load_icf_reference()

    if not concept_path.exists():
        print(f"  Building concept index with {embed_model}...")
        idx = eu.build_concept_index(train_df)
        idx.save(concept_path)
        print(f"  Saved → {concept_path}")
    else:
        print(f"  Concept index already exists: {concept_path}")

    if not code_path.exists():
        print(f"  Building ICF code index with {embed_model}...")
        idx = eu.build_icf_code_index(icf_df)
        idx.save(code_path)
        print(f"  Saved → {code_path}")
    else:
        print(f"  ICF code index already exists: {code_path}")

    return concept_path, code_path


def run_rag_with_index(
    concept_path: Path,
    code_path: Path,
    embed_model: str,
    llm_model: str,
    slug: str,
    limit: int | None = None,
    resume: bool = True,
):
    import utils.embed_utils as eu

    # Override embed model for query-time embedding; use EMBED_API_URL if set
    eu.EMBED_MODEL    = embed_model
    eu.OLLAMA_API_URL = os.getenv("EMBED_API_URL", os.environ["OLLAMA_API_URL"]).rstrip("/")
    eu.API_KEY        = os.getenv("EMBED_API_KEY", os.environ["COSYBIO_API_KEY"])

    from utils.embed_utils import (
        ConceptIndex, ICFCodeIndex,
        query_concept_index, query_icf_code_index,
    )

    out_path = OUTPUT_DIR / f"results_option_c_{slug}.jsonl"

    _, test_df = load_splits()
    if limit:
        test_df = test_df.head(limit)

    concept_index = ConceptIndex.load(concept_path)
    code_index    = ICFCodeIndex.load(code_path)
    print(f"  Concept index: {len(concept_index.metadata)} examples")
    print(f"  Code index:    {len(code_index.metadata)} codes")

    done_ids: set[int] = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["row_id"])
        if done_ids:
            print(f"  Resuming: {len(done_ids)} rows already done.")

    rows_to_run = test_df[~test_df["idx"].isin(done_ids)] if done_ids else test_df
    print(f"  Running {len(rows_to_run)} rows...")

    with open(out_path, "a" if resume else "w") as f_out:
        for _, row in tqdm(rows_to_run.iterrows(), total=len(rows_to_run)):
            konzept   = row["konzept"]
            gold      = list(row["gold_codes"])
            cohort    = row["cohort"]
            row_id    = int(row["idx"])

            candidates = query_icf_code_index(konzept, code_index, top_k=TOP_K_CODES)
            examples   = query_concept_index(konzept, concept_index, top_k=TOP_K_EXAMPLES)
            prompt     = build_rag_prompt(konzept, candidates, examples)
            result     = call_llm(prompt, model=llm_model, think=False)

            predicted  = result.get("predicted_codes", [])
            scores     = score_prediction(predicted, gold)

            record = {
                "row_id": row_id, "cohort": cohort,
                "konzept": konzept, "gold_codes": gold,
                "predicted_codes": predicted,
                "reasoning": result.get("reasoning", ""),
                "embed_model": embed_model, "llm_model": llm_model,
                **scores,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    return out_path


def print_comparison(baseline_path: Path, challenger_path: Path,
                     baseline_label: str, challenger_label: str):
    def load(path):
        return [json.loads(l) for l in open(path)]

    base = load(baseline_path)
    chal = load(challenger_path)

    def metrics(rows):
        n = len(rows)
        return {
            "exact": sum(r.get("exact", False) for r in rows) / n * 100,
            "one_up": sum(r.get("one_level_up", False) for r in rows) / n * 100,
            "comp": sum(r.get("component", False) for r in rows) / n * 100,
            "n": n,
        }

    bm = metrics(base)
    cm = metrics(chal)

    print("\n" + "=" * 60)
    print("EMBEDDING MODEL COMPARISON (RAG, same LLM)")
    print("=" * 60)
    print(f"{'Model':<30} {'Exact':>7} {'1-up':>7} {'Comp':>7}  n")
    print("-" * 60)
    print(f"{baseline_label:<30} {bm['exact']:>6.1f}% {bm['one_up']:>6.1f}% {bm['comp']:>6.1f}%  {bm['n']}")
    print(f"{challenger_label:<30} {cm['exact']:>6.1f}% {cm['one_up']:>6.1f}% {cm['comp']:>6.1f}%  {cm['n']}")
    delta_e = cm["exact"] - bm["exact"]
    delta_u = cm["one_up"] - bm["one_up"]
    delta_c = cm["comp"] - bm["comp"]
    sign = lambda x: "+" if x >= 0 else ""
    print(f"{'Δ (challenger − baseline)':<30} {sign(delta_e)}{delta_e:>5.1f}pp {sign(delta_u)}{delta_u:>5.1f}pp {sign(delta_c)}{delta_c:>5.1f}pp")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare embedding models for RAG-based ICF linking."
    )
    parser.add_argument("--embed-model", default="bge-m3:latest",
                        help="Challenger embedding model (default: bge-m3:latest)")
    parser.add_argument("--llm", default="qwen3.6:35b",
                        help="LLM to use for both runs (default: qwen3.6:35b)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N test rows (smoke test)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slug = args.embed_model.replace(":", "_").replace("/", "_")

    print(f"\nEmbedding comparison: qwen3-embedding:8b  vs  {args.embed_model}")
    print(f"LLM: {args.llm}")
    print(f"Test rows: {args.limit or 470}")

    # Build index for challenger model
    print(f"\n{'='*55}")
    print(f"Building index: {args.embed_model}")
    print("=" * 55)
    concept_path, code_path = build_index_for_model(args.embed_model, slug)

    # Run RAG with challenger embedding
    print(f"\n{'='*55}")
    print(f"Running RAG with {args.embed_model}")
    print("=" * 55)
    challenger_path = run_rag_with_index(
        concept_path, code_path,
        embed_model=args.embed_model,
        llm_model=args.llm,
        slug=slug,
        limit=args.limit,
        resume=not args.no_resume,
    )

    # Compare results
    baseline_path = OUTPUT_DIR / "results_option_c.jsonl"
    if baseline_path.exists() and challenger_path.exists():
        print_comparison(
            baseline_path, challenger_path,
            baseline_label=f"qwen3-embedding:8b + {args.llm}",
            challenger_label=f"{args.embed_model} + {args.llm}",
        )
    else:
        print(f"\nResults saved to {challenger_path}")
        print("Run baseline (Option C) first to enable comparison.")


if __name__ == "__main__":
    main()
