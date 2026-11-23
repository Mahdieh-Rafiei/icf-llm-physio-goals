"""
LLM Model Comparison: run RAG-based linking across multiple LLMs.

All models use the same RAG retrieval (same embedding index) so that
differences in accuracy are attributable solely to the LLM, not retrieval.

Models compared:
  Ollama server:
    qwen3.6:35b, deepseek-r1:32b, aya-expanse:32b, llama3.3:70b,
    qwen2.5:72b, gemma4:latest, llama4:latest, gpt-oss:120b
  Premium APIs (set keys in .env):
    gpt-4o         — OpenAI API (OPENAI_API_KEY)

Outputs per model: output/results_rag_{slug}.jsonl
Summary table:     output/llm_comparison_table.csv

Requires: output/concept_index.pkl and output/icf_code_index.pkl
          (run 02_build_index.py first; rebuild if EMBED_MODEL changed)
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, OUTPUT_DIR
from utils.llm_utils import call_llm, build_rag_prompt
from utils.eval_utils import score_prediction, aggregate_scores
from utils.embed_utils import (
    ConceptIndex,
    ICFCodeIndex,
    query_concept_index,
    query_icf_code_index,
)

TOP_K_CODES = 30
TOP_K_EXAMPLES = 6

# Models to compare. think=False for models without reasoning capability.
COMPARISON_MODELS = [
    {"name": "qwen3.6:35b",    "think": True,  "slug": "qwen3.6_35b"},
    {"name": "deepseek-r1:32b","think": True,  "slug": "deepseek-r1_32b"},
    {"name": "aya-expanse:32b","think": False, "slug": "aya-expanse_32b"},
    {"name": "llama3.3:70b",   "think": False, "slug": "llama3.3_70b"},
    {"name": "qwen2.5:72b",    "think": False, "slug": "qwen2.5_72b"},
    {"name": "gemma4:latest",  "think": False, "slug": "gemma4_latest"},
    {"name": "llama4:latest",      "think": False, "slug": "llama4_latest"},
    {"name": "gpt-oss:120b",      "think": False, "slug": "gpt-oss_120b"},
    # Premium API models — requires OPENAI_API_KEY in .env
    {"name": "gpt-4o",            "think": False, "slug": "gpt-4o",            "provider": "openai"},
]

# Optional extra models from an untracked local module (utils/local_providers.py)
try:
    from utils.local_providers import EXTRA_MODELS
    COMPARISON_MODELS += EXTRA_MODELS
except ImportError:
    pass


def warmup_model(model_name: str, think: bool, max_attempts: int = 30) -> bool:
    """
    Send tiny requests until the model responds within the proxy timeout.
    Ollama continues loading the model even when the proxy drops the connection,
    so repeated retries will eventually hit a warm model.
    """
    import requests as _req
    from utils.llm_utils import OLLAMA_API_URL, _headers
    print(f"  Warming up {model_name} (may take several minutes)...")
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _req.post(
                f"{OLLAMA_API_URL}/api/chat",
                headers=_headers(),
                json={"model": model_name, "stream": True, "think": think,
                      "messages": [{"role": "user", "content": "Reply with the single word: ready"}]},
                stream=True,
                timeout=115,  # just under the 120s proxy limit
            )
            resp.raise_for_status()
            content = ""
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    content += chunk.get("message", {}).get("content", "")
                    if chunk.get("done"):
                        break
            print(f"  Model warm! Response: {content.strip()[:30]!r}")
            return True
        except Exception as e:
            print(f"  Warmup attempt {attempt}/{max_attempts} failed ({type(e).__name__}), retrying in 5s...")
            time.sleep(5)
    print(f"  WARNING: model did not warm up after {max_attempts} attempts — proceeding anyway.")
    return False


def run_rag_for_model(
    model_cfg: dict,
    test_df,
    concept_index: ConceptIndex,
    code_index: ICFCodeIndex,
    limit: int | None = None,
    resume: bool = True,
) -> Path:
    slug = model_cfg["slug"]
    model_name = model_cfg["name"]
    think = model_cfg["think"]
    provider = model_cfg.get("provider", None)  # None = auto-detect (Ollama)

    out_path = OUTPUT_DIR / f"results_rag_{slug}.jsonl"

    done_ids: set[int] = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["row_id"])
        if done_ids:
            print(f"  Resuming: {len(done_ids)} rows already done.")

    df = test_df.head(limit) if limit else test_df
    rows_to_run = df[~df["idx"].isin(done_ids)] if done_ids else df

    if len(rows_to_run) == 0:
        print(f"  All rows already processed.")
        return out_path

    print(f"  Running {len(rows_to_run)} rows...")

    with open(out_path, "a" if resume else "w") as f_out:
        for _, row in tqdm(rows_to_run.iterrows(), total=len(rows_to_run)):
            konzept = row["konzept"]
            gold_codes = list(row["gold_codes"])
            cohort = row["cohort"]
            row_id = int(row["idx"])

            candidate_codes = query_icf_code_index(konzept, code_index, top_k=TOP_K_CODES)
            similar_examples = query_concept_index(konzept, concept_index, top_k=TOP_K_EXAMPLES)

            prompt = build_rag_prompt(konzept, candidate_codes, similar_examples)
            result = call_llm(prompt, model=model_name, think=think, provider=provider)

            predicted = result.get("predicted_codes", [])
            scores = score_prediction(predicted, gold_codes)

            record = {
                "row_id": row_id,
                "cohort": cohort,
                "konzept": konzept,
                "gold_codes": gold_codes,
                "predicted_codes": predicted,
                "reasoning": result.get("reasoning", ""),
                "model": model_name,
                **scores,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    return out_path


def build_comparison_table(model_cfgs: list[dict]) -> pd.DataFrame:
    all_summaries: dict[str, pd.DataFrame] = {}
    for cfg in model_cfgs:
        path = OUTPUT_DIR / f"results_rag_{cfg['slug']}.jsonl"
        if not path.exists():
            print(f"  Warning: {path.name} not found, skipping {cfg['name']}")
            continue
        results = []
        with open(path) as f:
            for line in f:
                results.append(json.loads(line))
        all_summaries[cfg["slug"]] = aggregate_scores(results)

    if not all_summaries:
        return pd.DataFrame()

    frames = []
    for slug, df in all_summaries.items():
        renamed = df.copy()
        renamed.columns = [f"{slug}_{col}" for col in df.columns]
        frames.append(renamed)

    return pd.concat(frames, axis=1)


def print_metric_section(table: pd.DataFrame, col_substr: str, label: str) -> None:
    cols = [c for c in table.columns if col_substr in c]
    if cols:
        print(f"\n{label}")
        print("-" * 60)
        print(table[cols].to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG linking with multiple LLMs and compare accuracy."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N test rows (for smoke test)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Restart from scratch (ignore existing results)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model names to run (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate index files
    concept_idx_path = OUTPUT_DIR / "concept_index.pkl"
    code_idx_path = OUTPUT_DIR / "icf_code_index.pkl"
    if not concept_idx_path.exists() or not code_idx_path.exists():
        print("ERROR: Index files not found. Run 02_build_index.py first.")
        sys.exit(1)

    print("Loading data and indices...")
    _, test_df = load_splits()
    concept_index = ConceptIndex.load(concept_idx_path)
    code_index = ICFCodeIndex.load(code_idx_path)
    print(f"  Concept index: {len(concept_index.metadata)} training examples")
    print(f"  Code index:    {len(code_index.metadata)} ICF codes")
    print(f"  Test rows:     {len(test_df)}")

    # Select models to run
    models_to_run = COMPARISON_MODELS
    if args.models:
        name_set = set(args.models)
        models_to_run = [m for m in COMPARISON_MODELS if m["name"] in name_set]
        if not models_to_run:
            available = [m["name"] for m in COMPARISON_MODELS]
            print(f"No matching models found. Available: {available}")
            sys.exit(1)

    print(f"\nComparing {len(models_to_run)} models (RAG, top-{TOP_K_CODES} codes, top-{TOP_K_EXAMPLES} examples):")
    for cfg in models_to_run:
        think_str = "thinking" if cfg["think"] else "no thinking"
        print(f"  {cfg['name']}  ({think_str})")

    # Run RAG for each model
    for cfg in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {cfg['name']}")
        print("=" * 60)
        if "provider" not in cfg:  # API models need no Ollama warmup
            warmup_model(cfg["name"], cfg["think"])
        out_path = run_rag_for_model(
            cfg, test_df, concept_index, code_index,
            limit=args.limit,
            resume=not args.no_resume,
        )
        print(f"  Results: {out_path}")

        # Per-model quick summary
        results = []
        with open(out_path) as f:
            for line in f:
                results.append(json.loads(line))
        summary = aggregate_scores(results)
        print(f"\n  Accuracy summary:")
        print(summary.to_string())

    # Build and print cross-model comparison
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON (RAG method)")
    print("=" * 70)

    table = build_comparison_table(models_to_run)
    if table.empty:
        print("No completed results to compare yet.")
        return

    csv_path = OUTPUT_DIR / "llm_comparison_table.csv"
    table.to_csv(csv_path)
    print(f"Full table saved to {csv_path}")

    print_metric_section(table, "exact_acc",        "Exact Match Accuracy")
    print_metric_section(table, "one_level_up_acc", "One-Level-Up Accuracy")
    print_metric_section(table, "component_acc",    "Component-Level Accuracy")


if __name__ == "__main__":
    main()
