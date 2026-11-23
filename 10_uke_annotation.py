"""
Step 10: Annotate UKE patient goal concepts with ICF component and chapter.

Uses llama4 (best model from Phase 1) to assign:
  - ICF component: b, s, d, or e
  - ICF chapter:   b1-b8, s1-s8, d1-d9, e1-e5 (30 chapters total)
  - Confidence:    high / medium / low

Two conditions run on all 529 UKE concepts:
  - Zero-shot: llama4 with component/chapter taxonomy only
  - RAG:       llama4 + top-6 retrieved examples from 4-cohort training data

Output:
  output/uke_annotations.csv     — full annotation table
  output/fig_uke_prevalence.png  — component + chapter distribution chart

Usage:
  python scripts/10_uke_annotation.py
  python scripts/10_uke_annotation.py --model "qwen2.5:72b"
  python scripts/10_uke_annotation.py --limit 20    # smoke test
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, OUTPUT_DIR
from utils.llm_utils import call_llm
from utils.embed_utils import ConceptIndex, query_concept_index

# ── ICF taxonomy ────────────────────────────────────────────────────────────
ICF_TAXONOMY = """ICF Components and Chapters:
Body Functions (b):
  b1=Mental functions, b2=Sensory functions and pain, b3=Voice and speech functions,
  b4=Cardiovascular/haematological/immune/respiratory functions,
  b5=Digestive/metabolic/endocrine functions, b6=Urogenital/reproductive functions,
  b7=Neuromusculoskeletal and movement-related functions, b8=Skin functions

Body Structures (s):
  s1=Nervous system, s2=Eye/ear and related structures, s3=Voice and speech structures,
  s4=Cardiovascular/immune/respiratory structures, s5=Digestive/metabolic/endocrine structures,
  s6=Urogenital/reproductive structures, s7=Movement structures, s8=Skin structures

Activities & Participation (d):
  d1=Learning and knowledge application, d2=General tasks and requirements,
  d3=Communication, d4=Mobility, d5=Self-care, d6=Domestic life,
  d7=Interpersonal interactions and relationships, d8=Major life areas,
  d9=Community/social/civic life

Environmental Factors (e):
  e1=Products and technology, e2=Natural and human-modified environment,
  e3=Support and relationships, e4=Attitudes, e5=Services/systems/policies"""

VALID_COMPONENTS = {"b", "s", "d", "e"}
VALID_CHAPTERS = {
    f"{c}{n}"
    for c, rng in [("b", range(1, 9)), ("s", range(1, 9)),
                   ("d", range(1, 10)), ("e", range(1, 6))]
    for n in rng
}

DATASET_PATH = Path(__file__).parent.parent / "dataset" / "UKE.csv"
TOP_K_EXAMPLES = 6


def build_zeroshot_prompt(konzept: str) -> str:
    return f"""You are an expert in the ICF (International Classification of Functioning, Disability and Health).

Assign the most appropriate ICF component and first-level chapter to this German rehabilitation goal concept.

{ICF_TAXONOMY}

Concept: "{konzept}"

Rules:
- Choose exactly ONE component (b, s, d, or e)
- Choose exactly ONE chapter (e.g. d4, b7, e3)
- Rate your confidence as high, medium, or low

Respond ONLY with this JSON:
{{"component": "d", "chapter": "d4", "confidence": "high"}}"""


def build_rag_prompt(konzept: str, examples: list[dict]) -> str:
    ex_lines = []
    for i, ex in enumerate(examples, 1):
        codes = ", ".join(ex["gold_codes"])
        # derive chapter from first code (first 2 chars if alphanumeric)
        first = ex["gold_codes"][0] if ex["gold_codes"] else ""
        chapter_hint = first[:2] if len(first) >= 2 and first[1].isdigit() else first[:1]
        comp_hint = first[0] if first else "?"
        ex_lines.append(
            f"Example {i} (similarity {ex['similarity']:.2f}):\n"
            f"  Concept: \"{ex['konzept']}\"\n"
            f"  ICF codes: {codes}  →  component={comp_hint}, chapter={chapter_hint}"
        )
    examples_block = "\n\n".join(ex_lines)

    return f"""You are an expert in the ICF (International Classification of Functioning, Disability and Health).

Assign the most appropriate ICF component and first-level chapter to this German rehabilitation goal concept.

{ICF_TAXONOMY}

--- Similar expert-annotated examples (for guidance) ---
{examples_block}
--- End of examples ---

Concept: "{konzept}"

Rules:
- Choose exactly ONE component (b, s, d, or e)
- Choose exactly ONE chapter (e.g. d4, b7, e3)
- Rate your confidence as high, medium, or low

Respond ONLY with this JSON:
{{"component": "d", "chapter": "d4", "confidence": "high"}}"""


def parse_annotation(result: dict) -> tuple[str, str, str]:
    """Extract and validate component, chapter, confidence from LLM result."""
    comp  = str(result.get("component", "")).strip().lower()
    chap  = str(result.get("chapter", "")).strip().lower()
    conf  = str(result.get("confidence", "")).strip().lower()

    if comp not in VALID_COMPONENTS:
        # try to infer from chapter
        comp = chap[0] if chap and chap[0] in VALID_COMPONENTS else "?"
    if chap not in VALID_CHAPTERS:
        chap = "?"
    if conf not in ("high", "medium", "low"):
        conf = "low"

    return comp, chap, conf


def run_annotation(
    df: pd.DataFrame,
    model: str,
    condition: str,
    concept_index,
    out_path: Path,
    resume: bool = True,
) -> pd.DataFrame:
    """Run one annotation condition (zeroshot or rag) on all UKE concepts."""

    done_ids: set[int] = set()
    if resume and out_path.exists():
        done = pd.read_csv(out_path)
        done_ids = set(done["row_id"].tolist())
        print(f"  Resuming: {len(done_ids)} rows already done.")

    rows_to_run = df[~df["row_id"].isin(done_ids)] if done_ids else df
    print(f"  Running {len(rows_to_run)} rows...")

    results = []
    with open(out_path, "a" if (resume and done_ids) else "w") as f_out:
        for _, row in tqdm(rows_to_run.iterrows(), total=len(rows_to_run)):
            konzept = row["set_goal"]
            row_id  = int(row["row_id"])

            if condition == "zeroshot":
                prompt = build_zeroshot_prompt(konzept)
                result = call_llm(prompt, model=model, think=False)
            else:  # rag
                examples = query_concept_index(konzept, concept_index, top_k=TOP_K_EXAMPLES)
                prompt   = build_rag_prompt(konzept, examples)
                result   = call_llm(prompt, model=model, think=False)

            comp, chap, conf = parse_annotation(result)

            record = {
                "row_id":    row_id,
                "set_goal":  konzept,
                "component": comp,
                "chapter":   chap,
                "confidence": conf,
                "raw_reasoning": result.get("reasoning", "")[:200],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            results.append(record)

    # Load all results (done + new)
    all_rows = []
    with open(out_path) as f:
        for line in f:
            all_rows.append(json.loads(line))
    return pd.DataFrame(all_rows)


def prevalence_analysis(
    df_zero: pd.DataFrame,
    df_rag:  pd.DataFrame,
    train_df_4cohort: pd.DataFrame,
) -> None:
    """Plot component and chapter prevalence for UKE vs 4-cohort data."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── helper: component counts ──────────────────────────────────────────
    def comp_counts(series):
        return series.value_counts().reindex(["b", "s", "d", "e"], fill_value=0)

    # derive component from 4-cohort primary code
    train_comps = train_df_4cohort["primary_code"].str[0].str.lower()
    train_comps = train_comps[train_comps.isin(["b", "s", "d", "e"])]

    comp_labels = {"b": "Body\nFunctions", "s": "Body\nStructures",
                   "d": "Activities &\nParticipation", "e": "Environmental\nFactors"}

    datasets = [
        ("UKE — Zero-shot", df_zero["component"]),
        ("UKE — RAG",       df_rag["component"]),
        ("4-cohort train",  train_comps),
    ]

    colors = ["#4C72B0", "#55A868", "#DD8452"]

    for ax, (title, ser), color in zip(axes[0], datasets, colors):
        counts = comp_counts(ser)
        pct = counts / counts.sum() * 100
        bars = ax.bar([comp_labels[k] for k in counts.index], pct, color=color, edgecolor="white")
        for bar, v in zip(bars, pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)

    # ── chapter distribution (top chapters) ──────────────────────────────
    def chapter_counts(series):
        return series.value_counts().head(15)

    chapter_datasets = [
        ("UKE — Zero-shot (top chapters)", df_zero["chapter"]),
        ("UKE — RAG (top chapters)",       df_rag["chapter"]),
        ("4-cohort train (top chapters)",  train_df_4cohort["primary_code"].str[:2].str.lower()),
    ]

    for ax, (title, ser), color in zip(axes[1], chapter_datasets, colors):
        counts = chapter_counts(ser[ser != "?"])
        counts.plot(kind="bar", ax=ax, color=color, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("ICF Component & Chapter Prevalence: UKE vs 4-Cohort Training Data",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_fig = OUTPUT_DIR / "fig_uke_prevalence.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved → {out_fig}")


def print_comparison_table(df_zero: pd.DataFrame, df_rag: pd.DataFrame) -> None:
    """Print side-by-side component/chapter agreement between zero-shot and RAG."""
    print("\n=== Component Agreement: Zero-shot vs RAG ===")
    agree = (df_zero["component"] == df_rag["component"]).mean() * 100
    print(f"  Agreement on component: {agree:.1f}%")

    agree_ch = (df_zero["chapter"] == df_rag["chapter"]).mean() * 100
    print(f"  Agreement on chapter:   {agree_ch:.1f}%")

    print("\n=== Component Distribution ===")
    comp_table = pd.DataFrame({
        "Zero-shot": df_zero["component"].value_counts(normalize=True).mul(100).round(1),
        "RAG":       df_rag["component"].value_counts(normalize=True).mul(100).round(1),
    }).reindex(["d", "b", "e", "s"]).fillna(0)
    comp_table["Δ (RAG-Zero)"] = (comp_table["RAG"] - comp_table["Zero-shot"]).round(1)
    print(comp_table.to_string())

    print("\n=== Confidence Distribution (RAG) ===")
    print(df_rag["confidence"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Annotate UKE patient goals with ICF component and chapter."
    )
    parser.add_argument("--model",   default="llama4:latest",
                        help="LLM to use (default: llama4:latest)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only first N rows (smoke test)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Restart from scratch")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load UKE data
    print(f"Loading UKE data from {DATASET_PATH}...")
    uke_df = pd.read_csv(DATASET_PATH)
    uke_df = uke_df.rename(columns={"Unnamed: 0": "row_id"})
    uke_df = uke_df.dropna(subset=["set_goal"])
    if args.limit:
        uke_df = uke_df.head(args.limit)
    print(f"  {len(uke_df)} concepts loaded")

    # Load concept index (4-cohort training data for RAG)
    concept_idx_path = OUTPUT_DIR / "concept_index.pkl"
    if not concept_idx_path.exists():
        print("ERROR: concept_index.pkl not found. Run 02_build_index.py first.")
        sys.exit(1)
    concept_index = ConceptIndex.load(concept_idx_path)
    print(f"  Concept index: {len(concept_index.metadata)} training examples")

    # Load 4-cohort training data for prevalence comparison
    train_df, _ = load_splits()

    resume = not args.no_resume

    # ── Run zero-shot ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"ZERO-SHOT annotation — {args.model}")
    print("=" * 55)
    zero_path = OUTPUT_DIR / "uke_zeroshot.jsonl"
    df_zero = run_annotation(uke_df, args.model, "zeroshot", concept_index, zero_path, resume)

    # ── Run RAG ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"RAG annotation — {args.model}")
    print("=" * 55)
    rag_path = OUTPUT_DIR / "uke_rag.jsonl"
    df_rag = run_annotation(uke_df, args.model, "rag", concept_index, rag_path, resume)

    # ── Merge into single CSV ─────────────────────────────────────────────
    merged = uke_df[["row_id", "set_goal"]].merge(
        df_zero[["row_id", "component", "chapter", "confidence"]].rename(
            columns={"component": "comp_zeroshot", "chapter": "chapter_zeroshot",
                     "confidence": "conf_zeroshot"}),
        on="row_id", how="left"
    ).merge(
        df_rag[["row_id", "component", "chapter", "confidence"]].rename(
            columns={"component": "comp_rag", "chapter": "chapter_rag",
                     "confidence": "conf_rag"}),
        on="row_id", how="left"
    )

    csv_path = OUTPUT_DIR / "uke_annotations.csv"
    merged.to_csv(csv_path, index=False)
    print(f"\n✅ Annotations saved → {csv_path}")
    print(merged.head(10).to_string(index=False))

    # ── Analysis ──────────────────────────────────────────────────────────
    print_comparison_table(df_zero, df_rag)
    prevalence_analysis(df_zero, df_rag, train_df)


if __name__ == "__main__":
    main()
