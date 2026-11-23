"""
Figure 2 (manuscript): top ICF first-level chapters in EHR Uni goals (RAG annotation).

Same plot as 11_uke_results.ipynb cell 11, read from output/uke_annotations.csv.

Usage:
  python scripts/15_fig_ehr_chapter_prevalence.py
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent
OUT_FIG = REPO / "papers" / "figures" / "fig4_uke_chapter_prevalence.png"

CHAPTER_LABELS = {
    "b1": "b1 Mental", "b2": "b2 Sensory/Pain", "b3": "b3 Voice/Speech",
    "b4": "b4 Cardiovascular", "b5": "b5 Digestive", "b6": "b6 Urogenital",
    "b7": "b7 Neuromusculoskeletal", "b8": "b8 Skin",
    "s1": "s1 Nervous", "s2": "s2 Eye/Ear", "s3": "s3 Voice",
    "s4": "s4 Cardiovascular", "s5": "s5 Digestive", "s6": "s6 Urogenital",
    "s7": "s7 Movement", "s8": "s8 Skin",
    "d1": "d1 Learning", "d2": "d2 General Tasks", "d3": "d3 Communication",
    "d4": "d4 Mobility", "d5": "d5 Self-care", "d6": "d6 Domestic Life",
    "d7": "d7 Interpersonal", "d8": "d8 Life Areas", "d9": "d9 Community",
    "e1": "e1 Products", "e2": "e2 Environment", "e3": "e3 Support",
    "e4": "e4 Attitudes", "e5": "e5 Services",
}
COLORS = {"d": "#4C72B0", "b": "#DD8452", "e": "#55A868", "s": "#C44E52"}


def main():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    df = pd.read_csv(REPO / "output" / "uke_annotations.csv")
    chap = df.loc[df["chapter_rag"] != "?", "chapter_rag"]
    n = len(chap)
    top = chap.value_counts().head(12)

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar([CHAPTER_LABELS.get(c, c) for c in top.index], top.values,
                  color=[COLORS[c[0]] for c in top.index], edgecolor="white")
    for bar, v in zip(bars, top.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v}\n({v / n * 100:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Number of EHR Uni goal statements")
    ax.set_xlabel("ICF first-level chapter")
    ax.set_title(f"ICF first-level chapters (chapter level) in EHR Uni goal statements (llama4 + RAG, n = {n})", fontweight="bold")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"n valid = {n}")
    print((top / n * 100).round(1).to_string())
    print(f"Saved → {OUT_FIG}")


if __name__ == "__main__":
    main()
