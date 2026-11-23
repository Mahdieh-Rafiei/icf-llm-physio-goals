"""
Option D: Cross-cohort Consistency Audit.

Takes all ~2,500 German "Konzept" entries from all four cohorts (no train/test
split — this uses the full dataset) and groups them by semantic similarity
using multilingual-e5-large-instruct embeddings.

Within each group of similar concepts, it checks whether the human annotators
assigned the same ICF codes. Disagreements are flagged and ranked by severity.

Two classes of findings:
  1. Cross-cohort inconsistency: same concept, different codes, different study.
     May reveal genuine methodological differences between annotation teams.
  2. Within-cohort inconsistency: same concept appears twice in the same
     study with different codes.
     Likely annotation error — high-confidence candidates for cleanup.

Outputs:
  output/consistency_audit.jsonl   — one record per inconsistent group
  output/consistency_audit_report.txt — human-readable ranked summary

Requires: embedding model (downloaded on first run) or existing index.
          Run 02_build_index.py first to reuse the model download.

Usage:
  python scripts/07_consistency_audit.py
  python scripts/07_consistency_audit.py --threshold 0.90   # stricter grouping
  python scripts/07_consistency_audit.py --top-n 50         # report top-50 only
"""

import sys
import json
import argparse
import textwrap
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_all_cohorts, OUTPUT_DIR, build_icf_lookup, load_icf_reference
from utils.eval_utils import (
    exact_match, one_level_up_match, component_match, NON_STANDARD
)
from utils.embed_utils import embed_texts, EMBED_MODEL

# Similarity threshold above which two concepts are considered "same meaning"
DEFAULT_THRESHOLD = 0.88
DEFAULT_TOP_N = 100


# ─── Grouping ───────────────────────────────────────────────────────────────

def find_similar_groups(
    embeddings: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """
    Graph-based connected-component grouping.
    Two concepts are connected if their cosine similarity >= threshold.
    Returns a list of groups (each group = list of row indices).
    Only returns groups with ≥2 members.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(embeddings)
    print(f"  Computing {n}×{n} cosine similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0.0)  # exclude self-similarity

    print(f"  Finding connected components (threshold={threshold})...")
    # BFS over adjacency implied by the threshold
    adjacency = sim_matrix >= threshold
    visited = np.zeros(n, dtype=bool)
    groups = []

    for start in range(n):
        if visited[start]:
            continue
        queue = [start]
        component = []
        while queue:
            node = queue.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            neighbors = np.where(adjacency[node])[0]
            for nb in neighbors:
                if not visited[nb]:
                    queue.append(nb)
        if len(component) >= 2:
            groups.append(sorted(component))

    return groups


# ─── Severity scoring ────────────────────────────────────────────────────────

def _code_set(row_codes: list[str]) -> set[str]:
    return set(c.lower().strip() for c in row_codes)


def disagreement_level(rows: list[dict]) -> str:
    """
    Classify the type of code disagreement across a group.
      'none'       — all identical code sets
      'qualifier'  — all are non-standard qualifiers (nd/nc/pf/hc) but differ
      'granularity'— same component+chapter but different depth (e.g. d450 vs d4500)
      'chapter'    — same component (b/d/s/e) but different chapter
      'component'  — different components (b vs d etc.)
      'mixed'      — mix of standard and non-standard
    """
    code_sets = [_code_set(r["gold_codes"]) for r in rows]

    # All identical
    if len(set(frozenset(cs) for cs in code_sets)) == 1:
        return "none"

    all_primary = [r["primary_code"] for r in rows]
    all_nonstd = all(c in NON_STANDARD for c in all_primary)
    any_nonstd = any(c in NON_STANDARD for c in all_primary)
    any_std = any(c not in NON_STANDARD for c in all_primary)

    if any_nonstd and any_std:
        return "mixed"
    if all_nonstd:
        return "qualifier"

    # All standard codes: compare at component and chapter level
    components = set(c[0] for c in all_primary if c and c[0] in "bsde")
    if len(components) > 1:
        return "component"

    # Same component — check chapter (first two chars)
    chapters = set(c[:2] for c in all_primary if len(c) >= 2)
    if len(chapters) > 1:
        return "chapter"

    return "granularity"


SEVERITY_ORDER = ["component", "mixed", "chapter", "granularity", "qualifier", "none"]


def severity_score(level: str) -> int:
    """Higher score = more severe disagreement."""
    try:
        return len(SEVERITY_ORDER) - SEVERITY_ORDER.index(level)
    except ValueError:
        return 0


def group_max_similarity(group_indices: list[int], embeddings: np.ndarray) -> float:
    """Maximum pairwise cosine similarity within the group."""
    from sklearn.metrics.pairwise import cosine_similarity
    vecs = embeddings[group_indices]
    sim = cosine_similarity(vecs)
    np.fill_diagonal(sim, 0.0)
    return float(sim.max())


# ─── Reporting ───────────────────────────────────────────────────────────────

def build_group_record(
    group_rows: list[dict],
    max_sim: float,
    icf_lookup: dict[str, str],
) -> dict:
    """Assemble a single group record for JSONL output."""
    level = disagreement_level(group_rows)
    cohorts = [r["cohort"] for r in group_rows]
    cross_cohort = len(set(cohorts)) > 1

    members = []
    for r in group_rows:
        codes_with_titles = [
            {"code": c, "title": icf_lookup.get(c, "(non-standard or unknown)")}
            for c in r["gold_codes"]
        ]
        members.append({
            "konzept": r["konzept"],
            "cohort": r["cohort"],
            "gold_codes": r["gold_codes"],
            "codes_with_titles": codes_with_titles,
        })

    return {
        "n_members": len(group_rows),
        "max_similarity": round(max_sim, 4),
        "disagreement_level": level,
        "severity_score": severity_score(level),
        "cross_cohort": cross_cohort,
        "cohorts": cohorts,
        "members": members,
    }


def format_group_text(record: dict, rank: int, icf_lookup: dict[str, str]) -> str:
    """Format one group as readable text for the report file."""
    header = (
        f"{'─'*70}\n"
        f"Group #{rank} | severity: {record['disagreement_level'].upper()} | "
        f"sim={record['max_similarity']:.3f} | "
        f"{'CROSS-COHORT' if record['cross_cohort'] else 'WITHIN-COHORT'} | "
        f"n={record['n_members']}\n"
    )
    lines = [header]
    for m in record["members"]:
        codes_str = ", ".join(
            f"{c['code']} ({c['title'][:40]})"
            for c in m["codes_with_titles"]
        )
        lines.append(f"  [{m['cohort']:10s}] {m['konzept'][:60]:<60}  →  {codes_str}")
    return "\n".join(lines) + "\n"


# ─── Main ────────────────────────────────────────────────────────────────────

def run_consistency_audit(threshold: float = DEFAULT_THRESHOLD, top_n: int = DEFAULT_TOP_N):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_jsonl = OUTPUT_DIR / "consistency_audit.jsonl"
    audit_report = OUTPUT_DIR / "consistency_audit_report.txt"

    print("Loading all cohorts (full dataset, no split)...")
    df = load_all_cohorts()
    print(f"  Total concepts: {len(df)}")
    print(f"  Cohort breakdown: {df['cohort'].value_counts().to_dict()}")

    print("\nLoading ICF reference for title lookup...")
    icf_df = load_icf_reference()
    icf_lookup = build_icf_lookup(icf_df)

    print(f"\nEmbedding model: {EMBED_MODEL}")
    konzepte = df["konzept"].tolist()
    print(f"Embedding {len(konzepte)} concepts via Ollama...")
    embeddings = embed_texts(konzepte, as_query=False, show_progress=True)
    print(f"  Embeddings shape: {embeddings.shape}")

    rows = df.to_dict(orient="records")

    print(f"\nFinding groups with cosine similarity ≥ {threshold}...")
    groups = find_similar_groups(embeddings, threshold)
    print(f"  Total groups (≥2 members): {len(groups)}")

    # Score and filter to inconsistent groups
    print("\nScoring groups for code disagreement...")
    inconsistent_records = []

    for grp_indices in tqdm(groups):
        group_rows = [rows[i] for i in grp_indices]
        level = disagreement_level(group_rows)
        if level == "none":
            continue  # consistent → skip

        # Compute max pairwise similarity for this group only (small submatrix)
        grp_embs = embeddings[grp_indices]
        from sklearn.metrics.pairwise import cosine_similarity as cs
        sub_sim = cs(grp_embs)
        np.fill_diagonal(sub_sim, 0.0)
        max_sim = float(sub_sim.max())

        record = build_group_record(group_rows, max_sim, icf_lookup)
        inconsistent_records.append(record)

    # Sort: severity first, then max_similarity (higher = more suspicious)
    inconsistent_records.sort(
        key=lambda r: (r["severity_score"], r["max_similarity"]),
        reverse=True,
    )

    print(f"\nFound {len(inconsistent_records)} inconsistent groups.")

    # Breakdown by type
    level_counts = defaultdict(int)
    cross_count = sum(1 for r in inconsistent_records if r["cross_cohort"])
    for r in inconsistent_records:
        level_counts[r["disagreement_level"]] += 1

    print("\nDisagreement breakdown:")
    for level in SEVERITY_ORDER:
        if level_counts[level]:
            print(f"  {level:12s}: {level_counts[level]}")
    print(f"  Cross-cohort: {cross_count} / {len(inconsistent_records)}")

    # Save JSONL
    with open(audit_jsonl, "w") as f:
        for rec in inconsistent_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nFull results saved to {audit_jsonl}")

    # Save text report
    top_records = inconsistent_records[:top_n]
    report_lines = [
        "=" * 70,
        "CROSS-COHORT CONSISTENCY AUDIT REPORT",
        f"Threshold: {threshold} | Total inconsistent groups: {len(inconsistent_records)}",
        f"Cross-cohort: {cross_count} | Within-cohort: {len(inconsistent_records) - cross_count}",
        "=" * 70,
        "",
        "Severity levels (worst → best):",
        "  COMPONENT  — disagreeing ICF component (b vs d vs s vs e)",
        "  MIXED      — mix of standard ICF code + non-standard qualifier (nd/nc/pf)",
        "  CHAPTER    — same component, different chapter (e.g. b1xx vs b7xx)",
        "  GRANULARITY— same chapter, different specificity (e.g. d450 vs d4500)",
        "  QUALIFIER  — both non-standard but different qualifier (nd vs nc vs pf)",
        "",
        f"Top {len(top_records)} most suspicious groups (ranked by severity, then similarity):",
        "",
    ]

    for rank, rec in enumerate(top_records, 1):
        report_lines.append(format_group_text(rec, rank, icf_lookup))

    report_text = "\n".join(report_lines)
    with open(audit_report, "w") as f:
        f.write(report_text)
    print(f"Human-readable report (top {len(top_records)}) saved to {audit_report}")

    # Print top-10 to stdout
    print("\n" + "=" * 70)
    print(f"TOP 10 MOST SUSPICIOUS INCONSISTENCIES (threshold={threshold})")
    print("=" * 70)
    for rank, rec in enumerate(inconsistent_records[:10], 1):
        print(format_group_text(rec, rank, icf_lookup))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-cohort consistency audit for ICF annotations"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for grouping (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--top-n", type=int, default=DEFAULT_TOP_N,
        help=f"Number of groups to include in text report (default: {DEFAULT_TOP_N})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing audit results (default: refuse to overwrite)"
    )
    args = parser.parse_args()

    audit_jsonl = OUTPUT_DIR / "consistency_audit.jsonl"
    if audit_jsonl.exists() and not args.force:
        print(f"Audit results already exist at {audit_jsonl}")
        print("Use --force to overwrite. The existing numbers are the authoritative source for the manuscript.")
        import sys; sys.exit(0)

    run_consistency_audit(threshold=args.threshold, top_n=args.top_n)
