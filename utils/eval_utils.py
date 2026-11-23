"""
Three-level hierarchical evaluation metrics for ICF code prediction.

  exact_match      - predicted code is in gold set (string equality)
  one_level_up     - 4-char prefix of predicted matches 4-char prefix of any gold code
  component_match  - first character (b/s/d/e) matches any gold code's component

Non-standard codes (nd, nc, pf, hc, nd-*) are only exact-matched.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.metrics import cohen_kappa_score


NON_STANDARD = {
    "nd", "nd-gh", "nd-b", "nd-d", "nd-e", "nd-s", "nd-mh", "nd-qol",
    "nc", "pf", "hc",
}
STANDARD_ICF_RE = re.compile(r"^[bsde]\d{3,}")


def _component(code: str) -> str | None:
    """Return 'b', 's', 'd', or 'e' for standard codes; None for non-standard."""
    code = code.lower().strip()
    if code in NON_STANDARD:
        return None
    if re.match(r"^[bsde]", code):
        return code[0]
    return None


def _prefix4(code: str) -> str | None:
    """Return 4-char prefix (e.g. 'd450') for standard codes with ≥4 chars."""
    code = code.lower().strip()
    if code in NON_STANDARD:
        return None
    if STANDARD_ICF_RE.match(code) and len(code) >= 4:
        return code[:4]
    return None


def exact_match(predicted: str, gold_codes: list[str]) -> bool:
    pred = predicted.lower().strip()
    return pred in [g.lower().strip() for g in gold_codes]


def one_level_up_match(predicted: str, gold_codes: list[str]) -> bool:
    """True if the 4-char prefix of predicted matches that of any gold code."""
    pred_p = _prefix4(predicted)
    if pred_p is None:
        return exact_match(predicted, gold_codes)
    gold_prefixes = {_prefix4(g) for g in gold_codes}
    gold_prefixes.discard(None)
    return pred_p in gold_prefixes


def component_match(predicted: str, gold_codes: list[str]) -> bool:
    """True if the component (b/s/d/e) of predicted matches any gold code."""
    pred_c = _component(predicted)
    if pred_c is None:
        return exact_match(predicted, gold_codes)
    gold_components = {_component(g) for g in gold_codes}
    gold_components.discard(None)
    return pred_c in gold_components


def score_prediction(
    predicted_codes: list[str],
    gold_codes: list[str],
) -> dict[str, bool]:
    """
    Multi-label scoring: a row is correct at each level if AT LEAST ONE predicted
    code satisfies the criterion against ANY gold code.
    """
    return {
        "exact": any(exact_match(p, gold_codes) for p in predicted_codes),
        "one_level_up": any(one_level_up_match(p, gold_codes) for p in predicted_codes),
        "component": any(component_match(p, gold_codes) for p in predicted_codes),
    }


def _primary_standard_component(codes: list[str]) -> str | None:
    """Return component of the first standard code in a list; None if all non-standard."""
    for code in codes:
        c = _component(code)
        if c is not None:
            return c
    return None


def compute_kappa(results: list[dict]) -> dict:
    """
    Compute Cohen's Kappa at the component level between predicted and gold codes.

    Rules:
    - Gold label  = component of the first standard gold code (b/s/d/e).
    - Pred label  = component of the first standard predicted code; 'x' if all non-standard.
    - Rows where gold has no standard code are excluded (non-standard qualifiers only).

    Returns a dict with overall kappa and per-cohort kappa values.
    """
    gold_labels, pred_labels, cohorts = [], [], []

    for rec in results:
        g = _primary_standard_component(rec["gold_codes"])
        if g is None:
            continue  # skip rows with only non-standard gold codes
        p = _primary_standard_component(rec.get("predicted_codes", []))
        if p is None:
            p = "x"  # model returned only non-standard or empty
        gold_labels.append(g)
        pred_labels.append(p)
        cohorts.append(rec.get("cohort", "unknown"))

    if not gold_labels:
        return {"overall": None, "per_cohort": {}}

    overall_kappa = cohen_kappa_score(gold_labels, pred_labels)

    per_cohort: dict[str, float | None] = {}
    df = pd.DataFrame({"gold": gold_labels, "pred": pred_labels, "cohort": cohorts})
    for cohort, grp in df.groupby("cohort"):
        if len(grp["gold"].unique()) < 2 or len(grp) < 2:
            per_cohort[cohort] = None
        else:
            per_cohort[cohort] = cohen_kappa_score(grp["gold"], grp["pred"])

    return {"overall": overall_kappa, "n": len(gold_labels), "per_cohort": per_cohort}


def aggregate_scores(results: list[dict]) -> pd.DataFrame:
    """
    Given a list of per-row result dicts (each with 'cohort', 'exact', 'one_level_up',
    'component'), return a summary DataFrame with per-cohort and overall stats.
    """
    df = pd.DataFrame(results)
    rows = []

    for cohort, grp in df.groupby("cohort"):
        rows.append({
            "cohort": cohort,
            "n": len(grp),
            "exact_acc": grp["exact"].mean(),
            "one_level_up_acc": grp["one_level_up"].mean(),
            "component_acc": grp["component"].mean(),
        })

    # Overall
    rows.append({
        "cohort": "ALL",
        "n": len(df),
        "exact_acc": df["exact"].mean(),
        "one_level_up_acc": df["one_level_up"].mean(),
        "component_acc": df["component"].mean(),
    })

    summary = pd.DataFrame(rows).set_index("cohort")
    for col in ["exact_acc", "one_level_up_acc", "component_acc"]:
        summary[col] = summary[col].map(lambda x: f"{x*100:.1f}%")
    return summary
