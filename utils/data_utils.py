"""
Data loading, normalization, and splitting utilities.
All four cohorts share a unified schema after loading.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).parent.parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUTPUT_DIR = REPO_ROOT / "output"

# Non-standard qualifier codes used by annotators
NON_STANDARD = {
    "nd", "nd-gh", "nd-b", "nd-d", "nd-e", "nd-s", "nd-mh", "nd-qol",
    "nc", "pf", "hc",
}
# Bare chapter-level component codes (e.g. "b", "d", "e", "s", "s2", "b1")
BARE_COMPONENT_RE = re.compile(r"^[bdse]\d{0,2}$")

_COHORT_CONFIG = {
    "Akut": {
        "file": "Akut_complete.xlsx",
        "sheet": "Akut",
        "link_cols": ["ICF_Link1", "ICF_Link2", "ICF_Link3"],
    },
    "Geriat": {
        "file": "Geriat_complete.xlsx",
        "sheet": "Geratrie",
        "link_cols": ["ICF_Link"],
    },
    "PostAkut": {
        "file": "PostAkut_complete.xlsx",
        "sheet": "PostAkut",
        "link_cols": ["ICF_Link1", "ICF_Link2", "ICF_Link3", "ICF_Link4"],
    },
    "Vertigo": {
        "file": "Vertigo_complete.xlsx",
        "sheet": "vertigo",
        "link_cols": ["ICF Link"],
    },
}


def normalize_code(code: str) -> str:
    """Lowercase and strip whitespace; leave valid non-standard codes as-is."""
    return str(code).strip().lower()


def load_cohort(cohort: str) -> pd.DataFrame:
    """
    Load one cohort into a DataFrame with columns:
      - konzept: str
      - gold_codes: list[str]  (all non-null ICF codes for the row)
      - primary_code: str      (first non-null code, for stratification)
      - cohort: str
    """
    cfg = _COHORT_CONFIG[cohort]
    path = DATASET_DIR / cfg["file"]
    df = pd.read_excel(path, sheet_name=cfg["sheet"])

    rows = []
    for _, row in df.iterrows():
        konzept = str(row["Konzept"]).strip()
        if not konzept or konzept.lower() == "nan":
            continue

        codes = []
        for col in cfg["link_cols"]:
            val = row.get(col)
            if pd.notna(val) and str(val).strip().lower() not in ("nan", ""):
                codes.append(normalize_code(str(val)))

        if not codes:
            codes = ["nd"]  # unlinkable → treat as nd

        rows.append({
            "konzept": konzept,
            "gold_codes": codes,
            "primary_code": codes[0],
            "cohort": cohort,
        })

    return pd.DataFrame(rows)


def load_all_cohorts() -> pd.DataFrame:
    """Concatenate all four cohorts into a single DataFrame."""
    dfs = [load_cohort(c) for c in _COHORT_CONFIG]
    combined = pd.concat(dfs, ignore_index=True)
    combined["idx"] = combined.index
    return combined


def train_test_split_per_cohort(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    80/20 split within each cohort, stratified by primary_code where possible.
    Codes with only one sample are forced into the training set.
    Returns (train_df, test_df).
    """
    train_parts, test_parts = [], []

    for cohort, grp in df.groupby("cohort"):
        code_counts = grp["primary_code"].value_counts()
        singleton_mask = grp["primary_code"].isin(
            code_counts[code_counts == 1].index
        )

        singletons = grp[singleton_mask]
        stratifiable = grp[~singleton_mask]

        if len(stratifiable) < 2:
            # Not enough data to split — everything goes to train
            train_parts.append(grp)
            continue

        try:
            tr, te = train_test_split(
                stratifiable,
                test_size=test_size,
                stratify=stratifiable["primary_code"],
                random_state=random_state,
            )
        except ValueError:
            tr, te = train_test_split(
                stratifiable,
                test_size=test_size,
                random_state=random_state,
            )

        train_parts.append(pd.concat([tr, singletons]))
        test_parts.append(te)

    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def load_icf_reference() -> pd.DataFrame:
    """Load BfArM ICF 2005 reference (1 454 codes with German titles)."""
    return pd.read_csv(DATASET_DIR / "icf_codes_reference.csv")


def build_icf_lookup(icf_df: pd.DataFrame) -> dict[str, str]:
    """Return {code: title_de} mapping."""
    return dict(zip(icf_df["code"].str.lower(), icf_df["title_de"]))


def icf_list_text(icf_df: pd.DataFrame) -> str:
    """Format full ICF code list as 'code - title' lines for LLM prompts."""
    lines = [f"{row['code']} - {row['title_de']}" for _, row in icf_df.iterrows()]
    return "\n".join(lines)


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_json(OUTPUT_DIR / "train.jsonl", orient="records", lines=True, force_ascii=False)
    test_df.to_json(OUTPUT_DIR / "test.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(train_df)} train / {len(test_df)} test rows")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    def _load(path):
        df = pd.read_json(path, orient="records", lines=True)
        df["gold_codes"] = df["gold_codes"].apply(
            lambda x: x if isinstance(x, list) else json.loads(x)
        )
        return df

    return _load(OUTPUT_DIR / "train.jsonl"), _load(OUTPUT_DIR / "test.jsonl")
