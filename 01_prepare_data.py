"""
Step 1: Load all four cohorts, create 80/20 per-cohort train/test splits,
and save them to output/train.jsonl and output/test.jsonl.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import (
    load_all_cohorts,
    train_test_split_per_cohort,
    save_splits,
)


def main():
    print("Loading all cohorts...")
    df = load_all_cohorts()
    print(f"  Total rows: {len(df)}")
    print(df["cohort"].value_counts().to_string())

    print("\nSplitting 80/20 per cohort (stratified by primary ICF code)...")
    train_df, test_df = train_test_split_per_cohort(df, test_size=0.20, random_state=42)

    print("\nTrain distribution:")
    print(train_df["cohort"].value_counts().to_string())
    print("\nTest distribution:")
    print(test_df["cohort"].value_counts().to_string())

    print("\nUnique codes in train pool:", train_df["primary_code"].nunique())
    print("Unique codes in test set: ", test_df["primary_code"].nunique())

    save_splits(train_df, test_df)
    print("\nDone. Files written to output/train.jsonl and output/test.jsonl")


if __name__ == "__main__":
    main()
