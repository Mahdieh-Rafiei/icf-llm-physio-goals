"""
Step 2: Embed all training concepts and ICF code descriptions using
the model set in EMBED_MODEL (.env) — currently qwen3-embedding:8b.
Saves two index files to output/.

NOTE: If you change EMBED_MODEL, delete output/*.pkl first and re-run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import load_splits, load_icf_reference, OUTPUT_DIR
from utils.embed_utils import build_concept_index, build_icf_code_index, EMBED_MODEL


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Embedding model: {EMBED_MODEL}")

    print("\nLoading train/test splits...")
    train_df, test_df = load_splits()
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    print("\nLoading ICF reference codes...")
    icf_df = load_icf_reference()
    print(f"  ICF codes: {len(icf_df)}")

    print("\n--- Building concept index (training examples) ---")
    concept_idx = build_concept_index(train_df)
    concept_path = OUTPUT_DIR / "concept_index.pkl"
    concept_idx.save(concept_path)
    print(f"  Saved → {concept_path}")

    print("\n--- Building ICF code index ---")
    code_idx = build_icf_code_index(icf_df)
    code_path = OUTPUT_DIR / "icf_code_index.pkl"
    code_idx.save(code_path)
    print(f"  Saved → {code_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
