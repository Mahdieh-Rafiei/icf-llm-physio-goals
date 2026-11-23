"""
Run the full ICF prediction pipeline:
  Step 1: Prepare data splits
  Step 2: Build embedding index        (rebuild after changing EMBED_MODEL)
  Step 3: Run zero-shot linking        (Option A)
  Step 4: Run few-shot linking         (Option B)
  Step 5: Run RAG-based linking        (Option C, default LLM)
  Step 6: Compare and evaluate A/B/C
  Step 7: Cross-cohort consistency audit
  Step 8: LLM model comparison         (RAG with 4 LLMs side-by-side)

Usage:
  python scripts/run_all.py                       # full run
  python scripts/run_all.py --limit 10            # 10 test rows (smoke test)
  python scripts/run_all.py --skip-embed          # skip index rebuild (already built)
  python scripts/run_all.py --options a c         # run only specific linking options
  python scripts/run_all.py --skip-audit          # skip consistency audit
  python scripts/run_all.py --skip-compare        # skip LLM model comparison
  python scripts/run_all.py --compare-models qwen3.6:35b deepseek-r1:32b
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N test rows per option")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip embedding index build (use existing output/*.pkl)")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip data preparation (use existing output/*.jsonl)")
    parser.add_argument("--options", nargs="+", choices=["a", "b", "c"],
                        default=["a", "b", "c"],
                        help="Which linking options to run (default: all)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Restart from scratch (ignore existing results)")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Skip the cross-cohort consistency audit")
    parser.add_argument("--audit-threshold", type=float, default=0.88,
                        help="Similarity threshold for consistency audit (default: 0.88)")
    parser.add_argument("--skip-compare", action="store_true",
                        help="Skip the LLM model comparison step")
    parser.add_argument("--compare-models", nargs="+", default=None,
                        help="Subset of model names for comparison (default: all 4)")
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    python = sys.executable

    if not args.skip_prepare:
        run([python, str(scripts_dir / "01_prepare_data.py")])

    if not args.skip_embed and "c" in args.options:
        run([python, str(scripts_dir / "02_build_index.py")])

    limit_args = ["--limit", str(args.limit)] if args.limit else []
    resume_args = ["--no-resume"] if args.no_resume else []

    if "a" in args.options:
        run([python, str(scripts_dir / "03_zeroshot.py")] + limit_args + resume_args)

    if "b" in args.options:
        run([python, str(scripts_dir / "04_fewshot.py")] + limit_args + resume_args)

    if "c" in args.options:
        run([python, str(scripts_dir / "05_rag.py")] + limit_args + resume_args)

    run([python, str(scripts_dir / "06_evaluate.py")])

    if not args.skip_audit:
        run([
            python, str(scripts_dir / "07_consistency_audit.py"),
            "--threshold", str(args.audit_threshold),
        ])

    if not args.skip_compare:
        compare_cmd = [python, str(scripts_dir / "08_llm_comparison.py")]
        if args.limit:
            compare_cmd += ["--limit", str(args.limit)]
        if args.no_resume:
            compare_cmd += ["--no-resume"]
        if args.compare_models:
            compare_cmd += ["--models"] + args.compare_models
        run(compare_cmd)


if __name__ == "__main__":
    main()
