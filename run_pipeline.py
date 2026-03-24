"""
RLHF Roast Generator — Full Pipeline Runner

Runs the complete RLHF pipeline end-to-end:
  1. Generate preference dataset (witty vs mean roasts)
  2. Supervised Fine-Tuning (SFT) on witty roasts
  3. Train reward model on preference pairs
  4. PPO alignment training
  5. Analysis, metrics, and overoptimization detection

Usage:
    # Run everything:
    python run_pipeline.py

    # Run specific steps:
    python run_pipeline.py --steps 1 2 3 4 5

    # Skip to analysis (if models already trained):
    python run_pipeline.py --steps 5
"""

import argparse
import os
import sys
import time

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))


def run_step(step_num, name, func):
    print("\n" + "#" * 70)
    print(f"# STEP {step_num}: {name}")
    print("#" * 70 + "\n")
    start = time.time()
    result = func()
    elapsed = time.time() - start
    print(f"\n  Step {step_num} completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="RLHF Roast Generator Pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Which steps to run (1-5). Default: all",
    )
    args = parser.parse_args()

    steps_to_run = set(args.steps)
    total_start = time.time()

    print("=" * 70)
    print("  RLHF ROAST GENERATOR - Full Pipeline")
    print("  Fine-tune GPT-2 to generate witty, clever roasts using RLHF")
    print("=" * 70)
    print(f"\nSteps to run: {sorted(steps_to_run)}")

    if 1 in steps_to_run:
        from generate_dataset import main as gen_main
        run_step(1, "Generate Preference Dataset", gen_main)

    if 2 in steps_to_run:
        from sft_train import main as sft_main
        run_step(2, "Supervised Fine-Tuning (SFT)", sft_main)

    if 3 in steps_to_run:
        from reward_model import main as rm_main
        run_step(3, "Reward Model Training", rm_main)

    if 4 in steps_to_run:
        from ppo_train import main as ppo_main
        run_step(4, "PPO Alignment Training", ppo_main)

    if 5 in steps_to_run:
        from analysis import main as analysis_main
        run_step(5, "Analysis & Metrics", analysis_main)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE - Total time: {total_elapsed:.1f}s")
    print("=" * 70)
    print("\nOutputs saved to:")
    print("  models/sft_model/        — SFT fine-tuned GPT-2")
    print("  models/reward_model/     — Trained reward model")
    print("  models/ppo_model/        — PPO-aligned model")
    print("  outputs/                 — Metrics, plots, and analysis")


if __name__ == "__main__":
    main()
