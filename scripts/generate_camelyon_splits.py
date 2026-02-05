#!/usr/bin/env python3
"""Generate paper-faithful CAMELYON splits for PathoROB benchmark.

Uses the exact allocation matrices from the PathoROB paper (Figure 14)
to create training sets with varying Cramér's V correlation levels.

Usage:
    python scripts/generate_camelyon_splits.py

    # Custom output directory
    python scripts/generate_camelyon_splits.py --output-dir output/splits

    # More repetitions
    python scripts/generate_camelyon_splits.py --repetitions 5
"""

import argparse
from pathlib import Path

import pandas as pd

from dino.eval.pathorob.splits import generate_apd_splits


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-faithful CAMELYON splits"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default="data/pathorob/camelyon/benchmark.csv",
        help="Path to CAMELYON benchmark CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pathorob/camelyon/splits",
        help="Directory to save generated splits",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of random repetitions (paper uses 3)",
    )
    parser.add_argument(
        "--id-test-fraction",
        type=float,
        default=0.2,
        help="Fraction of ID slides to hold out for ID test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_csv}...")
    df = pd.read_csv(args.data_csv)

    # Add sample_id if missing
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"sample_{i:05d}" for i in range(len(df))]

    # Summary
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Centers: {df['medical_center'].unique().tolist()}")
    print(f"  Labels: {df['label'].unique().tolist()}")
    print(f"\nSamples per center:")
    print(df["medical_center"].value_counts().to_string())

    # Generate splits
    print(f"\nGenerating paper-faithful splits...")
    print(f"  Output: {args.output_dir}")
    print(f"  Repetitions: {args.repetitions}")
    print(f"  ID test fraction: {args.id_test_fraction}")
    print(f"  Seed: {args.seed}")

    splits = generate_apd_splits(
        df=df,
        output_dir=Path(args.output_dir),
        dataset_name="camelyon",
        repetitions=args.repetitions,
        correlation_levels=[],  # Ignored in paper mode
        id_centers=["RUMC", "UMCU"],
        ood_centers=["CWZ", "RST", "LPON"],
        id_test_fraction=args.id_test_fraction,
        seed=args.seed,
        mode="paper",
    )

    # Summary of generated splits
    print(f"\nGenerated {len(splits)} splits:")
    print(f"  {args.repetitions} reps x 8 V levels = {args.repetitions * 8} splits")

    # Show V levels and training sizes
    print(f"\nSplit details (rep 0):")
    print(f"  {'V':>6} | {'Train':>6} | {'ID Test':>7} | {'OOD Test':>8} | {'V realized':>10}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*7} | {'-'*8} | {'-'*10}")

    for split_df in splits[:8]:  # First rep only
        v_target = split_df["correlation_level"].iloc[0]
        v_realized = split_df["cramers_v_realized"].iloc[0]
        train_n = len(split_df[split_df["partition"] == "train"])
        id_test_n = len(split_df[split_df["partition"] == "id_test"])
        ood_test_n = len(split_df[split_df["partition"] == "ood_test"])
        print(f"  {v_target:>6.2f} | {train_n:>6} | {id_test_n:>7} | {ood_test_n:>8} | {v_realized:>10.3f}")

    print(f"\nSplits saved to: {args.output_dir}/camelyon/")


if __name__ == "__main__":
    main()
