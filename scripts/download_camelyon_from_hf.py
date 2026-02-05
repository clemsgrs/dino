#!/usr/bin/env python3
"""
Download CAMELYON dataset from HuggingFace.

Downloads the PathoROB-CAMELYON dataset and saves images + benchmark CSV.

Usage:
    python scripts/download_camelyon_from_hf.py --output-dir data/pathorob/camelyon
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download CAMELYON from HuggingFace")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pathorob/camelyon",
        help="Output directory for images and CSV",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset from HuggingFace (bifold-pathomics/PathoROB-camelyon)...")
    dataset = load_dataset("bifold-pathomics/PathoROB-camelyon", split="train")
    print(f"Found {len(dataset)} samples")

    # Save images and build dataframe
    records = []
    for idx, sample in enumerate(tqdm(dataset, desc="Saving images")):
        filename = f"{idx:05d}.png"
        image_path = images_dir / filename
        sample["image"].save(image_path)

        records.append({
            "sample_id": f"sample_{idx:05d}",
            "image_path": str(image_path.absolute()),
            "label": sample["biological_class"],
            "medical_center": sample["medical_center"],
            "slide_id": sample["slide_id"],
        })

    # Save CSV
    df = pd.DataFrame(records)
    csv_path = output_dir / "benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark CSV to {csv_path}")

    # Save metadata
    centers = df["medical_center"].unique().tolist()
    metadata = {
        "source": "bifold-pathomics/PathoROB-camelyon",
        "total_samples": len(df),
        "centers": centers,
        "samples_per_center": df["medical_center"].value_counts().to_dict(),
        "labels": df["label"].unique().tolist(),
        "class_distribution": df["label"].value_counts().to_dict(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\nSummary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Centers: {centers}")
    print(f"  Labels: {metadata['labels']}")
    print(f"\nSamples per center:")
    for center, count in sorted(metadata["samples_per_center"].items()):
        print(f"  {center}: {count}")


if __name__ == "__main__":
    main()
