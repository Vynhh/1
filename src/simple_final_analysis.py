"""
Simple Final Analysis Script
Machine Learning TP1 - EFREI M1
"""

import pandas as pd
import numpy as np
import os
from utils import load_data


def analyze_datasets():
    """Analyze all three dataset categories"""
    print("🚀 Starting Final Analysis")
    print("=" * 50)

    categories = {
        "Original": "data/OriginalData/MDP",
        "D'": "data/CleanedData/MDP/D'",
        "D''": "data/CleanedData/MDP/D''",
    }

    all_results = []

    for category_name, directory in categories.items():
        print(f"\nAnalyzing {category_name} datasets...")

        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue

        datasets = [f for f in os.listdir(directory) if f.endswith(".arff")]
        print(f"Found {len(datasets)} datasets in {category_name}")

        for dataset in datasets:
            dataset_path = os.path.join(directory, dataset)
            data = load_data(dataset_path)

            if data is not None:
                n_samples, n_features = data.shape
                target_col = (
                    "Defective" if "Defective" in data.columns else data.columns[-1]
                )

                # Basic statistics
                missing_count = data.isnull().sum().sum()
                target_dist = data[target_col].value_counts()

                if len(target_dist) >= 2:
                    defective_ratio = target_dist.min() / target_dist.sum()
                else:
                    defective_ratio = 0

                result = {
                    "Dataset": dataset.replace(".arff", ""),
                    "Category": category_name,
                    "Samples": n_samples,
                    "Features": n_features - 1,
                    "Missing_Values": missing_count,
                    "Defective_Ratio": defective_ratio,
                }

                all_results.append(result)
                print(f"  {dataset}: {n_samples} samples, {n_features - 1} features")

    # Create DataFrame and save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("results/final_analysis_summary.csv", index=False)

        print(f"\n📊 Summary Results:")
        print("=" * 50)
        summary = (
            df.groupby("Category")
            .agg(
                {
                    "Samples": ["count", "mean", "std"],
                    "Features": ["mean"],
                    "Missing_Values": ["sum", "mean"],
                    "Defective_Ratio": ["mean", "std"],
                }
            )
            .round(3)
        )

        print(summary)

        print(f"\n✅ Results saved to results/final_analysis_summary.csv")
        return df
    else:
        print("❌ No datasets found!")
        return None


if __name__ == "__main__":
    analyze_datasets()
