"""
Final Comprehensive Analysis Script
Machine Learning TP1 - EFREI M1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")


def analyze_cleaned_data():
    """Analyze the cleaned datasets and compare with original"""
    print("🚀 Starting Comprehensive Analysis of NASA MDP Datasets")
    print("=" * 80)

    # Check if cleaned data directories exist
    d_prime_dir = "data/CleanedData/MDP/D'"
    d_double_prime_dir = "data/CleanedData/MDP/D''"

    results = []

    # Get datasets from original analysis
    original_results = pd.read_csv("results/dataset_comparison.csv")
    print(f"📊 Original dataset analysis loaded: {len(original_results)} datasets")

    # Analyze D' datasets if they exist
    if os.path.exists(d_prime_dir):
        print(f"\n🔍 Analyzing D' (cleaned) datasets...")
        d_prime_files = [f for f in os.listdir(d_prime_dir) if f.endswith(".arff")]
        print(f"Found {len(d_prime_files)} D' datasets: {d_prime_files}")

        for dataset_file in d_prime_files:
            dataset_name = dataset_file.replace(".arff", "")
            dataset_path = os.path.join(d_prime_dir, dataset_file)

            try:
                data = load_data(dataset_path)
                if data is not None:
                    # Quick analysis
                    n_samples, n_features = data.shape
                    target_col = (
                        "Defective" if "Defective" in data.columns else data.columns[-1]
                    )

                    # Missing values
                    missing_count = data.isnull().sum().sum()
                    missing_ratio = missing_count / (n_samples * n_features)

                    # Target distribution
                    target_dist = data[target_col].value_counts()
                    defective_ratio = (
                        target_dist.min() / target_dist.sum()
                        if len(target_dist) >= 2
                        else 0
                    )

                    results.append(
                        {
                            "Dataset": dataset_name,
                            "Category": "D'",
                            "Samples": n_samples,
                            "Features": n_features - 1,
                            "Missing_Values": missing_count,
                            "Missing_Ratio": missing_ratio,
                            "Defective_Ratio": defective_ratio,
                        }
                    )

                    print(
                        f"  ✅ {dataset_name}: {n_samples} samples, {missing_count} missing values"
                    )

            except Exception as e:
                print(f"  ❌ Error analyzing {dataset_name}: {e}")

    # Analyze D'' datasets if they exist
    if os.path.exists(d_double_prime_dir):
        print(f"\n🔍 Analyzing D'' (highly cleaned) datasets...")
        d_double_prime_files = [
            f for f in os.listdir(d_double_prime_dir) if f.endswith(".arff")
        ]
        print(f"Found {len(d_double_prime_files)} D'' datasets: {d_double_prime_files}")

        for dataset_file in d_double_prime_files:
            dataset_name = dataset_file.replace(".arff", "")
            dataset_path = os.path.join(d_double_prime_dir, dataset_file)

            try:
                data = load_data(dataset_path)
                if data is not None:
                    # Quick analysis
                    n_samples, n_features = data.shape
                    target_col = (
                        "Defective" if "Defective" in data.columns else data.columns[-1]
                    )

                    # Missing values
                    missing_count = data.isnull().sum().sum()
                    missing_ratio = missing_count / (n_samples * n_features)

                    # Target distribution
                    target_dist = data[target_col].value_counts()
                    defective_ratio = (
                        target_dist.min() / target_dist.sum()
                        if len(target_dist) >= 2
                        else 0
                    )

                    results.append(
                        {
                            "Dataset": dataset_name,
                            "Category": "D''",
                            "Samples": n_samples,
                            "Features": n_features - 1,
                            "Missing_Values": missing_count,
                            "Missing_Ratio": missing_ratio,
                            "Defective_Ratio": defective_ratio,
                        }
                    )

                    print(
                        f"  ✅ {dataset_name}: {n_samples} samples, {missing_count} missing values"
                    )

            except Exception as e:
                print(f"  ❌ Error analyzing {dataset_name}: {e}")

    # Create comparison if we have results
    if results:
        cleaned_df = pd.DataFrame(results)

        # Add original results for comparison
        original_df = original_results.copy()
        original_df["Category"] = "Original"
        original_df = original_df[
            [
                "Dataset",
                "Category",
                "Samples",
                "Features",
                "Missing_Values",
                "Missing_Ratio",
                "Defective_Ratio",
            ]
        ]

        # Combine all results
        all_results = pd.concat([original_df, cleaned_df], ignore_index=True)

        # Save comprehensive results
        all_results.to_csv("results/comprehensive_comparison.csv", index=False)
        print(
            f"\n✅ Comprehensive comparison saved to results/comprehensive_comparison.csv"
        )

        # Create summary
        print(f"\n📊 SUMMARY BY CATEGORY:")
        print("=" * 60)
        summary = (
            all_results.groupby("Category")
            .agg(
                {
                    "Samples": ["count", "mean"],
                    "Missing_Values": ["mean", "sum"],
                    "Missing_Ratio": "mean",
                    "Defective_Ratio": "mean",
                }
            )
            .round(4)
        )
        print(summary)

        # Create visualization
        create_comparison_visualization(all_results)

        return all_results
    else:
        print("❌ No cleaned datasets found to analyze")
        return None


def create_comparison_visualization(df):
    """Create visualization comparing dataset categories"""
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Dataset Comparison: Original vs Cleaned Data", fontsize=16, fontweight="bold"
    )

    # 1. Sample sizes
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x="Category", y="Samples", ax=ax1)
    ax1.set_title("Sample Sizes by Category")
    ax1.set_ylabel("Number of Samples")

    # 2. Missing value ratios
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x="Category", y="Missing_Ratio", ax=ax2)
    ax2.set_title("Missing Value Ratios")
    ax2.set_ylabel("Missing Value Ratio")

    # 3. Defective ratios
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x="Category", y="Defective_Ratio", ax=ax3)
    ax3.set_title("Defective Ratios")
    ax3.set_ylabel("Defective Ratio")

    # 4. Feature counts
    ax4 = axes[1, 1]
    sns.boxplot(data=df, x="Category", y="Features", ax=ax4)
    ax4.set_title("Number of Features")
    ax4.set_ylabel("Number of Features")

    plt.tight_layout()
    plt.savefig("results/cleaned_data_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\n📊 Comparison plots saved to results/cleaned_data_comparison.png")


def generate_final_summary():
    """Generate final project summary"""
    print(f"\n" + "=" * 100)
    print("FINAL PROJECT SUMMARY - ML TP1")
    print("=" * 100)

    print(f"\n🎯 PROJECT OBJECTIVES:")
    print("  ✅ Analyze NASA MDP software defect prediction datasets")
    print("  ✅ Implement multiple machine learning algorithms")
    print("  ✅ Compare model performance across datasets")
    print("  ✅ Evaluate data preprocessing impact")

    print(f"\n📈 MODELS IMPLEMENTED:")
    model_results = pd.read_csv("results/model_comparison.csv")
    print("  • Logistic Regression")
    print("  • Decision Tree")
    print("  • Random Forest")
    print("  • Support Vector Machine")
    print("  • K-Nearest Neighbors")
    print("  • Naive Bayes")

    print(f"\n🏆 BEST PERFORMING MODEL:")
    best_model = model_results.loc[model_results["F1-Score"].idxmax()]
    print(f"  Model: {best_model['Model']}")
    print(f"  Accuracy: {best_model['Accuracy']:.3f}")
    print(f"  F1-Score: {best_model['F1-Score']:.3f}")
    print(f"  ROC AUC: {best_model['ROC AUC']:.3f}")

    print(f"\n📊 DATASETS ANALYZED:")
    dataset_results = pd.read_csv("results/dataset_comparison.csv")
    print(f"  • Total datasets: {len(dataset_results)}")
    print(f"  • Average samples: {dataset_results['Samples'].mean():.0f}")
    print(f"  • Average features: {dataset_results['Features'].mean():.0f}")
    print(
        f"  • Average defective ratio: {dataset_results['Defective_Ratio'].mean():.3f}"
    )

    print(f"\n🔧 TECHNICAL ACHIEVEMENTS:")
    print("  ✅ ARFF file format support implemented")
    print("  ✅ Comprehensive data preprocessing pipeline")
    print("  ✅ Cross-validation for model evaluation")
    print("  ✅ Statistical analysis and visualization")
    print("  ✅ Automated comparative analysis across datasets")

    print(f"\n📁 GENERATED OUTPUTS:")
    print("  • results/model_comparison.csv - Model performance metrics")
    print("  • results/dataset_comparison.csv - Dataset characteristics")
    print("  • results/dataset_comparison_plots.png - Visualization plots")
    print("  • notebooks/ml_tp1_analysis.ipynb - Interactive analysis")

    print(f"\n💡 KEY INSIGHTS:")
    print("  1. Class imbalance significantly affects model performance")
    print("  2. Missing value handling is crucial for some datasets")
    print("  3. Feature scaling improves model convergence")
    print("  4. Ensemble methods (Random Forest) show robust performance")
    print("  5. Cross-validation reveals model stability across datasets")


if __name__ == "__main__":
    # Run the analysis
    results = analyze_cleaned_data()

    # Generate final summary
    generate_final_summary()

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("=" * 50)
    print("All results have been saved to the results/ directory.")
    print("Check the generated CSV files and plots for detailed insights.")
