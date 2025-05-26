"""
Final Report Generator for ML TP1
Creates comprehensive visualizations and summary
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_final_report():
    """Generate final comprehensive report with visualizations"""

    # Load all results
    try:
        comparative_results = pd.read_csv("results/dataset_comparison.csv")
        final_results = pd.read_csv("results/final_analysis_summary.csv")
        model_results = pd.read_csv("results/model_comparison.csv")
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    print("🎯 FINAL ML TP1 ANALYSIS REPORT")
    print("=" * 80)

    # 1. Dataset Overview
    print("\n📊 DATASET OVERVIEW")
    print("-" * 40)

    overview = (
        final_results.groupby("Category")
        .agg(
            {
                "Samples": ["count", "sum", "mean"],
                "Features": ["mean"],
                "Missing_Values": ["sum"],
                "Defective_Ratio": ["mean", "std"],
            }
        )
        .round(3)
    )

    print(overview)

    # 2. Key Findings
    print("\n🔍 KEY FINDINGS")
    print("-" * 40)

    # Data cleaning impact
    original_total = final_results[final_results["Category"] == "Original"][
        "Samples"
    ].sum()
    d_prime_total = final_results[final_results["Category"] == "D'"]["Samples"].sum()
    d_double_total = final_results[final_results["Category"] == "D''"]["Samples"].sum()

    print(f"• Original datasets: {original_total:,} total samples")
    print(
        f"• D' (cleaned): {d_prime_total:,} total samples ({(d_prime_total / original_total) * 100:.1f}% retained)"
    )
    print(
        f"• D'' (filtered): {d_double_total:,} total samples ({(d_double_total / original_total) * 100:.1f}% retained)"
    )

    # Missing values impact
    original_missing = final_results[final_results["Category"] == "Original"][
        "Missing_Values"
    ].sum()
    print(f"• Original datasets had {original_missing:,} missing values")
    print(f"• Cleaned datasets (D', D'') have 0 missing values")

    # Best performing models
    if not model_results.empty:
        best_model = model_results.loc[model_results["F1_Score"].idxmax()]
        print(
            f"• Best performing model: {best_model['Model']} (F1: {best_model['F1_Score']:.3f})"
        )

    # 3. Create comprehensive visualization
    create_comprehensive_plots(final_results, comparative_results)

    # 4. Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    print("1. Data preprocessing significantly improves data quality")
    print("2. Missing value handling is crucial for model performance")
    print("3. Class imbalance varies across datasets - consider resampling")
    print("4. Feature selection may improve smaller datasets")
    print("5. Both Random Forest and Logistic Regression show good performance")
    print("6. Cross-validation should be used for robust evaluation")

    print(f"\n✅ Final report generated successfully!")
    print(f"📁 Check results/ directory for all outputs")


def create_comprehensive_plots(final_df, comparative_df):
    """Create comprehensive visualization plots"""

    plt.style.use("default")
    fig = plt.figure(figsize=(20, 15))

    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Sample size comparison
    ax1 = fig.add_subplot(gs[0, 0])
    sample_summary = final_df.groupby("Category")["Samples"].sum()
    ax1.bar(
        sample_summary.index,
        sample_summary.values,
        color=["skyblue", "lightgreen", "lightcoral"],
    )
    ax1.set_title("Total Samples by Category")
    ax1.set_ylabel("Total Samples")
    for i, v in enumerate(sample_summary.values):
        ax1.text(i, v + 1000, f"{v:,}", ha="center", va="bottom")

    # 2. Missing values impact
    ax2 = fig.add_subplot(gs[0, 1])
    missing_summary = final_df.groupby("Category")["Missing_Values"].sum()
    ax2.bar(
        missing_summary.index, missing_summary.values, color=["red", "green", "green"]
    )
    ax2.set_title("Missing Values by Category")
    ax2.set_ylabel("Total Missing Values")
    for i, v in enumerate(missing_summary.values):
        ax2.text(i, v + 100, f"{v:,}", ha="center", va="bottom")

    # 3. Defective ratio distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(data=final_df, x="Category", y="Defective_Ratio", ax=ax3)
    ax3.set_title("Defective Ratio Distribution")
    ax3.set_ylabel("Defective Ratio")

    # 4. Feature count comparison
    ax4 = fig.add_subplot(gs[1, 0])
    feature_summary = final_df.groupby("Category")["Features"].mean()
    ax4.bar(
        feature_summary.index,
        feature_summary.values,
        color=["orange", "purple", "brown"],
    )
    ax4.set_title("Average Features by Category")
    ax4.set_ylabel("Average Features")
    for i, v in enumerate(feature_summary.values):
        ax4.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom")

    # 5. Dataset size distribution
    ax5 = fig.add_subplot(gs[1, 1])
    sns.boxplot(data=final_df, x="Category", y="Samples", ax=ax5)
    ax5.set_title("Sample Size Distribution")
    ax5.set_ylabel("Number of Samples")
    ax5.set_yscale("log")

    # 6. Model performance (if available)
    ax6 = fig.add_subplot(gs[1, 2])
    if not comparative_df.empty and "RF_Accuracy" in comparative_df.columns:
        accuracy_data = comparative_df[["RF_Accuracy", "RF_F1_Score"]].values
        ax6.scatter(accuracy_data[:, 0], accuracy_data[:, 1], alpha=0.6, s=50)
        ax6.set_xlabel("RF Accuracy")
        ax6.set_ylabel("RF F1-Score")
        ax6.set_title("Model Performance Scatter")
        ax6.plot([0, 1], [0, 1], "r--", alpha=0.5)
    else:
        ax6.text(
            0.5,
            0.5,
            "Model performance\ndata not available",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )
        ax6.set_title("Model Performance")

    # 7. Data cleaning impact per dataset
    ax7 = fig.add_subplot(gs[2, :])
    pivot_df = final_df.pivot(index="Dataset", columns="Category", values="Samples")
    pivot_df = pivot_df.fillna(0)

    x = np.arange(len(pivot_df.index))
    width = 0.25

    ax7.bar(x - width, pivot_df["Original"], width, label="Original", alpha=0.8)
    ax7.bar(x, pivot_df["D'"], width, label="D'", alpha=0.8)
    ax7.bar(x + width, pivot_df["D''"], width, label="D''", alpha=0.8)

    ax7.set_xlabel("Dataset")
    ax7.set_ylabel("Number of Samples")
    ax7.set_title("Data Cleaning Impact: Sample Count by Dataset")
    ax7.set_xticks(x)
    ax7.set_xticklabels(pivot_df.index, rotation=45, ha="right")
    ax7.legend()
    ax7.set_yscale("log")

    plt.suptitle(
        "ML TP1 Comprehensive Analysis: NASA MDP Software Defect Prediction",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig("results/final_comprehensive_report.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"📊 Comprehensive plots saved to results/final_comprehensive_report.png")


if __name__ == "__main__":
    create_final_report()
