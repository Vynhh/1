"""
Final Comprehensive Analysis Script
Machine Learning TP1 - EFREI M1
Analyzes Original vs Cleaned datasets and generates final results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, get_available_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")


def get_datasets_from_directory(directory: str):
    """Get available datasets from a specific directory"""
    import os

    datasets = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith(".arff"):
                datasets.append(file)
    return sorted(datasets)


def analyze_dataset_category(data_dir: str, category_name: str):
    """Analyze all datasets in a category (Original, D', D'')"""
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {category_name.upper()} DATASETS")
    print("=" * 80)

    datasets = get_datasets_from_directory(data_dir)
    if not datasets:
        print(f"No datasets found in {data_dir}")
        return None

    results = []

    for dataset_name in datasets:
        print(f"\nAnalyzing {category_name} - {dataset_name}")
        dataset_path = f"{data_dir}/{dataset_name}"

        try:
            data = load_data(dataset_path)
            if data is None:
                continue

            # Basic statistics
            n_samples, n_features = data.shape

            # Target analysis
            target_col = "Defective"
            if target_col not in data.columns:
                target_col = data.columns[-1]

            target_dist = data[target_col].value_counts()
            if len(target_dist) >= 2:
                defective_ratio = target_dist.min() / target_dist.sum()
                class_balance = target_dist.min() / target_dist.max()
            else:
                defective_ratio = 0
                class_balance = 0

            # Missing values
            missing_count = data.isnull().sum().sum()
            missing_ratio = missing_count / (n_samples * n_features)

            # Quick model evaluation
            try:
                # Preprocess
                df_copy = data.copy()
                numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
                if missing_count > 0:
                    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(
                        df_copy[numeric_cols].median()
                    )

                X = df_copy.drop(columns=[target_col])
                y = df_copy[target_col]

                if y.dtype == "object":
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                if len(np.unique(y)) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Random Forest
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf.fit(X_train_scaled, y_train)
                    rf_pred = rf.predict(X_test_scaled)

                    # Logistic Regression
                    lr = LogisticRegression(random_state=42, max_iter=1000)
                    lr.fit(X_train_scaled, y_train)
                    lr_pred = lr.predict(X_test_scaled)

                    rf_acc = accuracy_score(y_test, rf_pred)
                    rf_f1 = f1_score(y_test, rf_pred)
                    lr_acc = accuracy_score(y_test, lr_pred)
                    lr_f1 = f1_score(y_test, lr_pred)
                else:
                    rf_acc = rf_f1 = lr_acc = lr_f1 = 0

            except Exception as e:
                print(f"Error in model evaluation: {e}")
                rf_acc = rf_f1 = lr_acc = lr_f1 = 0

            results.append(
                {
                    "Dataset": dataset_name.replace(".arff", ""),
                    "Category": category_name,
                    "Samples": n_samples,
                    "Features": n_features - 1,
                    "Defective_Ratio": defective_ratio,
                    "Class_Balance": class_balance,
                    "Missing_Values": missing_count,
                    "Missing_Ratio": missing_ratio,
                    "RF_Accuracy": rf_acc,
                    "RF_F1": rf_f1,
                    "LR_Accuracy": lr_acc,
                    "LR_F1": lr_f1,
                }
            )

            print(f"  Samples: {n_samples}, Features: {n_features - 1}")
            print(f"  Defective ratio: {defective_ratio:.3f}, Missing: {missing_count}")
            print(f"  RF Acc: {rf_acc:.3f}, LR Acc: {lr_acc:.3f}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue

    return pd.DataFrame(results)


def create_comprehensive_comparison():
    """Create comprehensive comparison of all dataset categories"""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 100)
    # Analyze each category
    original_results = analyze_dataset_category("../data/OriginalData/MDP", "Original")
    d_prime_results = analyze_dataset_category("../data/CleanedData/MDP/D'", "D'")
    d_double_prime_results = analyze_dataset_category(
        "../data/CleanedData/MDP/D''", "D''"
    )

    # Combine all results
    all_results = []
    if original_results is not None:
        all_results.append(original_results)
    if d_prime_results is not None:
        all_results.append(d_prime_results)
    if d_double_prime_results is not None:
        all_results.append(d_double_prime_results)

    if not all_results:
        print("No results to combine!")
        return

    combined_df = pd.concat(all_results, ignore_index=True)
    # Save results
    combined_df.to_csv("../results/comprehensive_analysis.csv", index=False)
    print(f"\n✅ Comprehensive results saved to results/comprehensive_analysis.csv")

    # Create summary statistics
    create_summary_statistics(combined_df)
    create_comparison_plots(combined_df)

    return combined_df


def create_summary_statistics(df):
    """Create summary statistics by category"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS BY CATEGORY")
    print("=" * 80)

    summary_stats = (
        df.groupby("Category")
        .agg(
            {
                "Samples": ["mean", "std", "min", "max"],
                "Features": ["mean", "std"],
                "Defective_Ratio": ["mean", "std"],
                "Missing_Ratio": ["mean", "std"],
                "RF_Accuracy": ["mean", "std"],
                "LR_Accuracy": ["mean", "std"],
            }
        )
        .round(4)
    )

    print(summary_stats)
    # Save summary
    summary_stats.to_csv("../results/summary_statistics.csv")
    print(f"\n✅ Summary statistics saved to results/summary_statistics.csv")


def create_comparison_plots(df):
    """Create comprehensive comparison plots"""
    plt.style.use("default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Comprehensive Dataset Analysis: Original vs Cleaned Data",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Sample sizes by category
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x="Category", y="Samples", ax=ax1)
    ax1.set_title("Sample Sizes by Category")
    ax1.set_ylabel("Number of Samples")

    # 2. Defective ratios
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x="Category", y="Defective_Ratio", ax=ax2)
    ax2.set_title("Defective Ratios by Category")
    ax2.set_ylabel("Defective Ratio")

    # 3. Missing value ratios
    ax3 = axes[0, 2]
    sns.boxplot(data=df, x="Category", y="Missing_Ratio", ax=ax3)
    ax3.set_title("Missing Value Ratios by Category")
    ax3.set_ylabel("Missing Value Ratio")

    # 4. Random Forest Accuracy
    ax4 = axes[1, 0]
    sns.boxplot(data=df, x="Category", y="RF_Accuracy", ax=ax4)
    ax4.set_title("Random Forest Accuracy by Category")
    ax4.set_ylabel("RF Accuracy")

    # 5. Logistic Regression Accuracy
    ax5 = axes[1, 1]
    sns.boxplot(data=df, x="Category", y="LR_Accuracy", ax=ax5)
    ax5.set_title("Logistic Regression Accuracy by Category")
    ax5.set_ylabel("LR Accuracy")

    # 6. Accuracy comparison
    ax6 = axes[1, 2]
    melted_df = pd.melt(
        df,
        id_vars=["Category"],
        value_vars=["RF_Accuracy", "LR_Accuracy"],
        var_name="Model",
        value_name="Accuracy",
    )
    sns.boxplot(data=melted_df, x="Category", y="Accuracy", hue="Model", ax=ax6)
    ax6.set_title("Model Accuracy Comparison")
    ax6.set_ylabel("Accuracy")
    ax6.legend(title="Model")

    plt.tight_layout()
    plt.savefig(
        "../results/comprehensive_comparison_plots.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
    print(
        "\n📊 Comprehensive plots saved to results/comprehensive_comparison_plots.png"
    )


def generate_final_report(df):
    """Generate final analysis report"""
    print("\n" + "=" * 100)
    print("FINAL ANALYSIS REPORT")
    print("=" * 100)

    print("\n🔍 KEY FINDINGS:")
    print("-" * 50)

    # Dataset characteristics
    total_datasets = len(df)
    categories = df["Category"].unique()
    print(f"• Total datasets analyzed: {total_datasets}")
    print(f"• Categories: {', '.join(categories)}")

    # Sample size analysis
    avg_samples = df.groupby("Category")["Samples"].mean()
    print(f"\n📊 Average sample sizes:")
    for category in categories:
        print(f"  - {category}: {avg_samples[category]:.0f} samples")

    # Best performing models
    print(f"\n🏆 Best performing configurations:")
    best_rf = df.loc[df["RF_Accuracy"].idxmax()]
    best_lr = df.loc[df["LR_Accuracy"].idxmax()]
    print(
        f"  - Best RF: {best_rf['Dataset']} ({best_rf['Category']}) - {best_rf['RF_Accuracy']:.3f}"
    )
    print(
        f"  - Best LR: {best_lr['Dataset']} ({best_lr['Category']}) - {best_lr['LR_Accuracy']:.3f}"
    )

    # Data quality analysis
    print(f"\n🧹 Data quality insights:")
    avg_missing = df.groupby("Category")["Missing_Ratio"].mean()
    for category in categories:
        print(f"  - {category}: {avg_missing[category]:.4f} missing ratio")

    # Class imbalance analysis
    avg_defective = df.groupby("Category")["Defective_Ratio"].mean()
    print(f"\n⚖️ Class distribution:")
    for category in categories:
        print(f"  - {category}: {avg_defective[category]:.3f} defective ratio")

    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    print("1. Data preprocessing significantly impacts model performance")
    print(
        "2. Class imbalance varies widely across datasets - consider resampling techniques"
    )
    print("3. Missing value handling is crucial for some datasets")
    print("4. Both Random Forest and Logistic Regression show competitive performance")
    print("5. Feature selection and engineering could further improve results")


def main():
    """Main function for comprehensive analysis"""
    print("🚀 Starting Comprehensive Analysis of NASA MDP Datasets")
    print("=" * 80)

    # Create comprehensive comparison
    df = create_comprehensive_comparison()

    if df is not None and not df.empty:
        # Generate final report
        generate_final_report(df)

        print(f"\n✅ Analysis complete! Check the results/ directory for:")
        print("  - comprehensive_analysis.csv")
        print("  - summary_statistics.csv")
        print("  - comprehensive_comparison_plots.png")
    else:
        print("❌ No data to analyze!")


if __name__ == "__main__":
    main()
