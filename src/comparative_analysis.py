"""
Comparative Analysis Script for Multiple Datasets
Machine Learning TP1 - EFREI M1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_data, get_available_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")


def analyze_all_datasets():
    """Analyze all available datasets and compare their characteristics"""
    datasets = get_available_datasets()
    print(f"Found {len(datasets)} datasets to analyze: {datasets}")

    if not datasets:
        print("No datasets found!")
        return

    print(f"Found {len(datasets)} datasets: {datasets}")

    results_summary = []

    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Analyzing dataset: {dataset_name}")
        print("=" * 60)

        try:
            # Load dataset
            dataset_path = f"data/OriginalData/MDP/{dataset_name}"
            data = load_data(dataset_path)

            if data is None:
                continue

            # Basic info
            n_samples, n_features = data.shape
            n_features_actual = n_features - 1  # excluding target

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

            # Quick model performance
            try:
                # Preprocess data
                df_copy = data.copy()

                # Handle missing values
                numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
                if missing_count > 0:
                    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(
                        df_copy[numeric_cols].median()
                    )

                # Features and target
                X = df_copy.drop(columns=[target_col])
                y = df_copy[target_col]

                # Encode target
                if y.dtype == "object":
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                # Quick train/test split and model
                if len(np.unique(y)) > 1:  # Check if we have both classes
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train Random Forest
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf.fit(X_train_scaled, y_train)
                    y_pred = rf.predict(X_test_scaled)

                    rf_accuracy = accuracy_score(y_test, y_pred)
                    rf_f1 = f1_score(y_test, y_pred, average="weighted")
                else:
                    rf_accuracy = 0
                    rf_f1 = 0

            except Exception as e:
                print(f"Error in model training: {e}")
                rf_accuracy = 0
                rf_f1 = 0

            # Store results
            result = {
                "Dataset": dataset_name.replace(".arff", ""),
                "Samples": n_samples,
                "Features": n_features_actual,
                "Defective_Ratio": defective_ratio,
                "Class_Balance": class_balance,
                "Missing_Values": missing_count,
                "Missing_Ratio": missing_ratio,
                "RF_Accuracy": rf_accuracy,
                "RF_F1_Score": rf_f1,
            }

            results_summary.append(result)

            print(f"Samples: {n_samples}")
            print(f"Features: {n_features_actual}")
            print(f"Defective ratio: {defective_ratio:.3f}")
            print(f"Class balance: {class_balance:.3f}")
            print(f"Missing values: {missing_count} ({missing_ratio:.3f})")
            print(f"Quick RF Accuracy: {rf_accuracy:.3f}")
            print(f"Quick RF F1-Score: {rf_f1:.3f}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue

    # Create summary DataFrame
    if results_summary:
        summary_df = pd.DataFrame(results_summary)

        print(f"\n{'=' * 80}")
        print("DATASET COMPARISON SUMMARY")
        print("=" * 80)

        display_df = summary_df.round(3)
        print(display_df.to_string(index=False))

        # Save results
        summary_df.to_csv("results/dataset_comparison.csv", index=False)
        print(f"\n✅ Results saved to results/dataset_comparison.csv")

        # Visualizations
        create_comparison_plots(summary_df)

        return summary_df

    return None


def create_comparison_plots(summary_df):
    """Create comparison plots for all datasets"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Dataset Comparison Analysis", fontsize=16)

    # 1. Dataset sizes
    axes[0, 0].bar(summary_df["Dataset"], summary_df["Samples"])
    axes[0, 0].set_title("Number of Samples by Dataset")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("Number of Samples")

    # 2. Number of features
    axes[0, 1].bar(summary_df["Dataset"], summary_df["Features"])
    axes[0, 1].set_title("Number of Features by Dataset")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].set_ylabel("Number of Features")

    # 3. Defective ratio
    axes[0, 2].bar(summary_df["Dataset"], summary_df["Defective_Ratio"])
    axes[0, 2].set_title("Defective Ratio by Dataset")
    axes[0, 2].tick_params(axis="x", rotation=45)
    axes[0, 2].set_ylabel("Defective Ratio")
    axes[0, 2].axhline(
        y=0.1, color="r", linestyle="--", alpha=0.7, label="10% threshold"
    )
    axes[0, 2].legend()

    # 4. Class balance
    axes[1, 0].bar(summary_df["Dataset"], summary_df["Class_Balance"])
    axes[1, 0].set_title("Class Balance by Dataset")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].set_ylabel("Class Balance (min/max)")
    axes[1, 0].axhline(
        y=0.3, color="r", linestyle="--", alpha=0.7, label="Balance threshold"
    )
    axes[1, 0].legend()

    # 5. Model performance comparison
    axes[1, 1].bar(
        summary_df["Dataset"], summary_df["RF_Accuracy"], alpha=0.7, label="Accuracy"
    )
    axes[1, 1].bar(
        summary_df["Dataset"], summary_df["RF_F1_Score"], alpha=0.7, label="F1-Score"
    )
    axes[1, 1].set_title("Quick Random Forest Performance")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()

    # 6. Missing values ratio
    axes[1, 2].bar(summary_df["Dataset"], summary_df["Missing_Ratio"])
    axes[1, 2].set_title("Missing Values Ratio by Dataset")
    axes[1, 2].tick_params(axis="x", rotation=45)
    axes[1, 2].set_ylabel("Missing Values Ratio")

    plt.tight_layout()
    plt.savefig("results/dataset_comparison_plots.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("📊 Comparison plots saved to results/dataset_comparison_plots.png")


def main():
    """Main function for comparative analysis"""
    print("🔬 Starting Comparative Analysis of All Datasets")
    print("=" * 60)

    summary = analyze_all_datasets()

    if summary is not None:
        print("\n🎯 Analysis completed successfully!")
        print("Check the results/ directory for detailed outputs.")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")

        # Best balanced datasets
        balanced_datasets = summary[summary["Class_Balance"] > 0.3]["Dataset"].tolist()
        if balanced_datasets:
            print(f"✅ Well-balanced datasets: {', '.join(balanced_datasets)}")

        # Largest datasets
        largest_dataset = summary.loc[summary["Samples"].idxmax(), "Dataset"]
        print(f"📊 Largest dataset: {largest_dataset}")

        # Best performing datasets
        best_performer = summary.loc[summary["RF_F1_Score"].idxmax(), "Dataset"]
        print(f"🏆 Best performing dataset (Quick RF): {best_performer}")

        # Datasets with missing values
        missing_datasets = summary[summary["Missing_Values"] > 0]["Dataset"].tolist()
        if missing_datasets:
            print(f"⚠️  Datasets with missing values: {', '.join(missing_datasets)}")


if __name__ == "__main__":
    main()
