# ML TP1 Utilities
"""
Utility functions for Machine Learning TP1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import os
from typing import Tuple, List, Optional


def load_arff_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load data from an ARFF file"""
    try:
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)

        # Convert bytes to strings for categorical columns
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.decode("utf-8")

        print(f"Data loaded successfully from {os.path.basename(filepath)}: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading ARFF data from {filepath}: {e}")
        return None


def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load data from a CSV or ARFF file"""
    try:
        if filepath.endswith(".arff"):
            return load_arff_data(filepath)
        else:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {data.shape}")
            return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def get_available_datasets() -> List[str]:
    """Get list of available datasets"""
    datasets = []
    data_dir = "data/OriginalData/MDP"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".arff"):
                datasets.append(file)
    return sorted(datasets)


def explore_data(data: pd.DataFrame) -> None:
    """Basic data exploration"""
    print("\n" + "=" * 50)
    print("DATA EXPLORATION")
    print("=" * 50)

    print(f"\nDataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")

    print("\nData description:")
    print(data.describe())

    print("\nFirst 5 rows:")
    print(data.head())

    # Check target variable if exists
    if "defects" in data.columns:
        print("\nTarget variable distribution:")
        print(data["defects"].value_counts())


def plot_correlations(data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> None:
    """Plot correlation matrix"""
    plt.figure(figsize=figsize)

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    if len(numeric_data.columns) > 0:
        correlation_matrix = numeric_data.corr()

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            square=True,
        )
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric columns found for correlation analysis.")


def plot_target_distribution(data: pd.DataFrame, target_col: str = "defects") -> None:
    """Plot target variable distribution"""
    if target_col in data.columns:
        plt.figure(figsize=(10, 6))

        # Count plot
        plt.subplot(1, 2, 1)
        data[target_col].value_counts().plot(kind="bar")
        plt.title(f"{target_col.capitalize()} Distribution")
        plt.xlabel(target_col.capitalize())
        plt.ylabel("Count")

        # Pie chart
        plt.subplot(1, 2, 2)
        data[target_col].value_counts().plot(kind="pie", autopct="%1.1f%%")
        plt.title(f"{target_col.capitalize()} Proportion")
        plt.ylabel("")

        plt.tight_layout()
        plt.show()
    else:
        print(f"Column '{target_col}' not found in the dataset.")


def save_results(results, filename: str) -> None:
    """Save results to a file"""
    filepath = f"results/{filename}"
    if isinstance(results, dict):
        import json

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
    else:
        results.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
