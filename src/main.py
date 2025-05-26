"""
Machine Learning TP1 - EFREI M1
Main script for practical work - Software Defect Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import warnings

warnings.filterwarnings("ignore")

from utils import (
    load_data,
    get_available_datasets,
    explore_data,
    plot_correlations,
    plot_target_distribution,
    save_results,
)


def preprocess_data(data: pd.DataFrame) -> tuple:
    """
    Preprocess the data for machine learning

    Args:
        data: Raw dataset

    Returns:
        X: Features
        y: Target variable
    """
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)

    # Create a copy to avoid modifying original data
    df = data.copy()

    # Identify target column (usually 'defects' or last column)
    if "defects" in df.columns:
        target_col = "defects"
    elif "Defective" in df.columns:
        target_col = "Defective"
    else:
        target_col = df.columns[-1]

    print(f"Target column: {target_col}")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle missing values in features only (numeric columns)
    if X.isnull().sum().sum() > 0:
        print("Handling missing values in features...")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # Handle categorical target if needed
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Encoded target classes: {le.classes_}")

    # Convert boolean target to binary if needed
    if y.dtype == "bool":
        y = y.astype(int)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{pd.Series(y).value_counts()}")

    return X, y


def train_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Train multiple machine learning models

    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing targets

    Returns:
        Dictionary containing model results
    """
    print("\n" + "=" * 50)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 50)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42, probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        # ROC AUC for binary classification
        try:
            roc_auc = (
                roc_auc_score(y_test, y_pred_proba)
                if y_pred_proba is not None
                else None
            )
        except Exception:
            roc_auc = None

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    return results


def evaluate_and_visualize_results(results: dict, y_test) -> None:
    """
    Evaluate and visualize model results

    Args:
        results: Dictionary containing model results
        y_test: True test labels
    """
    print("\n" + "=" * 50)
    print("RESULTS ANALYSIS")
    print("=" * 50)

    # Create comparison DataFrame
    comparison_data = []
    for name, result in results.items():
        comparison_data.append(
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1-Score": result["f1_score"],
                "ROC AUC": result["roc_auc"],
                "CV Mean": result["cv_mean"],
                "CV Std": result["cv_std"],
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Comparison:")
    print(comparison_df.round(4))

    # Save results
    save_results(comparison_df, "model_comparison.csv")

    # Visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Accuracy comparison
    plt.subplot(2, 3, 1)
    plt.bar(comparison_df["Model"], comparison_df["Accuracy"])
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")

    # Plot 2: F1-Score comparison
    plt.subplot(2, 3, 2)
    plt.bar(comparison_df["Model"], comparison_df["F1-Score"])
    plt.title("F1-Score Comparison")
    plt.xticks(rotation=45)
    plt.ylabel("F1-Score")

    # Plot 3: Cross-validation scores
    plt.subplot(2, 3, 3)
    plt.bar(comparison_df["Model"], comparison_df["CV Mean"])
    plt.errorbar(
        range(len(comparison_df)),
        comparison_df["CV Mean"],
        yerr=comparison_df["CV Std"],
        fmt="none",
        color="red",
    )
    plt.title("Cross-Validation Scores")
    plt.xticks(range(len(comparison_df)), comparison_df["Model"], rotation=45)
    plt.ylabel("CV Score")

    # Plot 4: Confusion Matrix for best model
    best_model_name = comparison_df.loc[comparison_df["F1-Score"].idxmax(), "Model"]
    best_result = results[best_model_name]

    plt.subplot(2, 3, 4)
    cm = confusion_matrix(y_test, best_result["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Plot 5: All metrics comparison
    plt.subplot(2, 3, 5)
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(comparison_df))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i * width, comparison_df[metric], width, label=metric)

    plt.title("All Metrics Comparison")
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.xticks(x + width * 1.5, comparison_df["Model"], rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"\nBest model: {best_model_name}")
    print(f"Best F1-Score: {results[best_model_name]['f1_score']:.4f}")


def main():
    """
    Main function for the ML practical work
    """
    print("Machine Learning TP1 - EFREI M1")
    print("Software Defect Prediction")
    print("=" * 50)

    # Get available datasets
    datasets = get_available_datasets()
    print(f"\nAvailable datasets: {datasets}")

    if not datasets:
        print("No datasets found! Please check the data directory.")
        return

    # Use the first dataset (you can change this)
    dataset_name = datasets[0]  # CM1.arff
    dataset_path = f"data/OriginalData/MDP/{dataset_name}"

    print(f"\nLoading dataset: {dataset_name}")

    # Load and explore data
    data = load_data(dataset_path)
    if data is None:
        print("Failed to load data!")
        return

    explore_data(data)

    # Visualize data
    plot_correlations(data)

    # Check if target column exists and plot distribution
    target_cols = ["defects", "Defective", "bugs"]
    target_col = None
    for col in target_cols:
        if col in data.columns:
            target_col = col
            break

    if target_col is None:
        # Assume last column is target
        target_col = data.columns[-1]

    plot_target_distribution(data, target_col)

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Evaluate and visualize results
    evaluate_and_visualize_results(results, y_test)

    print("\n" + "=" * 50)
    print("TP1 COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()
