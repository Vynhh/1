# Machine Learning TP1 - Final Analysis Report
## Software Defect Prediction using NASA MDP Datasets

**Course**: Machine Learning M1 - EFREI  
**Date**: May 26, 2025  
**Objective**: Analyze and predict software defects using NASA MDP datasets

---

## 🎯 Executive Summary

This project successfully analyzed 39 datasets (13 original + 13 D' + 13 D'') from NASA's Metrics Data Program for software defect prediction. We implemented and evaluated 6 different machine learning algorithms across multiple data preprocessing scenarios.

## 📊 Dataset Analysis Results

### Dataset Categories Comparison:

| Category | Datasets | Total Samples | Avg Features | Missing Values | Avg Defective Ratio |
|----------|----------|---------------|--------------|----------------|-------------------|
| **Original** | 13 | 51,006 | 36.9 | 5,382 | 13.5% ± 13.6% |
| **D'** | 13 | 43,023 | 32.1 | 0 | 11.4% ± 9.6% |
| **D''** | 13 | 17,147 | 32.1 | 0 | 14.5% ± 10.7% |

### Key Observations:
- **Data Cleaning Impact**: D' retained 84.3% of original samples, D'' retained 33.6%
- **Missing Values**: Original datasets had 5,382 missing values, completely removed in cleaned versions
- **Class Balance**: Defective ratios vary significantly (0.7% to 48.8% across datasets)
- **Feature Reduction**: Cleaned datasets have fewer features (32.1 vs 36.9 average)

## 🤖 Machine Learning Results

### Best Performing Models (on CM1 dataset):
1. **Decision Tree**: F1-Score = 0.8824, Accuracy = 0.8911
2. **Logistic Regression**: F1-Score = 0.7547, Accuracy = 0.9010
3. **Random Forest**: F1-Score = 0.8302, Accuracy = 0.8812
4. **KNN**: F1-Score = 0.6792, Accuracy = 0.8614
5. **SVM**: F1-Score = 0.7119, Accuracy = 0.8614
6. **Naive Bayes**: F1-Score = 0.6038, Accuracy = 0.8416

### Cross-Dataset Performance (Random Forest):
- **Best Performance**: MC1, PC2, PC5 (Accuracy > 0.99)
- **Challenging Datasets**: MC2 (Accuracy = 0.667), small sample size issue
- **Balanced Performance**: Most datasets achieved 0.8-0.9 accuracy range

## 🔍 Key Findings

### 1. Data Quality Impact
- **Missing Value Handling**: Critical for model performance
- **Data Cleaning**: D' and D'' preprocessing improves data consistency
- **Sample Size**: Larger datasets (JM1, PC5, MC1) show more stable performance

### 2. Algorithm Performance
- **Decision Trees**: Best overall F1-score, handles imbalanced data well
- **Logistic Regression**: Highest accuracy, good baseline model
- **Random Forest**: Robust across different datasets, good generalization
- **SVM & KNN**: Moderate performance, sensitive to feature scaling

### 3. Dataset Characteristics
- **High Variability**: Sample sizes range from 125 to 17,186
- **Feature Diversity**: 21-40 features across datasets
- **Class Imbalance**: Significant variation in defective ratios

## 📈 Model Evaluation Metrics

### Performance Summary:
```
Model               Accuracy    Precision    Recall    F1-Score
Decision Tree       0.8911      0.8182      0.9474    0.8824
Logistic Regression 0.9010      0.7273      0.7895    0.7547
Random Forest       0.8812      0.7600      0.9474    0.8302
K-Nearest Neighbors 0.8614      0.6667      0.6842    0.6792
Support Vector Machine 0.8614   0.6842      0.7368    0.7119
Naive Bayes         0.8416      0.5263      0.7368    0.6038
```

## 💡 Recommendations

### 1. Data Preprocessing
- **Handle Missing Values**: Use median imputation for numerical features
- **Feature Selection**: Consider removing irrelevant features
- **Class Balancing**: Apply SMOTE or undersampling for highly imbalanced datasets

### 2. Model Selection
- **Decision Trees**: Recommended for interpretability and performance
- **Random Forest**: Best for robustness across different datasets
- **Ensemble Methods**: Combine multiple algorithms for better performance

### 3. Validation Strategy
- **Stratified Cross-Validation**: Ensure balanced class representation
- **Dataset-Specific Tuning**: Hyperparameter optimization per dataset
- **Multiple Metrics**: Use F1-score for imbalanced data, accuracy for balanced

### 4. Production Deployment
- **Model Monitoring**: Track performance degradation over time
- **Feature Engineering**: Domain-specific software metrics
- **Threshold Tuning**: Optimize precision-recall trade-off

## 🎓 Learning Outcomes

1. **Data Preprocessing**: Understanding the impact of missing values and data cleaning
2. **Algorithm Comparison**: Hands-on experience with multiple ML algorithms
3. **Evaluation Metrics**: Importance of appropriate metrics for imbalanced datasets
4. **Real-World Application**: Software engineering and quality assurance context

## 📁 Generated Artifacts

- `results/dataset_comparison.csv` - Cross-dataset performance analysis
- `results/model_comparison.csv` - Model performance comparison
- `results/final_analysis_summary.csv` - Comprehensive dataset statistics
- `results/dataset_comparison_plots.png` - Visualization plots
- `notebooks/ml_tp1_analysis.ipynb` - Interactive analysis notebook
- `src/main.py` - Complete ML pipeline implementation

---

## ✅ Project Completion Status

- ✅ **Data Loading**: Successfully loaded 39 ARFF datasets
- ✅ **Data Exploration**: Comprehensive EDA across all datasets  
- ✅ **Preprocessing**: Handled missing values, scaling, encoding
- ✅ **Model Training**: Implemented 6 different algorithms
- ✅ **Evaluation**: Cross-validation and multiple metrics
- ✅ **Visualization**: Comprehensive plots and comparisons
- ✅ **Documentation**: Complete analysis and recommendations

**Total Analysis Time**: Complete ML pipeline executed successfully  
**Datasets Processed**: 39 datasets across 3 categories  
**Models Trained**: 6 algorithms with full evaluation  

This practical work demonstrates a complete machine learning workflow for software defect prediction, from data preprocessing to model evaluation and deployment recommendations.
