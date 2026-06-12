# Telco Customer Churn Analysis

End-to-end machine learning project using the IBM Telco Customer Churn dataset to predict customer churn, compare classification models, and uncover key behavioral drivers of churn through clustering and feature importance analysis.

---

## Objectives

- Predict customer churn using supervised learning models
- Compare multiple classification approaches under consistent evaluation
- Identify key drivers of churn using permutation feature importance
- Segment customers using unsupervised clustering (K-Means + PCA visualization)
- Evaluate models using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

---

## Dataset

IBM Telco Customer Churn Dataset  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- ~7,000 customer records
- Mix of categorical and numerical features
- Binary target: churn (Yes/No)

---

## Models Tested

- Logistic Regression (scaled pipeline baseline)
- Random Forest
- XGBoost

---

## Methodology

### Data Preprocessing
- Handled missing values in `TotalCharges`
- One-hot encoded categorical variables
- Converted target variable to binary format
- Removed customer ID from feature space

### Feature Engineering
- Standard scaling for distance-based methods
- PCA for 2D clustering visualization (not used for supervised learning)

### Unsupervised Learning
- K-Means clustering on PCA-reduced feature space
- Elbow method + silhouette score used to select cluster count
- Cluster profiling based on churn rate, tenure, and revenue

### Supervised Learning
- Stratified 5-fold cross-validation
- Pipeline-based model training (scaling included where needed)
- Evaluation using ROC-AUC, F1, precision, and recall

### Model Interpretability
- Permutation feature importance for global explanations
- Cluster-level behavioral analysis for segmentation insights

---

## Key Findings

- Logistic Regression performed competitively with more complex models while remaining highly interpretable
- Clustering revealed distinct customer segments with significantly different churn behavior and revenue contribution

---

## Visualizations

- Elbow + silhouette plots for cluster selection
- PCA-based cluster visualization
- ROC curves comparing model performance
- Precision-recall curves
- Feature importance heatmaps
- Cluster-level business KPI breakdowns

---

## Future Work

- Hyperparameter tuning (especially XGBoost depth/regularization)
- Stability testing of clustering assignments
- More rigorous validation of cluster utility in supervised models
- Feature interaction analysis (beyond permutation importance)

---

## Tools & Libraries

- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost
- Plotly / Matplotlib
- KaggleHub

---

## Notes

- PCA was used only for visualization, not for model training
- Clustering is exploratory and used for segmentation insight, not as a predictive feature