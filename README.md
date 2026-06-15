# Telco Customer Churn Analysis

End-to-end machine learning project using the IBM Telco Customer Churn dataset to predict customer churn, compare classification models, identify key churn drivers, and segment customers through clustering and feature importance analysis.
---


## Project Workflow

1. Load and clean Telco customer data
2. Encode categorical features and prepare modeling dataset
3. Perform customer segmentation using PCA + K-Means clustering
4. Train and evaluate Logistic Regression, Random Forest, and XGBoost models using stratified cross-validation
5. Compare models using ROC-AUC, Precision, Recall, F1, and Average Precision
6. Analyze model drivers using permutation importance and SHAP
7. Visualize customer segments, model performance, and feature importance

---


## Objectives

- Predict customer churn using supervised learning models
- Compare multiple classification approaches under consistent evaluation
- Identify key drivers of churn using permutation feature importance and SHAP explainability
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

- 7,043 original customer records
- 7,032 records retained after cleaning
- 11 records removed due to missing `TotalCharges` values after numeric conversion
- Mix of categorical and numerical features
- Binary target: churn (Yes/No)

---

## Modeling Approaches

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
- Class imbalance handled through XGBoost weighting
- Evaluation using ROC-AUC, F1, precision, and recall


### Model Interpretability
- Permutation feature importance for model-agnostic global explanations 
- SHAP analysis for feature-level explanation of Logistic Regression predictions 
- Cluster-level behavioral analysis for segmentation insights

---

## Customer Segmentation Results

K-Means clustering was performed on PCA-reduced feature space for exploratory segmentation. A 3-cluster solution was selected as a balance between elbow-method results and business interpretability. Silhouette scores favored k=2, so the 3-cluster solution should be interpreted as a practical segmentation choice rather than a purely score-maximizing result.

| Cluster | Customers | Churn Rate | Total Revenue | Avg Tenure | Avg Monthly Charges |
|---------|----------:|-----------:|--------------:|-----------:|--------------------:|
| 0 | 1,520 | 7.4% | $1.01M | 30.7 months | $21.08 |
| 1 | 3,183 | 45.1% | $3.85M | 17.2 months | $69.01 |
| 2 | 2,329 | 13.7% | $11.19M | 54.4 months | $87.57 |

Cluster 1 represents the largest and highest-risk customer segment, while Cluster 2 represents the highest-revenue, longest-tenure segment and accounts for approximately 70% of total customer revenue.

## Model Results

| Model               | Accuracy  | Precision | Recall    | F1 Score  | ROC-AUC   | AP Score  |
| ------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | **0.804** | 0.658     | 0.547     | 0.597     | **0.844** | **0.653** |
| Random Forest       | 0.797     | **0.691** | 0.426     | 0.527     | 0.842     | 0.652     |
| XGBoost             | 0.757     | 0.531     | **0.739** | **0.618** | 0.833     | 0.646     |

*Metrics reported using stratified 5-fold cross-validation. ROC-AUC standard deviation ranged from 0.009–0.010 across models, indicating stable performance across folds.*

Logistic Regression achieved the strongest ROC-AUC and Average Precision while remaining highly interpretable. XGBoost, after class imbalance adjustment, achieved the highest recall and F1 score, making it a strong candidate for retention use cases where identifying as many churn-risk customers as possible is more important than minimizing false positives.

## Key Findings

- Logistic Regression achieved the highest ROC-AUC (0.844) and Average Precision Score (0.653), making it the strongest overall model for threshold-independent ranking performance.
- XGBoost achieved the highest recall (0.739) and F1 score (0.618), suggesting it may be preferable when the business objective is to identify as many potential churners as possible.
- Random Forest achieved the highest precision (0.691), but at the cost of substantially lower recall.
- Model selection depends on the business objective: Logistic Regression is preferable for interpretability and balanced discrimination, while XGBoost may be preferable for recall-oriented retention campaigns.
- Customer churn was strongly associated with contract structure, internet service type, pricing/service features, and customer lifecycle indicators.
- Customer segmentation identified distinct groups with materially different churn rates, tenure profiles, and revenue contribution.
- ROC-AUC standard deviation remained below 0.01 across all models, indicating stable performance across cross-validation folds.


---


## Visualizations

### Cluster Selection

![Elbow and Silhouette Analysis](churn_images/Elb_Sil.png)

### PCA Cluster Visualization

![PCA Cluster Visualization](churn_images/PCA_Clust.png)

### Model Performance

![ROC-AUC Comparison](churn_images/ROC_AUC.png)

![Precision-Recall Comparison](churn_images/Precision-Recall_Curve.png)

### Explainability

![Permutation Importance Heatmap](churn_images/Feature_Importance_Heatmap.png)

![SHAP Feature Importance](churn_images/SHAP_Bar.png)

![SHAP Beeswarm Plot](churn_images/SHAP_Beeswarm.png)


---

## Future Work

- Tune classification thresholds based on retention campaign cost/benefit assumptions
- Perform hyperparameter tuning for XGBoost and Random Forest
- Evaluate model calibration for probability-based churn scoring
- Test cluster stability across random seeds and alternative dimensionality reduction approaches
- Explore whether cluster membership improves supervised model performance
- Add a simple Streamlit or Dash dashboard for business-facing model results

---

## Tools & Libraries

- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost
- Plotly / Matplotlib
- KaggleHub
- SHAP
- Docker

---

## Notes

- PCA was used only for visualization, not for model training
- Clustering is exploratory and used for segmentation insight, not as a predictive feature

---

## Running the Project

### Run Locally

```bash
pip install -r requirements.txt
python src/run_churn_analysis.py
```

### Run with Docker

```bash
docker build -t churn-modeling .
docker run --rm churn-modeling
```

To persist generated outputs locally:

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/churn_images:/app/churn_images" \
  churn-modeling
```

The workflow generates model evaluation results, clustering outputs, feature importance summaries, and visualizations used throughout this project.


## Repository Structure

```text
churn_modeling/
├── src/
│   └── run_churn_analysis.py
├── churn_images/
│   ├── Elb_Sil.png
│   ├── Feature_Importance_Heatmap.png
│   ├── PCA_Clust.png
│   ├── Precision-Recall_Curve.png
│   ├── ROC_AUC.png
│   ├── SHAP_Bar.png
│   └── SHAP_Beeswarm.png
├── outputs/
│   ├── cluster_summary.csv
│   ├── driver_table.csv
│   ├── model_results.csv
│   └── ...
├── Dockerfile
├── Retention.ipynb
├── requirements.txt
└── README.md
```

