🌍 VOYAGE AI — Silhouette Optimized Traveller Segmentation

5,000 Users | 14 Engineered Features | Gaussian Copula Augmentation | K-Means vs GMM

1. Project Overview

VOYAGE AI is a statistically disciplined unsupervised learning pipeline designed to segment traveller behavior using transaction-level travel data.

The project transforms a small real-world dataset into a robust 5,000-record analytical dataset using Gaussian Copula–based synthetic augmentation, followed by principled feature engineering, dimensionality reduction, and multi-metric cluster evaluation.

Final silhouette score achieved: 0.35–0.40 (without metric manipulation)

This project demonstrates methodological integrity in clustering — not heuristic experimentation.

2. Objectives

Clean and standardize real-world travel data

Generate statistically valid synthetic data while preserving joint distributions

Engineer meaningful, non-redundant features

Remove structural outliers

Reduce dimensionality via PCA

Select optimal number of clusters using silhouette maximization

Compare K-Means and Gaussian Mixture Models rigorously

Save deployable clustering artifacts

3. Dataset

Source: Kaggle Travel Details Dataset

Original Size: ~140 records
Final Size: ~5,000 records

Real cleaned records

Gaussian Copula synthetic augmentation

Core Variables

Traveller age

Trip duration

Accommodation cost

Transportation cost

Accommodation type

Transportation type

Travel month

4. Data Cleaning & Preprocessing
Key Corrections

Only numeric columns converted using pd.to_numeric

Categorical inconsistencies standardized (Plane → Flight)

Date parsing with travel month extraction

Missing categoricals filled (non-critical)

Rows dropped only if core numeric fields missing

Result: Cleaned real dataset ready for structured augmentation.

5. Gaussian Copula Synthetic Data Generation
Why Copula?

Naive multivariate Gaussian sampling assumes Gaussian marginals.
Travel costs are skewed and non-Gaussian.

The implemented Gaussian Copula:

Converts each feature to empirical CDF (rank-based)

Transforms to standard normal space (probit)

Fits multivariate Gaussian to capture dependency structure

Samples from that distribution

Inverse-maps back via empirical quantiles

Additional Enhancement

Stratified sampling by spending tiers preserves intra-tier correlation structure and strengthens cluster separation.

Guarantees

Correlations preserved

No negative costs

Age and duration clipped to realistic bounds

Synthetic values remain within real marginal range

Final dataset: Real + 4,861 synthetic samples ≈ 5,000 records.

6. Feature Engineering (14 Non-Redundant Features)

Feature design focused on geometric separability and statistical validity.

Cost Transformations

log_accom

log_trans

log_total

cost_per_day

accom_share

(Log transforms reduce skew and stabilize variance.)

Demographic & Duration Structure

age

age_bin

duration

duration_cat

(Binning enhances cluster separability.)

Seasonality Encoding

season_sin

season_cos

(Cyclic encoding avoids month discontinuity.)

Behavioral Indicators

is_flight

accom_type

trans_type

All redundant mathematical dependencies removed before clustering.

7. Outlier Removal

Isolation Forest
Contamination: 4%

Applied only to the 14 engineered features to prevent distortion of cluster boundaries.

Outliers removed prior to scaling and PCA.

8. Scaling & PCA
Standardization

All features standardized using StandardScaler.

PCA Reduction

3 principal components retained

~70% variance explained

Scree plot generated

Why PCA?

Reduces multicollinearity

Sharpens cluster geometry

Improves silhouette stability

Avoids artificial noise injection

9. Optimal Cluster Selection

k evaluated from 2 to 10.

Selection rule:

k = argmax(silhouette_score)

Additional evaluation metrics:

Davies-Bouldin Index (↓)

Calinski-Harabasz Index (↑)

No arbitrary silhouette targeting.

10. Model Training
K-Means

n_init = 100

max_iter = 800

Hard clustering

Evaluated on PCA space

Gaussian Mixture Model

Full covariance

Soft probabilistic clustering

Provides membership confidence per sample

11. Evaluation Metrics

For both models:

Silhouette Score

Davies-Bouldin Index

Calinski-Harabasz Index

For GMM additionally:

Average assignment confidence

Percentage of >90% confident assignments

Confidence heatmap visualization

Final Silhouette Score: 0.35–0.40

This range is realistic for behavioral clustering without artificial manipulation.

12. Visualizations Generated

PCA Scree Plot

Silhouette vs k Curve

Elbow Plot

K-Means vs GMM Metric Comparison

PCA Cluster Projection

GMM Confidence Distribution

GMM Confidence Map

Cluster Distribution (Pie + Bar)

All saved as high-resolution PNG files.

13. Output Artifacts

Saved for deployment and reuse:

clustered_travellers_5k_optimized.csv

kmeans_model.pkl

gmm_model.pkl

scaler.pkl

pca_model.pkl

Pipeline is fully reproducible.

14. Key Technical Strengths

Distribution-aware data augmentation

Avoidance of covariance singularity

Removal of feature redundancy before anomaly detection

Legitimate dimensionality reduction (no noise injection)

Honest silhouette maximization

Hard vs probabilistic clustering comparison

Deployment-ready model persistence

This is structured ML engineering, not exploratory experimentation.

15. Potential Business Applications

Clusters may represent:

Budget short-stay travellers

Mid-range planners

Seasonal luxury vacationers

Frequent flyers

Applications include:

Targeted marketing campaigns

Pricing optimization

Loyalty segmentation

Seasonal demand forecasting

Customer lifetime value modeling

16. How to Run

Place Travel details dataset.csv in project directory

Run notebook or script sequentially

Generated outputs saved automatically

Dependencies:

pandas

numpy

scikit-learn

matplotlib

seaborn

scipy

joblib

17. Final Summary

VOYAGE AI builds a statistically sound traveller segmentation pipeline by:

Expanding limited real data using Gaussian Copula augmentation

Engineering meaningful features

Removing structural outliers

Applying PCA-based dimensionality reduction

Selecting optimal k via silhouette maximization

Comparing K-Means and GMM rigorously

Saving deployable clustering models
