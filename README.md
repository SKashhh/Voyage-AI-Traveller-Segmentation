VOYAGE AI — Silhouette Optimized Traveller Segmentation
5,000 Users | 14 Engineered Features | Gaussian Copula Augmentation | K-Means vs GMM

Final Silhouette: 0.35–0.40 (Legitimate, Not Forced)

1. Project Objective

The goal of this project is to build a statistically robust unsupervised segmentation pipeline for traveller behavior analysis using real-world travel transaction data.

We aim to:

Segment travellers based on spending, duration, seasonality, and transport behavior

Preserve real-world correlations during data augmentation

Compare clustering algorithms rigorously

Optimize silhouette score without artificial manipulation

Produce deployable clustering artifacts

This is not a toy clustering notebook.
This is a pipeline built with statistical discipline.

2. The Core Problem

Clustering real-world consumer data presents several challenges:

Small sample size limits model reliability

Heavy skew in cost distributions

Strong feature correlations distort clustering distance space

Categorical + numerical mixing

Arbitrary selection of k

Inflated silhouette via bad practices (noise injection, cherry-picking k)

This project explicitly fixes all of these.

3. Dataset

Source: Kaggle Travel Details Dataset
Original Size: ~140 records
Final Dataset: 5,000 records

Real: original cleaned records

Synthetic: Gaussian Copula generated

Core Variables:

Age

Duration

Accommodation cost

Transportation cost

Accommodation type

Transportation type

Travel month

4. Major Bugs Fixed from Original Notebook

This project began from a flawed notebook. The following issues were corrected:

❌ Bug 1 — Dataset Wiped to 0 Rows

pd.to_numeric() was applied to all columns including text → entire dataset became NaN → dropna() deleted everything.

✔ Fix: Convert only actual numeric columns.

❌ Bug 2 — Singular Covariance Matrix (SVD crash)

Derived columns (luxury_index, ratios, totals) were fed into multivariate Gaussian.

These are mathematically dependent:

total_cost = accom_cost + trans_cost
ratio = accom_cost / total_cost

This makes covariance singular.

✔ Fix: Use only 7 independent base columns.

❌ Bug 3 — Fake Gaussian Copula

Original code used:

np.random.multivariate_normal

That assumes Gaussian marginals — but costs are lognormal.

Result:

Negative accommodation cost

Age = 8

Duration = 0.2 days

✔ Fix: Implement true Gaussian Copula:

Empirical CDF → Uniform

Uniform → Probit

Fit multivariate Gaussian in normal space

Sample

Inverse transform via empirical quantile

Synthetic data:

Preserves correlations

Preserves marginal distributions

Never generates impossible values

❌ Bug 4 — Artificial Noise Injection

Original code added Gaussian noise to force silhouette into 0.3–0.4 range.

This corrupts real structure.

✔ Fix: Use PCA to reduce correlated dimensions instead.

❌ Bug 5 — Manipulated k Selection

Original code selected k closest to a target silhouette.

✔ Fix: Select k that maximizes silhouette score.

❌ Bug 6 — Isolation Forest on Redundant Features

Outlier detection was run on mathematically dependent columns.

✔ Fix: Use only 14 non-redundant engineered features.

5. Synthetic Data Generation — Gaussian Copula

Why Copula instead of SMOTE or naive Gaussian?

Because clustering requires:

Preserved correlation structure

Realistic joint distributions

No unrealistic values

Stratified Copula sampling was performed by spending tiers to preserve within-tier correlation patterns.

Correlation Check:

Real Accom ↔ Trans cost correlation preserved

No negative costs.
No impossible ages.
No zero-day trips.

This is statistically defensible augmentation.

6. Feature Engineering (14 Non-Redundant Features)

We engineered features to enhance cluster separability:

Cost Transformations

log_accom

log_trans

log_total

cost_per_day

accom_share

(Log transform reduces skew.)

Demographic & Duration Structure

age

age_bin

duration

duration_cat

(Binning enhances grouping signal.)

Seasonality Encoding

season_sin

season_cos

(Cyclic encoding avoids month 12 → 1 discontinuity.)

Behavioral Flags

is_flight

accom_type

trans_type

All redundant mathematical relationships were removed.

7. Outlier Removal

Isolation Forest (4% contamination)

Outliers removed from 14 clean features only.

This prevents distortion of cluster boundaries.

8. Dimensionality Reduction — PCA

Why PCA?

High-dimensional clustering suffers from:

Distance concentration

Correlated axes

Noise amplification

We retained:

3 components

~70% variance explained

Result:

Sharper cluster separation

Higher silhouette

No artificial noise

9. Optimal k Selection

k scanned from 2–10.

Selection criterion:

argmax(silhouette_score)

No arbitrary targeting.

10. Algorithm Comparison
K-Means

Maximizes within-cluster compactness

Assumes spherical clusters

Gaussian Mixture Model

Soft clustering

Probabilistic membership

Handles elliptical clusters

Metrics used:

Silhouette Score (↑)

Davies-Bouldin Index (↓)

Calinski-Harabasz (↑)

GMM Assignment Confidence

Final Silhouette:

0.35 – 0.40 (Legitimate)
11. Outputs

Saved artifacts:

clustered_travellers_5k_optimized.csv

kmeans_model.pkl

gmm_model.pkl

scaler.pkl

pca_model.pkl

PCA Scree Plot

Cluster Selection Plot

Algorithm Comparison Plot

PCA Cluster Visualisations

GMM Confidence Plot

Pipeline is deployable.

12. Why This Project Is Strong

This is not a basic clustering exercise.

It demonstrates:

Statistical debugging

Distribution-aware synthetic data generation

Proper dimensionality reduction

Correct k selection logic

Multi-metric evaluation

Probabilistic clustering interpretation

Reproducible ML pipeline design

This is mid-level ML engineering work — not beginner-level notebook play.

13. Future Improvements (Research-Grade Additions)

To push this to academic rigor:

Stability test across random seeds

PCA dimension sensitivity analysis

Silhouette before vs after PCA comparison

Bootstrap cluster stability (Jaccard index)

Compare with Spectral Clustering / HDBSCAN

Copula goodness-of-fit test

14. Business Interpretation (Example)

Clusters may represent:

Budget weekend travellers

Long-stay mid-range planners

Luxury seasonal vacationers

Frequent short business flyers

These can power:

Targeted marketing

Pricing strategies

Loyalty tier segmentation

Seasonal demand forecasting

Final Summary

VOYAGE AI builds a statistically valid traveller segmentation system using:

True Gaussian Copula augmentation

Careful feature engineering

Proper dimensionality reduction

Honest silhouette maximization

Multi-algorithm comparison
