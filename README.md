# 🧳 Voyage AI — Traveller Segmentation Engine

> Unsupervised clustering pipeline that segments 5,000 travellers into 6 behavioural personas using K-Means and GMM on real + synthetic travel data.

---

## 📋 Overview

Voyage AI ingests a real-world Kaggle travel dataset, augments it with Gaussian Copula synthetic data to reach 5,000 users, engineers 14 behavioural features, and clusters travellers into 6 named personas — enabling personalised recommendations, targeted marketing, and demand forecasting for a travel platform.

**Key results:**
- Silhouette Score: `0.35–0.40` ✓
- Dataset: `5,000 users` (real + synthetic)
- Features: `14` engineered features
- Clusters: `6` named traveller personas
- Algorithms: `K-Means` + `GMM` (compared side-by-side)

---

## 🗂️ Project Structure

```
voyage-ai/
│
├── VoyageAi_27feb_updated.ipynb     # Main pipeline notebook (all 13 steps)
├── Travel details dataset.csv       # Raw Kaggle input data
│
├── outputs/
│   ├── clustered_travellers_5k_optimized.csv   # Final labelled dataset
│   ├── kmeans_model.pkl                         # Saved K-Means model
│   ├── gmm_model.pkl                            # Saved GMM model
│   ├── scaler.pkl                               # StandardScaler
│   └── pca_model.pkl                            # PCA transform (3 components)
│
└── plots/
    ├── 00_pca_scree.png                  # PCA variance scree plot
    ├── 01_cluster_selection.png          # Silhouette + elbow curves
    ├── 02_algorithm_comparison.png       # K-Means vs GMM metrics
    ├── 03_cluster_sizes.png              # User count per cluster
    ├── 04_cluster_heatmap.png            # Feature heatmap by cluster
    ├── 05_pca_scatter.png                # PCA scatter coloured by cluster
    ├── 06_traveller_personas.png         # All 6 persona cards
    ├── 07_gmm_confidence.png             # GMM assignment confidence
    ├── boxplot_individual_parameters.png
    ├── boxplot_combined_normalized.png
    └── boxplot_before_after_outliers.png
```

---

## ⚙️ Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Import libraries |
| 2 | Load & clean raw Kaggle data |
| 3 | Encode categorical features |
| 4 | Gaussian Copula synthetic data generation |
| 5 | Feature engineering (14 features) |
| 6 | Outlier removal via Isolation Forest (4%) |
| 7 | StandardScaler + PCA (3 components, ~70% variance) |
| 8 | Find optimal k via silhouette scan (k=2–10) |
| 9 | Train K-Means & GMM |
| 10 | Algorithm comparison (Silhouette, Davies-Bouldin, Calinski-Harabasz) |
| 11 | Boxplot visualisations of key parameters |
| 12 | Save models & labelled CSV |
| 13 | Cluster profiling & traveller personas |

---

## 👤 Traveller Personas

| Cluster | Persona | Description |
|---------|---------|-------------|
| 0 | 🏕️ Budget Backpacker | Long trips on a tight daily budget. Values experiences over comfort. Prefers hostels, buses and trains. |
| 1 | 🎒 Weekend Wanderer | Short budget getaways of 1–4 days. Often solo or couples. Ground transport, budget hotels. |
| 2 | 🌍 Extended Explorer | Long, well-funded trips. Slow travel or work-travel blend. Heavy spend on accommodation. |
| 3 | ✈️ Luxury Jet-Setter | High-spend flyers targeting premium destinations. Hotels or resorts. Flies almost exclusively. |
| 4 | 🛳️ Comfort Cruiser | Mature travellers with mid-to-high budgets. Prioritise reliability and comfort over adventure. |
| 5 | 🏖️ Family Vacationer | Mid-budget travellers with longer stays. Mix of transport modes. Resorts or apartments preferred. |

---

## 🔧 Features Engineered (14 total)

| Feature | Description |
|---------|-------------|
| `log_accom` | Log-transformed accommodation cost |
| `log_trans` | Log-transformed transport cost |
| `log_total` | Log-transformed total trip cost |
| `cost_per_day` | Total cost ÷ trip duration |
| `accom_share` | Accommodation as % of total budget |
| `age` | Traveller age |
| `age_bin` | Age binned into 4 groups |
| `duration` | Trip length in days |
| `duration_cat` | Duration binned (short/mid/long) |
| `season_sin` | Cyclic sine encoding of travel month |
| `season_cos` | Cyclic cosine encoding of travel month |
| `is_flight` | Binary flag: flew vs other transport |
| `accom_type` | Encoded accommodation type |
| `trans_type` | Encoded transport type |

---

## 📊 Model Performance

| Metric | K-Means | GMM |
|--------|---------|-----|
| Silhouette ↑ | ~0.37 | ~0.36 |
| Davies-Bouldin ↓ | lower is better | — |
| Calinski-Harabasz ↑ | higher is better | — |
| Avg Assignment Confidence | N/A | ~0.85+ |

---

## 🚀 How to Run

**Requirements:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib
```

**Run the notebook:**
1. Place `Travel details dataset.csv` in the same folder as the notebook
2. Open `VoyageAi_27feb_updated.ipynb` in Jupyter
3. Run all cells top to bottom (Kernel → Restart & Run All)
4. All plots and model files will be saved to the working directory

**To use saved models on new data:**
```python
import joblib, numpy as np

scaler = joblib.load('scaler.pkl')
pca    = joblib.load('pca_model.pkl')
kmeans = joblib.load('kmeans_model.pkl')

# new_data: DataFrame with the same 14 FEATURES columns
X_scaled = scaler.transform(new_data[FEATURES])
X_pca    = pca.transform(X_scaled)
cluster  = kmeans.predict(X_pca)
```

---

## 📁 Data

**Source:** [Kaggle — Travel Details Dataset](https://www.kaggle.com/)

**Raw columns used:**
- `Traveler age`, `Duration (days)`, `Accommodation cost`, `Transportation cost`
- `Accommodation type`, `Transportation type`, `Start date`

**Synthetic data:** ~4,861 records generated via Gaussian Copula, stratified by spending tier. Preserves real-world correlations between cost variables.

---

## 📌 Design Decisions

**Why Gaussian Copula?** Simple random augmentation breaks inter-variable correlations (e.g. high accommodation cost co-occurring with high transport cost). The Copula preserves the full joint distribution of the real data.

**Why cyclic month encoding?** A raw integer treats December (12) and January (1) as far apart. Sine/cosine encoding makes them adjacent, which is correct for seasonality.

**Why PCA to 3 components?** Captures ~70% of variance while reducing noise. Empirically gives the best silhouette score vs higher component counts.

**Why both K-Means and GMM?** K-Means gives hard, fast cluster assignments. GMM provides soft probabilistic confidence scores — useful for identifying borderline travellers who sit between two segments.

