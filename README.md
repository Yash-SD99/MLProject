# 🎯 Intelligent Customer Segmentation with Automatic Cluster Selection

An end-to-end Machine Learning application that automatically discovers meaningful customer segments using unsupervised learning, with an interactive Streamlit dashboard for business intelligence and customer analytics.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Technologies Used](#technologies-used)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Automatic Cluster Selection](#automatic-cluster-selection)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Application Pages](#application-pages)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## 🎯 Introduction

This project implements an intelligent customer segmentation system that leverages unsupervised machine learning to automatically group mall customers based on their demographic characteristics, annual income, and spending behavior. Unlike traditional approaches that require manual selection of cluster count, this system **automatically determines the optimal number of clusters** using a transparent, metrics-based selection strategy.

The application provides a professional, interactive web dashboard built with Streamlit that enables business stakeholders to explore customer segments, understand their characteristics, and derive actionable marketing strategies.

---

## ❓ Problem Statement

Retail businesses collect vast amounts of customer data but often struggle to extract actionable insights from it. Manually segmenting customers is subjective, time-consuming, and doesn't scale. There is a need for an automated system that can:

1. Discover natural groupings within customer data without prior labels.
2. Automatically determine the optimal number of customer segments.
3. Generate meaningful business interpretations and marketing recommendations.
4. Provide an interactive interface for exploring customer segments.

---

## 🎯 Objectives

1. **Automated Segmentation**: Build an unsupervised ML pipeline that automatically segments customers without manual intervention.
2. **Optimal Cluster Selection**: Implement a transparent, metrics-based approach to automatically determine the best number of clusters and clustering algorithm.
3. **Business Intelligence**: Generate actionable customer profiles, marketing strategies, and business insights from discovered segments.
4. **Interactive Dashboard**: Develop a professional web application for exploring and visualizing customer segments.
5. **Prediction Capability**: Allow prediction of which segment a new customer belongs to.
6. **Reproducibility**: Ensure the entire pipeline is reproducible with saved models and preprocessing objects.

---

## 📊 Dataset Description

**Mall Customer Segmentation Dataset**

| Feature | Description | Type |
|---------|-------------|------|
| CustomerID | Unique identifier for each customer | Integer |
| Gender | Customer's gender (Male/Female) | Categorical |
| Age | Customer's age in years | Integer (18-70) |
| Annual Income (k$) | Annual income in thousands of dollars | Integer (15-137) |
| Spending Score (1-100) | Score assigned based on spending behavior and purchasing data | Integer (1-100) |

- **Total Records**: 200 customers
- **Missing Values**: None
- **Duplicate Records**: None

> **Note**: The Spending Score is a composite metric assigned by the mall based on customer behavior and purchasing patterns. It is not derived from individual transaction histories.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **scikit-learn** | Machine learning (KMeans, GMM, metrics, preprocessing) |
| **Matplotlib** | Static data visualization |
| **Seaborn** | Statistical data visualization |
| **Plotly** | Interactive data visualization |
| **Streamlit** | Web application framework |
| **Joblib** | Model persistence and serialization |

---

## 🤖 Machine Learning Algorithms

### K-Means Clustering
- Partitions data into K clusters by minimizing within-cluster sum of squares (inertia).
- Fast, scalable, and effective for spherical clusters.
- Tested across K = 2 to 10.

### Gaussian Mixture Model (GMM)
- A probabilistic model that represents data as a mixture of Gaussian distributions.
- More flexible than K-Means — can model elliptical clusters.
- Uses Expectation-Maximization (EM) algorithm for parameter estimation.
- Evaluated using Bayesian Information Criterion (BIC).

---

## 🔬 Automatic Cluster Selection Methodology

This is the **core innovation** of the project. Rather than arbitrarily picking K=5 or relying solely on the elbow method, the system uses a **composite scoring approach**:

### Step 1: Evaluate Multiple Configurations
Both KMeans and GMM are evaluated across K = 2 to 10 with multiple metrics.

### Step 2: Compute Composite Score (KMeans)
For each K value, a composite score is computed:

$$\text{Score}(K) = 0.4 \times \hat{S}(K) + 0.3 \times (1 - \hat{DB}(K)) + 0.3 \times \hat{CH}(K)$$

Where:
- $\hat{S}(K)$ = Min-Max normalized Silhouette Score (higher is better)
- $\hat{DB}(K)$ = Min-Max normalized Davies-Bouldin Index (lower is better, so inverted)
- $\hat{CH}(K)$ = Min-Max normalized Calinski-Harabasz Score (higher is better)

### Step 3: Compare Algorithms
The best KMeans configuration (highest composite score) is compared with the best GMM configuration (lowest BIC with best silhouette).

### Step 4: Select Optimal Configuration
The algorithm and K with the highest overall quality is selected, with a detailed explanation of why.

---

## 📏 Evaluation Metrics

| Metric | Range | Optimal | Description |
|--------|-------|---------|-------------|
| **Silhouette Score** | [-1, 1] | Higher | Measures how similar a point is to its own cluster vs neighboring clusters |
| **Davies-Bouldin Index** | [0, ∞) | Lower | Average similarity ratio of each cluster with its most similar cluster |
| **Calinski-Harabasz Score** | [0, ∞) | Higher | Ratio of between-cluster dispersion to within-cluster dispersion |
| **Inertia (WCSS)** | [0, ∞) | Lower | Within-cluster sum of squared distances to centroid |
| **BIC** | (-∞, ∞) | Lower | Bayesian Information Criterion — balances model fit with complexity |

---

## 🏗️ Project Architecture

```
customer-segmentation/
│
├── data/
│   └── Mall_Customers.csv          # Original dataset
│
├── models/
│   ├── clustering_model.pkl         # Trained clustering model
│   └── scaler.pkl                   # Fitted StandardScaler
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data loading, cleaning, scaling
│   ├── clustering.py                # KMeans, GMM, auto-selection
│   ├── cluster_evaluation.py        # Metrics and evaluation
│   ├── customer_profiling.py        # Segment profiles and naming
│   └── recommendations.py           # Business insights and strategies
│
├── app.py                           # Streamlit dashboard
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore rules
```

### Module Descriptions

| Module | Responsibility |
|--------|---------------|
| `data_preprocessing.py` | Data loading, inspection, cleaning, encoding, feature selection, scaling |
| `clustering.py` | KMeans and GMM implementation, automatic cluster selection with composite scoring |
| `cluster_evaluation.py` | Clustering quality evaluation, algorithm comparison, metrics reporting |
| `customer_profiling.py` | Cluster profiling, segment naming, high-value customer identification |
| `recommendations.py` | Marketing strategies, business interpretations, targeting recommendations |
| `app.py` | Interactive Streamlit dashboard with 7 pages |

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd MLProject
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Launch the Dashboard
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### What Happens on First Run
1. The ML pipeline runs automatically.
2. Data is loaded, cleaned, and preprocessed.
3. Multiple clustering configurations are evaluated.
4. The optimal algorithm and cluster count are selected automatically.
5. The final model is trained and saved.
6. Customer profiles and recommendations are generated.
7. The dashboard becomes fully interactive.

---

## 📱 Application Pages

| # | Page | Description |
|---|------|-------------|
| 1 | **Executive Overview** | KPI cards, dataset summary, cluster distribution |
| 2 | **Exploratory Data Analysis** | Feature distributions, correlations, scatter plots |
| 3 | **Automatic Cluster Selection** | Elbow curve, metric comparisons, selection explanation |
| 4 | **Customer Segmentation** | Interactive 2D/3D cluster visualizations with filters |
| 5 | **Business Intelligence** | Segment profiles, marketing strategies, high-value customers |
| 6 | **Predict New Customer** | Real-time prediction for new customer data |
| 7 | **Download Reports** | Export segmentation results, summaries, and metrics |

---

## 📈 Results

### Automatically Discovered Customer Segments

The system automatically identifies distinct customer segments based on the interplay between Annual Income and Spending Score. Typical segments discovered include:

- **High Income, High Spenders** — Premium customers with strong purchasing power and willingness to spend
- **High Income, Low Spenders** — Affluent customers who are conservative with spending — prime targets for engagement campaigns
- **Low Income, High Spenders** — Budget-conscious but active shoppers — value loyalty programs
- **Low Income, Low Spenders** — Price-sensitive customers requiring value-oriented approaches
- **Average Customers** — Moderate income and spending — the mainstream customer base

> The actual segments and their names are derived dynamically from the data, not hardcoded.

---

## 🔮 Future Improvements

1. **DBSCAN Integration** — Add density-based clustering for discovering arbitrarily-shaped clusters.
2. **Hierarchical Clustering** — Implement agglomerative clustering with dendrogram visualization.
3. **Dimensionality Reduction** — Apply PCA or t-SNE for better high-dimensional visualization.
4. **Real-time Data Ingestion** — Connect to live customer databases for streaming segmentation.
5. **A/B Test Integration** — Link segments to marketing A/B test results for strategy validation.
6. **Advanced Profiling** — Incorporate RFM (Recency, Frequency, Monetary) analysis when transaction data is available.
7. **API Endpoint** — Expose the prediction model via a REST API for integration with other systems.
8. **Automated Reporting** — Schedule periodic PDF/email reports with updated segmentation results.
9. **Multi-Dataset Support** — Allow users to upload their own customer datasets.
10. **Cloud Deployment** — Deploy on Streamlit Cloud, AWS, or GCP for production use.

---

## 📄 License

This project is developed for academic and portfolio demonstration purposes.

---

## 👤 Author

Built as a Computer Engineering academic project demonstrating end-to-end machine learning application development.

---

*Built with ❤️ using Python, scikit-learn, and Streamlit*
