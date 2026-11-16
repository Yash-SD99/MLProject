"""
cluster_evaluation.py
---------------------
Evaluation utilities for comparing clustering results across algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class ClusterEvaluator:
    """Generate evaluation reports and comparison tables for clustering results."""

    def generate_evaluation_report(
        self,
        scaled_features: np.ndarray,
        labels: np.ndarray,
        algorithm_name: str,
    ) -> dict:
        """
        Compute core clustering quality metrics.

        Returns dict with:
            silhouette_score, davies_bouldin_score,
            calinski_harabasz_score, n_clusters, algorithm
        """
        n_clusters = len(set(labels))
        report = {
            "silhouette_score": float(silhouette_score(scaled_features, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(scaled_features, labels)),
            "calinski_harabasz_score": float(calinski_harabasz_score(scaled_features, labels)),
            "n_clusters": n_clusters,
            "algorithm": algorithm_name,
        }
        print(f"[ClusterEvaluator] {algorithm_name} — Silhouette: {report['silhouette_score']:.4f}, "
              f"DB: {report['davies_bouldin_score']:.4f}, CH: {report['calinski_harabasz_score']:.2f}")
        return report

    def compare_algorithms(
        self,
        kmeans_metrics: dict,
        gmm_metrics: dict,
        k_range,
    ) -> pd.DataFrame:
        """
        Build a side-by-side comparison DataFrame of KMeans vs GMM metrics.

        Parameters
        ----------
        kmeans_metrics : dict from ClusteringEngine.evaluate_kmeans_range
        gmm_metrics    : dict from ClusteringEngine.evaluate_gmm_range
        k_range        : range or list of k values evaluated

        Returns
        -------
        pd.DataFrame with one row per k and columns for both algorithms.
        """
        k_list = list(k_range)
        comparison = pd.DataFrame({
            "k": k_list,
            "KMeans_Silhouette": kmeans_metrics["silhouette_score"],
            "KMeans_Davies_Bouldin": kmeans_metrics["davies_bouldin_score"],
            "KMeans_Calinski_Harabasz": kmeans_metrics["calinski_harabasz_score"],
            "KMeans_Inertia": kmeans_metrics["inertia"],
            "GMM_Silhouette": gmm_metrics["silhouette_score"],
            "GMM_BIC": gmm_metrics["bic"],
            "GMM_AIC": gmm_metrics["aic"],
        })
        return comparison

    def get_cluster_sizes(self, labels: np.ndarray) -> dict:
        """Return a dict mapping each cluster id to its member count."""
        unique, counts = np.unique(labels, return_counts=True)
        sizes = {int(cid): int(cnt) for cid, cnt in zip(unique, counts)}
        print(f"[ClusterEvaluator] Cluster sizes: {sizes}")
        return sizes

    def generate_metrics_dataframe(
        self,
        kmeans_metrics: dict,
        gmm_metrics: dict,
        k_range,
    ) -> pd.DataFrame:
        """
        Produce a tidy DataFrame with all metrics for CSV export.

        Columns: k, Algorithm, Metric, Value
        """
        k_list = list(k_range)
        rows = []

        # KMeans metrics
        for metric_name, values in kmeans_metrics.items():
            for k, val in zip(k_list, values):
                rows.append({"k": k, "Algorithm": "KMeans", "Metric": metric_name, "Value": val})

        # GMM metrics
        for metric_name, values in gmm_metrics.items():
            for k, val in zip(k_list, values):
                rows.append({"k": k, "Algorithm": "GMM", "Metric": metric_name, "Value": val})

        metrics_df = pd.DataFrame(rows)
        return metrics_df
