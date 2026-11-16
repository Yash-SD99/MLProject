"""
clustering.py
-------------
Clustering engine supporting KMeans and Gaussian Mixture Models.
Includes automated optimal-cluster selection with composite scoring.
"""

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib


class ClusteringEngine:
    """Run, evaluate, and auto-select the best clustering configuration."""

    # ------------------------------------------------------------------ #
    #  Single-run helpers
    # ------------------------------------------------------------------ #
    def run_kmeans(self, scaled_features: np.ndarray, k: int):
        """
        Fit a KMeans model with *k* clusters.

        Returns
        -------
        model : KMeans
        labels : np.ndarray
        """
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        labels = model.fit_predict(scaled_features)
        print(f"[ClusteringEngine] KMeans fitted with k={k}")
        return model, labels

    def run_gmm(self, scaled_features: np.ndarray, n_components: int):
        """
        Fit a GaussianMixture model with *n_components* components.

        Returns
        -------
        model : GaussianMixture
        labels : np.ndarray
        """
        model = GaussianMixture(n_components=n_components, covariance_type="full",
                                n_init=5, max_iter=300, random_state=42)
        model.fit(scaled_features)
        labels = model.predict(scaled_features)
        print(f"[ClusteringEngine] GMM fitted with n_components={n_components}")
        return model, labels

    # ------------------------------------------------------------------ #
    #  Range evaluation
    # ------------------------------------------------------------------ #
    def evaluate_kmeans_range(self, scaled_features: np.ndarray, k_range=range(2, 11)) -> dict:
        """
        Evaluate KMeans for every k in *k_range*.

        Returns dict with lists: inertia, silhouette_score,
        davies_bouldin_score, calinski_harabasz_score.
        """
        metrics = {
            "inertia": [],
            "silhouette_score": [],
            "davies_bouldin_score": [],
            "calinski_harabasz_score": [],
        }
        for k in k_range:
            model, labels = self.run_kmeans(scaled_features, k)
            metrics["inertia"].append(float(model.inertia_))
            metrics["silhouette_score"].append(float(silhouette_score(scaled_features, labels)))
            metrics["davies_bouldin_score"].append(float(davies_bouldin_score(scaled_features, labels)))
            metrics["calinski_harabasz_score"].append(float(calinski_harabasz_score(scaled_features, labels)))
        return metrics

    def evaluate_gmm_range(self, scaled_features: np.ndarray, k_range=range(2, 11)) -> dict:
        """
        Evaluate GMM for every k in *k_range*.

        Returns dict with lists: bic, aic, silhouette_score.
        """
        metrics = {
            "bic": [],
            "aic": [],
            "silhouette_score": [],
        }
        for k in k_range:
            model, labels = self.run_gmm(scaled_features, k)
            metrics["bic"].append(float(model.bic(scaled_features)))
            metrics["aic"].append(float(model.aic(scaled_features)))
            metrics["silhouette_score"].append(float(silhouette_score(scaled_features, labels)))
        return metrics

    # ------------------------------------------------------------------ #
    #  Auto-select optimal clusters (KEY METHOD)
    # ------------------------------------------------------------------ #
    def auto_select_optimal_clusters(self, scaled_features: np.ndarray, k_range=range(2, 11)) -> dict:
        """
        Automatically determine the best clustering algorithm and k.

        Strategy
        --------
        1. Run both KMeans and GMM evaluations over *k_range*.
        2. For KMeans, compute a composite score per k:
             composite = 0.4 * norm_silhouette
                       + 0.3 * (1 - norm_davies_bouldin)
                       + 0.3 * norm_calinski_harabasz
           where each metric is min-max normalised to [0, 1].
        3. For GMM, pick the k with the lowest BIC.
        4. Compare the best KMeans composite score with the best GMM
           silhouette score and choose the winner.

        Returns dict with:
            best_algorithm, best_k, best_score, explanation,
            kmeans_metrics, gmm_metrics, composite_scores
        """
        k_list = list(k_range)

        # --- Evaluate both algorithms ---
        kmeans_metrics = self.evaluate_kmeans_range(scaled_features, k_range)
        gmm_metrics = self.evaluate_gmm_range(scaled_features, k_range)

        # --- Helper: min-max normalisation ---
        def _minmax(values):
            arr = np.array(values, dtype=float)
            mn, mx = arr.min(), arr.max()
            if mx - mn == 0:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        # --- KMeans composite scores ---
        norm_sil = _minmax(kmeans_metrics["silhouette_score"])
        norm_db = _minmax(kmeans_metrics["davies_bouldin_score"])
        norm_ch = _minmax(kmeans_metrics["calinski_harabasz_score"])

        composite_scores = 0.4 * norm_sil + 0.3 * (1 - norm_db) + 0.3 * norm_ch
        composite_list = composite_scores.tolist()

        best_km_idx = int(np.argmax(composite_scores))
        best_km_k = k_list[best_km_idx]
        best_km_score = float(composite_scores[best_km_idx])

        # --- GMM: best by BIC (lower is better) ---
        best_gmm_bic_idx = int(np.argmin(gmm_metrics["bic"]))
        best_gmm_k = k_list[best_gmm_bic_idx]
        best_gmm_sil = float(gmm_metrics["silhouette_score"][best_gmm_bic_idx])

        # --- Compare ---
        if best_km_score >= best_gmm_sil:
            best_algorithm = "KMeans"
            best_k = best_km_k
            best_score = best_km_score
            explanation = (
                f"KMeans was selected with k={best_k}. "
                f"Its composite score ({best_score:.4f}) — combining normalised silhouette "
                f"(weight 0.4), inverted Davies-Bouldin (weight 0.3), and Calinski-Harabasz "
                f"(weight 0.3) — was equal to or higher than the best GMM silhouette score "
                f"({best_gmm_sil:.4f} at k={best_gmm_k}). "
                f"KMeans with k={best_k} therefore provides the best trade-off between "
                f"cluster compactness and separation."
            )
        else:
            best_algorithm = "GMM"
            best_k = best_gmm_k
            best_score = best_gmm_sil
            explanation = (
                f"GMM was selected with k={best_k}. "
                f"Its silhouette score ({best_gmm_sil:.4f}) exceeded the best KMeans "
                f"composite score ({best_km_score:.4f} at k={best_km_k}). "
                f"GMM with {best_k} components better captures the underlying cluster "
                f"structure, possibly because the clusters are not strictly spherical."
            )

        print(f"[ClusteringEngine] Auto-selected: {best_algorithm} with k={best_k} "
              f"(score={best_score:.4f})")

        return {
            "best_algorithm": best_algorithm,
            "best_k": best_k,
            "best_score": best_score,
            "explanation": explanation,
            "kmeans_metrics": kmeans_metrics,
            "gmm_metrics": gmm_metrics,
            "composite_scores": composite_list,
        }

    # ------------------------------------------------------------------ #
    #  Final model training & persistence
    # ------------------------------------------------------------------ #
    def train_final_model(self, scaled_features: np.ndarray, algorithm: str, k: int):
        """
        Train the final clustering model.

        Parameters
        ----------
        algorithm : str   — 'KMeans' or 'GMM'
        k         : int   — number of clusters / components

        Returns
        -------
        model, labels
        """
        if algorithm == "KMeans":
            return self.run_kmeans(scaled_features, k)
        elif algorithm == "GMM":
            return self.run_gmm(scaled_features, k)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'KMeans' or 'GMM'.")

    def save_model(self, model, filepath: str) -> None:
        """Persist a trained model to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"[ClusteringEngine] Model saved to {filepath}")
