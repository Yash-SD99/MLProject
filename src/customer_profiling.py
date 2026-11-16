"""
customer_profiling.py
---------------------
Builds rich customer-segment profiles and identifies high-value customers.
"""

import numpy as np
import pandas as pd


class CustomerProfiler:
    """Create, name, and summarise customer cluster profiles."""

    # ------------------------------------------------------------------ #
    #  Cluster profiles
    # ------------------------------------------------------------------ #
    def create_cluster_profiles(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        feature_names: list,
    ) -> list:
        """
        Build a profile dict for each cluster.

        Parameters
        ----------
        df            : Processed DataFrame (all original cols + Gender_Encoded).
        labels        : Cluster labels array (length == len(df)).
        feature_names : e.g. ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

        Returns
        -------
        List of profile dicts, one per cluster.
        """
        df = df.copy()
        df["Cluster"] = labels
        total = len(df)

        profiles = []
        for cluster_id in sorted(df["Cluster"].unique()):
            cluster_df = df[df["Cluster"] == cluster_id]
            count = len(cluster_df)

            # Gender distribution
            gender_counts = cluster_df["Gender"].value_counts().to_dict()
            male_count = gender_counts.get("Male", 0)
            female_count = gender_counts.get("Female", 0)

            profile = {
                "cluster_id": int(cluster_id),
                "count": count,
                "percentage": round(count / total * 100, 2),
                "mean_age": round(float(cluster_df["Age"].mean()), 2),
                "mean_income": round(float(cluster_df["Annual Income (k$)"].mean()), 2),
                "mean_spending_score": round(float(cluster_df["Spending Score (1-100)"].mean()), 2),
                "male_count": int(male_count),
                "female_count": int(female_count),
            }
            profiles.append(profile)

        print(f"[CustomerProfiler] Created profiles for {len(profiles)} clusters.")
        return profiles

    # ------------------------------------------------------------------ #
    #  Segment naming
    # ------------------------------------------------------------------ #
    def assign_segment_names(self, profiles: list) -> list:
        """
        Assign descriptive segment names based on data-derived thresholds.

        Thresholds (relative to overall mean across ALL clusters, weighted
        by cluster size):
          - > mean * 1.2  →  High
          - < mean * 0.8  →  Low
          - otherwise     →  Average

        Segment name is composed as "<Income Level> Income, <Spending Level> Spenders".
        """
        # Compute overall weighted means
        total_customers = sum(p["count"] for p in profiles)
        overall_income = sum(p["mean_income"] * p["count"] for p in profiles) / total_customers
        overall_spending = sum(p["mean_spending_score"] * p["count"] for p in profiles) / total_customers

        high_income_threshold = overall_income * 1.2
        low_income_threshold = overall_income * 0.8
        high_spending_threshold = overall_spending * 1.2
        low_spending_threshold = overall_spending * 0.8

        print(f"[CustomerProfiler] Overall mean income: {overall_income:.2f}, "
              f"spending: {overall_spending:.2f}")

        for profile in profiles:
            # Income level
            if profile["mean_income"] > high_income_threshold:
                income_level = "High"
            elif profile["mean_income"] < low_income_threshold:
                income_level = "Low"
            else:
                income_level = "Average"

            # Spending level
            if profile["mean_spending_score"] > high_spending_threshold:
                spending_level = "High"
            elif profile["mean_spending_score"] < low_spending_threshold:
                spending_level = "Low"
            else:
                spending_level = "Average"

            # Compose name
            if income_level == "Average" and spending_level == "Average":
                segment_name = "Average Customers"
            else:
                segment_name = f"{income_level} Income, {spending_level} Spenders"

            profile["segment_name"] = segment_name
            print(f"  Cluster {profile['cluster_id']}: {segment_name}")

        return profiles

    # ------------------------------------------------------------------ #
    #  Summary DataFrame
    # ------------------------------------------------------------------ #
    def get_cluster_summary_df(self, profiles: list) -> pd.DataFrame:
        """Return a clean summary DataFrame from the profiles list."""
        rows = []
        for p in profiles:
            rows.append({
                "Cluster": p["cluster_id"],
                "Segment Name": p.get("segment_name", "N/A"),
                "Customers": p["count"],
                "Percentage": p["percentage"],
                "Avg Age": p["mean_age"],
                "Avg Income": p["mean_income"],
                "Avg Spending Score": p["mean_spending_score"],
                "Male Count": p["male_count"],
                "Female Count": p["female_count"],
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  High-value customer identification
    # ------------------------------------------------------------------ #
    def identify_high_value_customers(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        top_percentile: int = 20,
    ) -> pd.DataFrame:
        """
        Identify customers whose Annual Income AND Spending Score are both
        in the top *top_percentile* percent of the dataset.

        Returns a DataFrame of those high-value customers with their cluster.
        """
        df = df.copy()
        df["Cluster"] = labels

        income_threshold = np.percentile(df["Annual Income (k$)"], 100 - top_percentile)
        spending_threshold = np.percentile(df["Spending Score (1-100)"], 100 - top_percentile)

        high_value = df[
            (df["Annual Income (k$)"] >= income_threshold)
            & (df["Spending Score (1-100)"] >= spending_threshold)
        ].copy()

        print(f"[CustomerProfiler] Found {len(high_value)} high-value customers "
              f"(top {top_percentile}% income >= {income_threshold}, "
              f"spending >= {spending_threshold})")
        return high_value
