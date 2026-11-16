"""
data_preprocessing.py
---------------------
Handles all data loading, inspection, cleaning, encoding, feature selection,
and scaling for the Customer Segmentation project.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class DataPreprocessor:
    """End-to-end data preprocessing pipeline for Mall Customers data."""

    # ------------------------------------------------------------------ #
    #  Loading & Inspection
    # ------------------------------------------------------------------ #
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load a CSV file and return a pandas DataFrame."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at: {filepath}")
        df = pd.read_csv(filepath)
        print(f"[DataPreprocessor] Loaded {len(df)} rows, {len(df.columns)} columns from {filepath}")
        return df

    def inspect_data(self, df: pd.DataFrame) -> dict:
        """Return a dictionary with key descriptive statistics of the DataFrame."""
        inspection = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_count": int(df.duplicated().sum()),
            "head": df.head(),
            "describe": df.describe(include="all"),
        }
        print(f"[DataPreprocessor] Shape: {inspection['shape']}, "
              f"Missing values: {sum(inspection['missing_values'].values())}, "
              f"Duplicates: {inspection['duplicate_count']}")
        return inspection

    # ------------------------------------------------------------------ #
    #  Cleaning
    # ------------------------------------------------------------------ #
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop or fill missing values. Returns a cleaned DataFrame."""
        missing_total = df.isnull().sum().sum()
        if missing_total == 0:
            print("[DataPreprocessor] No missing values found — skipping.")
            return df

        # For numerical columns, fill with median; for categorical, fill with mode.
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"[DataPreprocessor] Filled '{col}' missing values with median ({median_val}).")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"[DataPreprocessor] Filled '{col}' missing values with mode ({mode_val}).")
        return df

    # ------------------------------------------------------------------ #
    #  Encoding
    # ------------------------------------------------------------------ #
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the 'Gender' column using LabelEncoder.
        Female -> 0, Male -> 1.  Adds a new column 'Gender_Encoded'.
        """
        df = df.copy()
        le = LabelEncoder()
        df["Gender_Encoded"] = le.fit_transform(df["Gender"])
        print(f"[DataPreprocessor] Encoded 'Gender': {dict(zip(le.classes_, le.transform(le.classes_)))}")
        return df

    # ------------------------------------------------------------------ #
    #  Feature Selection & Scaling
    # ------------------------------------------------------------------ #
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the three clustering features and return as a DataFrame."""
        feature_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
        for col in feature_cols:
            if col not in df.columns:
                raise KeyError(f"Expected column '{col}' not found in DataFrame.")
        feature_df = df[feature_cols].copy()
        print(f"[DataPreprocessor] Selected features: {feature_cols}")
        return feature_df

    def scale_features(self, feature_df: pd.DataFrame):
        """
        Apply StandardScaler to the feature DataFrame.

        Returns
        -------
        scaled_features : np.ndarray
            Scaled feature array.
        scaler : StandardScaler
            The fitted scaler object (for inverse transforms later).
        """
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        print("[DataPreprocessor] Features scaled with StandardScaler.")
        return scaled_features, scaler

    # ------------------------------------------------------------------ #
    #  Distribution Stats
    # ------------------------------------------------------------------ #
    def get_feature_distributions(self, df: pd.DataFrame) -> dict:
        """Return distribution statistics for each numerical column."""
        distributions = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            distributions[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
                "skewness": float(df[col].skew()),
                "kurtosis": float(df[col].kurtosis()),
            }
        return distributions

    # ------------------------------------------------------------------ #
    #  Full Pipeline
    # ------------------------------------------------------------------ #
    def run_pipeline(self, filepath: str) -> dict:
        """
        Execute the full preprocessing pipeline end-to-end.

        Returns
        -------
        dict with keys:
            original_df, processed_df, features, scaled_features,
            scaler, feature_names, inspection
        """
        # 1. Load
        original_df = self.load_data(filepath)

        # 2. Inspect
        inspection = self.inspect_data(original_df)

        # 3. Handle missing values
        processed_df = self.handle_missing_values(original_df)

        # 4. Encode categorical features
        processed_df = self.encode_features(processed_df)

        # 5. Select clustering features
        feature_names = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
        features = self.select_features(processed_df)

        # 6. Scale
        scaled_features, scaler = self.scale_features(features)

        # 7. Persist the scaler
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"[DataPreprocessor] Scaler saved to {scaler_path}")

        return {
            "original_df": original_df,
            "processed_df": processed_df,
            "features": features,
            "scaled_features": scaled_features,
            "scaler": scaler,
            "feature_names": feature_names,
            "inspection": inspection,
        }
