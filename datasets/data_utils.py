"""
Utility functions for the IU Akademie Python visual-analytics assignment.

Functions
---------
align_grid(df_a, df_b)
    Filter two data frames to their common x-values.

rmse(a, b)
    Root-mean-square error between two numeric arrays.

match_train_to_ideal(train_df, ideal_df)
    Return dict {train_curve : ideal_curve} with minimal RMSE.

compute_residuals_df(train_df, ideal_df)
    Return long-form DataFrame (curve, rmse).

build_residual_table(test_df, ideal_df)
    Classify each test point to the closest ideal curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Grid handling helpers
# ------------------------------------------------------------
def align_grid(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return df_a and df_b filtered to their shared `x` grid (inner join)."""
    common_mask = df_a["x"].isin(df_b["x"])
    df_a_aligned = df_a[common_mask].sort_values("x")
    df_b_aligned = df_b[df_b["x"].isin(df_a["x"])].sort_values("x")
    return df_a_aligned, df_b_aligned


# ------------------------------------------------------------
# Error metric
# ------------------------------------------------------------
def rmse(array_a: np.ndarray, array_b: np.ndarray) -> float:
    """Compute root-mean-square error between two arrays of equal length."""
    return float(np.sqrt(np.mean((array_a - array_b) ** 2)))


# ------------------------------------------------------------
# Curve matching
# ------------------------------------------------------------
def match_train_to_ideal(
    train_df: pd.DataFrame, ideal_df: pd.DataFrame
) -> dict[str, str]:
    """
    For each y-column in *train_df* find the ideal-curve column
    with the lowest RMSE. Return mapping {train_col : ideal_col}.
    """
    matches: dict[str, str] = {}
    train_df, ideal_df = align_grid(train_df, ideal_df)

    train_cols = [c for c in train_df.columns if c != "x"]
    ideal_cols = [c for c in ideal_df.columns if c != "x"]

    for tcol in train_cols:
        best_col = None
        best_err = float("inf")
        for icol in ideal_cols:
            error = rmse(train_df[tcol].to_numpy(), ideal_df[icol].to_numpy())
            if error < best_err:
                best_err, best_col = error, icol
        matches[tcol] = best_col  # type: ignore[arg-type]
    return matches


# ------------------------------------------------------------
# Residual analysis
# ------------------------------------------------------------
def compute_residuals_df(
    train_df: pd.DataFrame, ideal_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Return DataFrame with columns *curve* and *rmse*
    summarising the fit quality of each train curve.
    """
    matches = match_train_to_ideal(train_df, ideal_df)
    rows: list[dict[str, float | str]] = []

    train_df, ideal_df = align_grid(train_df, ideal_df)
    for tcol, icol in matches.items():
        error = rmse(train_df[tcol].to_numpy(), ideal_df[icol].to_numpy())
        rows.append({"curve": tcol, "rmse": error})
    return pd.DataFrame(rows)


def build_residual_table(test_df: pd.DataFrame, ideal_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (x, y) in *test_df* find the ideal-curve value with the
    smallest absolute error. Return a tidy table for interactive plots.
    """
    _, ideal_df = align_grid(test_df, ideal_df)
    ideal_cols = [c for c in ideal_df.columns if c != "x"]

    records: list[dict[str, float | str]] = []
    for _, row in test_df.iterrows():
        x_val, y_val = row["x"], row["y"]
        slice_ideal = ideal_df.loc[ideal_df["x"] == x_val, ideal_cols].iloc[0]
        best_col = (slice_ideal - y_val).abs().idxmin()
        records.append(
            {"x": x_val, "y_test": y_val,
                "y_ideal": slice_ideal[best_col], "curve": best_col}
        )
    return pd.DataFrame(records)
