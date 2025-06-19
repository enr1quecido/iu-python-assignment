# codeinhuman 🔴import pandas as pd
🔴import numpy as np


def align_grid(df_a, df_b):
    """Return df_a and df_b filtered to their common `x` values (inner join on x)."""
    common = df_a["x"].isin(df_b["x"])
    return df_a[common].sort_values("x"), df_b[df_b["x"].isin(df_a["x"])].sort_values("x")


def rmse(a, b):
    """Root-mean-square error between two equal-length numeric arrays."""
    return np.sqrt(np.mean((a - b) ** 2))


def match_train_to_ideal(train_df, ideal_df):
    """
    For each y-column in train_df find the ideal_df column with minimum RMSE.
    Returns dict {train_col: ideal_col}.
    """
    matches = {}
    train_df, ideal_df = align_grid(train_df, ideal_df)
    y_train_cols = [c for c in train_df.columns if c != "x"]
    y_ideal_cols = [c for c in ideal_df.columns if c != "x"]

    for tcol in y_train_cols:
        best_col = None
        best_rmse = float("inf")
        for icol in y_ideal_cols:
            err = rmse(train_df[tcol].values, ideal_df[icol].values)
            if err < best_rmse:
                best_rmse, best_col = err, icol
        matches[tcol] = best_col
    return matches


def compute_residuals_df(train_df, ideal_df):
    """
    Return long-form DataFrame with columns: curve, rmse.
    """
    matches = match_train_to_ideal(train_df, ideal_df)
    rows = []
    train_df, ideal_df = align_grid(train_df, ideal_df)
    for tcol, icol in matches.items():
        r = rmse(train_df[tcol].values, ideal_df[icol].values)
        rows.append({"curve": tcol, "rmse": r})
    return pd.DataFrame(rows)


def build_residual_table(test_df, ideal_df):
    """
    For each test point choose the ideal curve with minimal absolute error.
    Returns DataFrame suitable for interactive exploration.
    """
    _, ideal_df = align_grid(test_df, ideal_df)
    records = []
    y_ideal_cols = [c for c in ideal_df.columns if c != "x"]

    for idx, row in test_df.iterrows():
        x_val, y_val = row["x"], row["y"]
        slice_ideal = ideal_df.loc[ideal_df["x"]
                                   == x_val, y_ideal_cols].iloc[0]
        best_col = (slice_ideal - y_val).abs().idxmin()
        records.append(
            {"x": x_val, "y_test": y_val,
                "y_ideal": slice_ideal[best_col], "curve": best_col}
        )
    return pd.DataFrame(records)
