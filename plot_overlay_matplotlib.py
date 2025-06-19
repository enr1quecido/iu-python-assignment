"""
Overlay plot: each train curve with its best-matching ideal curve.

Produces `fig_overlay.png` (300 dpi) in the project root.
"""

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import match_train_to_ideal

# ---------- load data ----------
train_df = pd.read_csv(
    "C:/Users/lanta/Desktop/python/assignment/dataset_train.csv")
ideal_df = pd.read_csv(
    "C:/Users/lanta/Desktop/python/assignment/dataset_ideal.csv")

# ---------- find best matches ----------
matches = match_train_to_ideal(train_df, ideal_df)

# ---------- plotting ----------
fig, ax = plt.subplots(figsize=(8, 4.5))

for train_col, ideal_col in matches.items():
    ax.plot(  # solid line: train data
        train_df["x"],
        train_df[train_col],
        label=f"{train_col} (train)",
        linewidth=1.0,
    )
    ax.plot(  # dashed line: ideal data
        ideal_df["x"],
        ideal_df[ideal_col],
        linestyle="--",
        label=f"{ideal_col} (ideal)",
    )

ax.set_title("Train vs. Ideal Curves (Overlay)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(ncol=4, fontsize="small")
fig.tight_layout()
fig.savefig("fig_overlay.png", dpi=300)
plt.show()
