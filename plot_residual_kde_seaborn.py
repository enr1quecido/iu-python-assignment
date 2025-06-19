"""
Kernel-density plot of RMSE values for all train curves.

Outputs `fig_kde.png` (300 dpi).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_utils import compute_residuals_df

# ---------- load ----------
train_df = pd.read_csv(
    "C:/Users/lanta/Desktop/python/assignment/dataset_train.csv")
ideal_df = pd.read_csv(
    "C:/Users/lanta/Desktop/python/assignment/dataset_ideal.csv")

# ---------- residuals ----------
residuals = compute_residuals_df(train_df, ideal_df)

# ---------- bar plot ----------
sns.set_style("whitegrid")
ax = sns.barplot(
    data=residuals,
    x="curve",
    y="rmse",
    palette="pastel",
)
ax.set_title("RMSE per Train Curve")
ax.set_xlabel("train curve")
ax.set_ylabel("root mean square error")
fig = ax.get_figure()
fig.tight_layout()
fig.savefig("fig_kde.png", dpi=300)   # keep same filename for consistency
plt.show()
