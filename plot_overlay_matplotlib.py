# codeinhuman 🔴import pandas as pd
from data_utils import match_train_to_ideal
🔴import matplotlib.pyplot as plt

train_df = pd.read_csv("datasets/dataset_train.csv")
ideal_df = pd.read_csv("datasets/dataset_ideal.csv")

matches = match_train_to_ideal(train_df, ideal_df)

fig, ax = plt.subplots(figsize=(8, 4.5))
for tcol, icol in matches.items():
    ax.plot(train_df["x"], train_df[tcol],
            label=f"{tcol} (train)", linewidth=1)
    ax.plot(ideal_df["x"], ideal_df[icol],
            linestyle="--", label=f"{icol} (ideal)")

ax.set_title("Train vs. Ideal Curves (Overlay)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(ncol=4, fontsize="small")
fig.tight_layout()
fig.savefig("fig_overlay.png", dpi=300)
plt.show()
