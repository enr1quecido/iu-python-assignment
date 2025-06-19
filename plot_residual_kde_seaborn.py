# codeinhuman 🔴import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from data_utils import compute_residuals_df

train_df = pd.read_csv("datasets/dataset_train.csv")
ideal_df = pd.read_csv("datasets/dataset_ideal.csv")

residuals = compute_residuals_df(train_df, ideal_df)

sns.set_style("darkgrid")
fig = sns.kdeplot(
    data=residuals,
    x="rmse",
    hue="curve",
    fill=True,
    common_norm=False,
    linewidth=1
).figure
fig.suptitle("RMSE Distribution per Train Curve")
fig.savefig("fig_kde.png", dpi=300)
plt.show()
