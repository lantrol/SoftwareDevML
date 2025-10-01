# Plots of False Positive and Negative count per genre and category
# List of image names in order of worst classified

from torchvision import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

test = datasets.ImageFolder(root=f"../data/test")
data_test = pd.DataFrame([[row[0].split("/")[-1], row[1]] for row in test.imgs])
data_test = data_test.rename(columns = {0: "name", 1: "label"})

df_category = pd.read_csv("../data/categories.csv") # Only for non_smoking
df_genres = pd.read_csv("../data/genres.csv")

df_logits = pd.read_csv("../data/test_logits.csv")
df_preds = pd.read_csv("../data/test_preds.csv")

df = pd.merge(data_test, df_genres, on="name", how="inner")

df["logit_0"] = df_logits["prob_0"]
df["logit_1"] = df_logits["prob_1"]
df["pred"] = df_preds["pred"]

# ---- False positives and false negatives by genre ----
df_incorrect_0 = df.loc[df["label"] != df["pred"].astype("int32")]

error_mat = []
for genre in df["genre"].unique():
    values = df_incorrect_0.loc[df_incorrect_0["genre"] == genre].groupby(["label", "pred"]).size().to_numpy()
    error_mat.append(values)
    print()

error_mat = np.array(error_mat).transpose()
print(error_mat)

sb.heatmap(
    error_mat, cmap="Blues", annot=True,
    xticklabels=["Woman", "Man"],
    yticklabels=["False Positive", "False Negative"],
    vmin=0
)
plt.savefig("../reports/figures/wrong_pred_per_genre.png", dpi=300)
plt.show()


# ---- False positives and false negatives by category ----
df_incorrect_1 = df.loc[df["label"] == 0].loc[df["label"] != df["pred"].astype("int32")]
df_incorrect_1 = pd.merge(df_incorrect_1, df_category, on="name", how="inner")

error_mat = []
for cat in df_incorrect_1["category"].unique():
    values = df_incorrect_1.loc[df_incorrect_1["category"] == cat].groupby(["label", "pred"]).size().to_numpy()
    error_mat.append(values)

error_mat = np.array(error_mat).transpose()
print(error_mat)

a = list(df_incorrect_1["category"].unique())

sb.heatmap(
    error_mat, cmap="Blues", annot=True,
    xticklabels=a,
    yticklabels=["False Positive"],
    vmin=0
)
plt.savefig("../reports/figures/wrong_pred_per_category.png", dpi=300)
plt.show()

# ---- Failed Image Names from worst to best ----
worst = np.argsort(np.absolute((df_incorrect_0["logit_1"] - df_incorrect_0["logit_0"]).to_numpy()), axis=None)
print(df_incorrect_0.iloc[worst[::-1]].drop_duplicates())
