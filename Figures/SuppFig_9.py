# ==== IMPORTS ====
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import scikit_posthocs as sp
# ==== CONFIGURATION ====
BASE_PATH = Path(__file__).resolve().parent.parent
PATH_DATA = BASE_PATH / "Data"
#PATH_OUTPUT =

plt.rcParams.update({
    'font.family': 'arial',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 10,
    'svg.fonttype': 'none'
})


# ==== LOAD DATA ====
labels = pd.read_csv(PATH_DATA / "data_atlas.csv")
df_con = pd.read_csv(PATH_DATA / "data_con_figures.csv")

df_ExI = pd.read_csv(PATH_DATA / "ExI_results.csv")
df_ExI = df_ExI[df_ExI.SleepState == "Wake"].reset_index(drop=True)

# ==== GROUP CONNECTIONS ====
regions_MT = ['Mesiotemporal', 'Amygdala', 'Hippocampus']
regions_CX = ['Orbitofrontal', 'Operculo-insular', 'Dorsofrontal', 'Central',
              'Cingular', 'Parietal', 'Superotemporal', 'Inferotemporal',
              'Occipital']
df_ExI["Group_Region"] = "Limbic-Limbic"

df_ExI.loc[
    np.isin(df_ExI.StimR, regions_MT) & np.isin(df_ExI.ChanR, regions_CX),
    "Group_Region"
] = "Limbic-Ncx"

df_ExI.loc[
    np.isin(df_ExI.StimR, regions_CX) & np.isin(df_ExI.ChanR, regions_MT),
    "Group_Region"
] = "Ncx-Limbic"

df_ExI.loc[
    np.isin(df_ExI.StimR, regions_CX) & np.isin(df_ExI.ChanR, regions_CX),
    "Group_Region"
] = "Ncx-Ncx"


### ==== Plot ExI by connection group ====

metric = "AUC"

# ---- Kruskal-Wallis Test ----
groups = [
    group[metric].values
    for _, group in df_ExI.groupby("Group_Region")
]

stat, p_value = kruskal(*groups)

plt.figure(figsize=(6, 4))

# ---- Violin Plot ----
sns.violinplot(
    data=df_ExI,
    x="Group_Region",
    y=metric,
)

# Annotate Kruskal–Wallis result
plt.text(
    x=0.5,
    y=df_ExI[metric].max() * 1.1,
    s=f"Kruskal-Wallis\np = {p_value:.3e}",
    ha="center",
    va="bottom",
    fontsize=10
)


# ---- Dunn's Post Hoc Test ----
posthoc = sp.posthoc_dunn(
    df_ExI, val_col=metric, group_col="Group_Region", p_adjust="bonferroni"
)

x_positions = np.arange(len(posthoc))
significant_pairs = [
    (i, j, posthoc.iloc[i, j])
    for i in range(len(posthoc))
    for j in range(i + 1, len(posthoc))
    if posthoc.iloc[i, j] < 0.05
]


# ---- Annotate Significant Pairwise Comparisons ----
y_max = df_ExI[metric].max() * 1.05

for i, j, p_val in significant_pairs:
    plt.plot([i, j], [y_max, y_max], color="black", lw=1)
    plt.text(
        (i + j) / 2,
        y_max,
        f"p={p_val:.3e}",
        ha="center",
        fontsize=8
    )
    y_max += df_ExI[metric].max() * 0.05  # Step upward


# ---- Labels and Formatting ----
plt.title(f"{metric} by Group Region with Pairwise Comparisons")
plt.xlabel("Group Region")
plt.ylabel(metric)
plt.ylim([0, df_ExI[metric].max() * 1.3])
plt.yticks([0, 0.5, 1])


# ---- Save Figure ----
# figname = os.path.join(PATH_OUTPUT, f"{metric}_wake_distribution.svg")
#plt.savefig(figname, bbox_inches="tight")
plt.show()
print('plotting')