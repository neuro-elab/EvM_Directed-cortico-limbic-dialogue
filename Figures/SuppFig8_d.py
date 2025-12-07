# ==== IMPORTS ====
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# List of all regions in plot order
regions_all = (
    labels.drop_duplicates("region")
    .sort_values("plot_order")["region"]
    .values
)

# Add unique connection ID per subject–connection pair
df_con["Con_ID"] = df_con.groupby(["Subj", "sc_id", "rc_id"]).ngroup()

# Convert significance to percentage
df_con["Sig"] = df_con["Sig"] * 100

# ==== DEFINE REGION GROUPS ====
regions_MT = ["Mesiotemporal", "Amygdala", "Hippocampus"]
regions_CX = [
    "Orbitofrontal", "Operculo-insular", "Dorsofrontal", "Central",
    "Cingular", "Parietal", "Superotemporal", "Inferotemporal", "Occipital"
]
# =============================================================================
# 1. INTERSUBJECT VARIABILITY ANALYSIS
# =============================================================================

# Keep only unilateral hemisphere stimulations
df = df_con[df_con.Hemi != "B"].reset_index(drop=True)

# Treat 0% significance as missing (direction uninformative)
df.loc[df.Sig == 0, "Sig"] = np.nan

# Average directional metrics per subject per region pair
df_subj = (
    df.groupby(["Subj", "StimR", "ChanR"], as_index=False)[["DI", "Sig", "d_inf"]]
    .mean()
)

# Keep only valid anatomical regions
mask_regions = np.isin(df_subj.StimR, regions_all) & np.isin(df_subj.ChanR, regions_all)
df_subj = df_subj[mask_regions].reset_index(drop=True)

# Remove self-connections (same region)
df_subj = df_subj[df_subj.StimR != df_subj.ChanR].reset_index(drop=True)

# Create reverse-direction dataset to obtain Sig_BA
df_subj_inv = (
    df_subj[["Subj", "StimR", "ChanR", "Sig"]]
    .rename(columns={"StimR": "ChanR", "ChanR": "StimR", "Sig": "Sig_BA"})
)

# Merge A→B and B→A
df_subj = df_subj.merge(df_subj_inv, on=["Subj", "StimR", "ChanR"], how="left")

# Create region pair identifier
df_subj["pair"] = df_subj["StimR"] + "_" + df_subj["ChanR"]

# Identify directional pairs with main connectivity direction
df_subj_main = df_subj.groupby("pair", as_index=False)["DI"].mean()
main_pairs = df_subj_main.loc[df_subj_main.DI > 0, "pair"].unique()

df_subj = df_subj[df_subj["pair"].isin(main_pairs)].reset_index(drop=True)

# Count subjects per pair
pair_counts = df_subj.groupby("pair")["Subj"].nunique()

# Keep only pairs with sufficiently many subjects (≥7)
valid_pairs = pair_counts[pair_counts >= 7].index
df_subj_filtered = df_subj[df_subj["pair"].isin(valid_pairs)].reset_index(drop=True)


# =============================================================================
# 2. PLOTTING
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(6.8, 4), sharex=True)

# --- Plot 1: A→B significance ---
ax = axes[0]
sns.stripplot(
    x="pair", y="Sig", data=df_subj_filtered,
    dodge=False, jitter=True, s=2, ax=ax, legend=False
)
sns.violinplot(
    x="pair", y="Sig", data=df_subj_filtered,
    ax=ax, inner="box", linewidth=0.5,
    color="white", edgecolor="black", density_norm="width"
)
ax.set_ylabel("Sig AB [%]")

# --- Plot 2: B→A significance ---
ax = axes[1]
sns.stripplot(
    x="pair", y="Sig_BA", data=df_subj_filtered,
    dodge=False, jitter=True, s=2, ax=ax, legend=False
)
sns.violinplot(
    x="pair", y="Sig_BA", data=df_subj_filtered,
    ax=ax, inner="box", linewidth=0.5,
    color="white", edgecolor="black", density_norm="width"
)
ax.set_ylabel("Sig BA [%]")

# --- Plot 3: Directionality index ---
ax = axes[2]
sns.stripplot(
    x="pair", y="DI", data=df_subj_filtered,
    dodge=False, jitter=True, s=2, ax=ax, legend=False
)
sns.violinplot(
    x="pair", y="DI", data=df_subj_filtered,
    ax=ax, inner="box", linewidth=0.5,
    color="white", edgecolor="black", density_norm="width"
)
ax.set_ylabel("DI")
ax.set_ylim([-1, 1])
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

print("done")