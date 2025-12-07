# ==== IMPORTS ====
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Add unique connection ID
df_con["Con_ID"] = df_con.groupby(["Subj", "sc_id", "rc_id"]).ngroup()

# Convert significance value to %
df_con["Sig"] = df_con["Sig"] * 100


# ==== DEFINE REGION GROUPS ====
regions_MT = ["Mesiotemporal", "Amygdala", "Hippocampus"]
regions_CX = [
    "Orbitofrontal", "Operculo-insular", "Dorsofrontal", "Central",
    "Cingular", "Parietal", "Superotemporal", "Inferotemporal", "Occipital"
]

# =============================================================================
# 1. DIRECTIONAL RESPONSE PROBABILITIES
# =============================================================================

# Merge A→B and B→A versions of each connection
df = df_con.merge(
    df_con[["sc_id", "rc_id", "Sig"]].rename(
        columns={"sc_id": "rc_id", "rc_id": "sc_id", "Sig": "Sig_BA"}
    ),
    on=["sc_id", "rc_id"],
    how="left"
)

# Keep only unique direction pairs (A < B)
df_AB = df[df["sc_id"] < df["rc_id"]].reset_index(drop=True)

# Assign directional probabilities
df_AB["P_AB"] = df_AB["Sig"]
df_AB["P_BA"] = df_AB["Sig_BA"]

# Swap values if only B→A significant
mask_only_BA = (df_AB["Sig"] == 0) & (df_AB["Sig_BA"] > 0)
df_AB.loc[mask_only_BA, ["P_AB", "P_BA"]] = df_AB.loc[mask_only_BA, ["Sig_BA", "Sig"]].values

# Remove rows with undefined directionality
df_AB = df_AB[~np.isnan(df_AB["DI"])].reset_index(drop=True)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(4, 3))
scatter = ax.scatter(
    x=df_AB["P_AB"], y=df_AB["P_BA"],
    c=df_AB["d_inf"], s=0.5, cmap="cool"
)

ax.set_aspect(1)  # Keep x:y ratio equal

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Inflated Euclidean distance [mm]")

# Adjust colorbar height to match plot
cbar.ax.set_position([0.9, ax.get_position().y0, 0.02, ax.get_position().height])

ax.set_xlabel("Response Probability AB [%]", fontsize=6)
ax.set_ylabel("Response Probability BA [%]", fontsize=6)
ax.set_title(f"N={len(df_AB)}")

plt.tight_layout()
plt.show()


# =============================================================================
# 2. DIRECTIONALITY BY DISTANCE
# =============================================================================

# Reuse merged df to avoid recomputation
df2 = df.copy()

# Keep only unique pairs (A < B) and valid DI
df_AB2 = df2[df2["sc_id"] < df2["rc_id"]].reset_index(drop=True)
df_AB2 = df_AB2[~np.isnan(df_AB2["DI"])].reset_index(drop=True)

# Use absolute direction index
df_AB2["DI"] = abs(df_AB2["DI"])

fig, ax = plt.subplots(figsize=(4, 3))
hb = ax.hexbin(
    x=df_AB2["d_inf"], y=df_AB2["DI"],
    gridsize=60, cmap="binary", bins="log"
)

ax.set_xlabel("Inflated Euclidean distance [mm]", fontsize=6)
ax.set_ylabel("Response Probability [%]", fontsize=6)
ax.set_title(f"N={len(df_AB2)}", fontsize=8)

cbar = fig.colorbar(hb, ax=ax)
cbar.set_label("log10(counts)")
cbar.ax.set_position([0.9, ax.get_position().y0, 0.02, ax.get_position().height])

plt.tight_layout()
plt.show()


# =============================================================================
# 3. PROBABILITY BY DISTANCE (ALL SIGNIFICANT CONNECTIONS)
# =============================================================================

df3 = df_con[df_con["Sig"] > 0].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(4, 3))
hb = ax.hexbin(
    x=df3["d_inf"], y=df3["Sig"],
    gridsize=60, cmap="binary", bins="log"
)

ax.set_xlabel("Inflated Euclidean distance [mm]", fontsize=6)
ax.set_ylabel("Response Probability [%]", fontsize=6)
ax.set_title(f"N={len(df3)}", fontsize=8)

cbar = fig.colorbar(hb, ax=ax)
cbar.set_label("log10(counts)")
cbar.ax.set_position([0.9, ax.get_position().y0, 0.02, ax.get_position().height])

plt.tight_layout()
plt.show()

print("Plots completed successfully.")


# =============================================================================
# 4. CORRELATIONS
# =============================================================================
# Select long-range significant connections only
df4 = df_con[(df_con["Sig"] > 0) & (df_con.Group != "local")].reset_index(drop=True)
df4 = df4[['Subj', 'sc_id', 'rc_id', 'Sig', 'DI', 'peak_latency', 'd_inf', 'LL']]

# Create inverse direction latencies (rc→sc)
df4_inv = (
    df4[['Subj', 'sc_id', 'rc_id', 'peak_latency']]
    .rename(columns={'peak_latency': 'pl_BA', 'sc_id': 'rc_id', 'rc_id': 'sc_id'})
    .reset_index(drop=True)
)

df4 = df4.merge(df4_inv, on=['Subj', 'sc_id', 'rc_id'], how='left')

# Compute peak-latency difference between directions
df4['latency_diff'] = df4['peak_latency'] - df4['pl_BA']

# Correlation metric specifications
metrics = [
    ('Sig', 'LL',            'Response probability [%]', 'Response magnitude LL [uV/ms]'),
    ('peak_latency', 'LL',   'Peak latency [ms]',        'Response magnitude LL [uV/ms]'),
    ('Sig', 'peak_latency',  'Response probability [%]', 'Peak latency [ms]'),
    ('DI', 'latency_diff',   'Directionality index',     'Peak latency difference [Δms]')
]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(6, 5))
axes = axes.flatten()

for ax, (x_col, y_col, x_label, y_label) in zip(axes, metrics):

    # Filter valid rows for columns under analysis
    df_filtered = df4[[x_col, y_col]].dropna()

    # Hexbin plot
    hb = ax.hexbin(
        df_filtered[x_col],
        df_filtered[y_col],
        gridsize=50,
        cmap='binary',
        bins='log'
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(counts)')

    # Compute linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_filtered[x_col], df_filtered[y_col]
    )

    # Plot regression line
    x_fit = np.linspace(df_filtered[x_col].min(), df_filtered[x_col].max(), 100)
    ax.plot(x_fit, intercept + slope * x_fit, 'r')

    # Annotate with fit statistics
    text = (
        f'Intercept: {intercept:.2f}\n'
        f'Slope: {slope:.2f}\n'
        f'R²: {r_value**2:.2f}\n'
        f'p-value: {p_value:.2e}'
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=6,
            verticalalignment='top')

    # Labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

plt.suptitle(
    f'Connection metric correlations across long-range connections, '
    f'N={len(df_filtered)}',
    fontsize=9
)

plt.tight_layout()
plt.show()
print("Correlation plots completed.")