
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from connectogram_helpers import connectogram_region, reorder_letters
from pathlib import Path
import seaborn as sns
# ==== CONFIGURATION ====
PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
PATH_fig  = '/Figures' # TODO
# ==== LOAD DATA ====
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))

# Filter connections and prepare the data
con_filtere = con_all[(con_all.Hemi != 'B') & (con_all.Group != 'local')].reset_index(drop=True)
con_filtere_reverse = con_filtere[['sc_id', 'rc_id', 'Num_trial']].reset_index(drop=True)
con_filtere_reverse = con_filtere_reverse.rename(columns={'sc_id': 'rc_id', 'rc_id': 'sc_id', 'Num_trial': 'Num_trial_in'})
con_filtere = con_filtere.merge(con_filtere_reverse, on=['sc_id', 'rc_id'], how='left')

con_filtere_reverse = con_filtere[['sc_id', 'rc_id', 'Sig']].reset_index(drop=True)
con_filtere_reverse = con_filtere_reverse.rename(columns={'sc_id': 'rc_id', 'rc_id': 'sc_id', 'Sig': 'Sig_in'})
con_filtere = con_filtere.merge(con_filtere_reverse, on=['sc_id', 'rc_id'], how='left')
con_filtere['Sig_diff'] = con_filtere['Sig'] - con_filtere['Sig_in']
con_filtere.loc[(con_filtere.Sig == 0) & (con_filtere.Sig_in == 0), ['Sig_diff', 'peak_latency']] = np.nan

# Threshold for filtering
num_threshold = 10
con_filtere.loc[(con_filtere.Num_trial_in < num_threshold) | (con_filtere.Num_trial < num_threshold), ['Sig_diff', 'DI']] = np.nan

# Get color map from table
region_color_map = labels.drop_duplicates('region').set_index('region')['plot_color'].to_dict()

# Filter and compute statistics
regions_all = np.unique(labels.drop_duplicates('region').region)
# only true connections
df_plot = con_filtere[con_filtere.Sig>0].reset_index(drop=True)
region_plot = ['Amygdala', 'Hippocampus']

fig, axes = plt.subplots(1,len(region_plot), figsize=(12, 6), sharey=True)

for ax, reg in zip(axes, region_plot):
    # Data where current region is StimR
    df_reg = df_plot[df_plot.StimR == reg].copy()


    # Compute mean and std per region
    stats = df_reg.groupby("ChanR")["peak_latency"].agg(
        mean="mean",
        std="std"
    )

    # Sort regions by mean latency
    stats = stats.sort_values("mean")

    # Map colors
    stats["color"] = stats.index.map(region_color_map)

    # Bar plot (mean latency)
    ax.bar(stats.index, stats["mean"], color=stats["color"])
    ax.axhline(0.065, color="k", linestyle="--")
    # Bar plot with std as error bars
    ax.bar(
        stats.index,
        stats["mean"],
        yerr=stats["std"],
        capsize=4,
        color=stats["color"]
    )


    # Formatting
    ax.set_title(f"Peak Latency from {reg} → Other Regions")
    ax.set_ylabel("Peak Latency")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(PATH_fig, 'PeakLatency_Regions.svg'))
plt.show()
print('test')