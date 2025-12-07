
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from connectogram_helpers import connectogram_region, reorder_letters
from pathlib import Path
from matplotlib.colors import LogNorm
import seaborn as sns

plt.rcParams.update({
    'font.family': 'arial',
    'font.size': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'svg.fonttype': 'none',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 10
})

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
# plot all connections
for dist_metric_label, dist_metric in zip(['Inflated', 'Euclidean'],['d', 'd_inf']):
    fig, ax = plt.subplots(figsize=(4,3))

    hb = ax.hexbin(
        df_plot[dist_metric],
        df_plot["Sig"],
        gridsize=40,         # resolution of hexes
        norm=LogNorm(),      # log scale color mapping
        mincnt=1,            # ignore empty bins
    )

    # Labels & layout

    ax.set_ylabel("Response Probability")   # only left subplot needs y label
    ax.set_xlabel(f"{dist_metric_label} Distance [mm]")
    ax.set_title(f"All Connections (N={len(df_plot)})")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count (log scale)")

    plt.tight_layout()
    plt.savefig(os.path.join(PATH_fig, f'AllCon_Prob_distance_{dist_metric}.svg'))
    plt.show()



###3# region specific
region_plot = ['Amygdala', 'Hippocampus']

for dist_metric_label, dist_metric in zip(['Inflated', 'Euclidean'],['d', 'd_inf']):
    fig, axes = plt.subplots(1,len(region_plot), figsize=(6,4), sharey=True)

    for ax, reg in zip(axes, region_plot):
        # Data where current region is StimR
        df_reg = df_plot[df_plot.StimR == reg].copy()

        # Map color of ChanR region
        df_reg["color"] = df_reg["ChanR"].map(region_color_map)

        ax.scatter(
            df_reg[dist_metric],
            df_reg["Sig"],
            s=20,                # marker size
            alpha=0.6,           # transparency to reduce overlap
            edgecolor="k",       # optional outline
            linewidth=0.5,
            c=df_reg["color"]    # region-based coloring
        )

        ax.set_title(f"Stim Region: {reg}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Response Probability")   # only left subplot needs y label
    axes[0].set_xlabel(f"{dist_metric_label} Distance [mm]")
    axes[1].set_xlabel(f"{dist_metric_label} Distance [mm]")
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_fig, f'Limbic_Prob_distance_{dist_metric}.svg'))
    plt.show()
    print('test')
