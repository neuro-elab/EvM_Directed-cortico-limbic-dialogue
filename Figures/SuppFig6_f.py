# connectogram_plot.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from pathlib import Path
# ==== CONFIGURATION ====
PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
FIGSIZE = (5, 5)
PATH_fig  = '/Figures' # TODO
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


# ==== LOAD DATA ====
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))
ftract_df = pd.read_csv(os.path.join(PATH_Data, 'FTRACT_probability_table.csv'))

regions_all = np.unique(labels.drop_duplicates('region').region)
region_color_map = labels.drop_duplicates('region').set_index('region')['plot_color'].to_dict()
stim_colors = labels['plot_color'].tolist()
all_labels = labels['abbreviation'].tolist()
list_region = labels['region'].tolist()

regions_MT = ['Mesiotemporal', 'Amygdala', 'Hippocampus']
regions_CX = ['Orbitofrontal', 'Operculo-insular', 'Dorsofrontal', 'Central',
              'Cingular', 'Parietal', 'Superotemporal', 'Inferotemporal','Insulo-temporal',
              'Occipital']


# ==== PROCESS CONNECTO DATA ====
df_plot = con_all[
    (con_all.Hemi != 'B') &
    np.isin(con_all.StimR, regions_all) &
    np.isin(con_all.ChanR, regions_all)
].reset_index(drop=True)

#
col_keep = ['Subj', 'StimR', 'ChanR', 'Sig']
df_plot = df_plot[col_keep].reset_index(drop=True)
df_plot['Con'] = 0
df_plot.loc[df_plot.Sig>0, 'Con'] = 1

df_plot = df_plot.groupby(['Subj', 'StimR', 'ChanR'], as_index=False)['Con'].mean().reset_index(drop=True)
print('test')

df_plot['StimGroup'] = df_plot['StimR']
df_plot['ChanGroup'] = 'Limbic'
df_plot.loc[np.isin(df_plot.StimR, regions_CX), 'StimGroup'] = 'Neocortex'
df_plot.loc[np.isin(df_plot.ChanR, regions_CX), 'ChanGroup'] = 'Neocortex'
##
x_order = ['Amygdala', 'Hippocampus', 'Neocortex']
fig, axes = plt.subplots(1, 2, figsize=(6,4), sharey=True)

for ix, receiving in enumerate(['Limbic', 'Neocortex']):
    df_sel = df_plot[df_plot['ChanGroup'] == receiving].copy()
    df_sel = df_sel[df_sel['StimGroup'] != 'Mesiotemporal']
    df_sel = df_sel.groupby(['Subj', 'StimGroup'], as_index=False)['Con'].mean()

    # Boxplot (no fliers, with transparency via patch artist)
    sns.boxplot(
        x='StimGroup', y='Con', data=df_sel, ax=axes[ix],
        order=x_order, showfliers=False
    )

    # Add transparency to box face after drawing
    for patch in axes[ix].artists:
        patch.set_alpha(0.5)

    # Overlay raw datapoints
    sns.stripplot(
        x='StimGroup', y='Con', data=df_sel, ax=axes[ix],
        order=x_order, color='k', dodge=True
    )

    axes[ix].set_title(f"Receiving: {receiving}")
    axes[ix].set_xlabel('Stimulating Region')
    axes[ix].set_ylabel('Connection Density [%]')
    axes[ix].tick_params(axis='x', labelrotation=45)

# Apply y-axis only once
axes[1].set_ylim(0, 1.1)
axes[1].set_yticks([0, 0.5, 1], ['0', '50', '100'])

plt.tight_layout()
plt.savefig(os.path.join(PATH_fig, 'Limbic_Neocortex_density.svg'))
plt.show()
print('test')