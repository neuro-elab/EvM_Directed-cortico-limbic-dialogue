
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from connectogram_helpers import connectogram_region_sleep_mod, reorder_letters
from pathlib import Path
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.colors as mcolors
from matplotlib.legend import Legend
import matplotlib as mpl
# ==== CONFIGURATION ====

PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
PATH_output = '/Users/ellenvanmaren/Desktop/Insel/PhD_Projects/EL_experiment/Codes/DATA/Figures'
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
df_con = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))
df_P = pd.read_csv(os.path.join(PATH_Data, 'sleep_P_stats.csv'))
df_mg = pd.read_csv(os.path.join(PATH_Data, 'sleep_mod_stats.csv'))
# Filter out unwanted subjects and local channels
df_con = df_con[
    (~np.isin(df_con.Subj, ['EL012', 'EL013'])) &
    (df_con.Group != 'local') &
    (df_con.Hemi != 'B')
].reset_index(drop=True)


regions_MT = ['Mesiotemporal', 'Amygdala', 'Hippocampus']
regions_CX = ['Orbitofrontal', 'Operculo-insular', 'Dorsofrontal', 'Central',
              'Cingular', 'Parietal', 'Superotemporal', 'Inferotemporal','Insulo-temporal',
              'Occipital']

# Get color map from table
region_color_map = labels.drop_duplicates('region').set_index('region')['plot_color'].to_dict()
# Filter and compute statistic
regions_all = labels.drop_duplicates('region').sort_values('plot_order')['region'].values
col_analysis  = ['Norm_NREM_Mag', 'Norm_REM_Mag','Diff_NREM_P', 'Diff_REM_P', 'DI', 'Sig', 'peak_latency']

df_plot = df_con[df_con.Sig>0].groupby(['StimR','ChanR'], as_index=False)[col_analysis].mean().reset_index(drop=True)
df_plot['grouped'] ='unknown'
df_plot.loc[(df_plot.StimR == 'Amygdala')&np.isin(df_plot.ChanR,regions_CX),'grouped'] ='Amy-Neo'
df_plot.loc[(df_plot.StimR == 'Hippocampus')&np.isin(df_plot.ChanR,regions_CX),'grouped'] ='Hipp-Neo'
df_plot.loc[(df_plot.ChanR == 'Amygdala')&np.isin(df_plot.StimR,regions_CX),'grouped'] ='Neo-Amy'
df_plot.loc[(df_plot.ChanR == 'Hippocampus')&np.isin(df_plot.StimR,regions_CX),'grouped'] ='Neo-Hipp'
df_plot.loc[np.isin(df_plot.ChanR,regions_CX)&np.isin(df_plot.StimR,regions_CX),'grouped'] ='Inter Neo'
df_plot.loc[np.isin(df_plot.ChanR,regions_MT)&np.isin(df_plot.StimR,regions_MT),'grouped'] ='Inter Limbic'

df_plot = df_plot[df_plot['grouped'] != 'unknown'].reset_index(drop=True)
df_plot['Prob'] = df_plot['Sig']**2*20
## preparation
# Define meaningful order
group_order = [
    'Amy-Neo', 'Hipp-Neo',
    'Neo-Amy', 'Neo-Hipp',
    'Inter Limbic',
    'Inter Neo'
]

# Register colormap only if not already registered
if 'seismic_gray' not in mpl.colormaps:
    _seismic_gray = [
        (0.0, 0.0, 0.3), (0.0, 0.0, 0.9),
        (0.9, 0.9, 0.9), (0.9, 0.0, 0.0),
        (0.5, 0.0, 0.0)
    ]
    seismic_gray = mcolors.LinearSegmentedColormap.from_list('seismic_gray', _seismic_gray, N=50)
    mpl.colormaps.register(seismic_gray)
else:
    seismic_gray = mpl.colormaps['seismic_gray']
# Seaborn sizes range
size_range = (10, 100)

def map_size(val):
    """Map value from original size column to marker area used by sns.scatterplot"""
    return size_range[0] + val * (size_range[1] - size_range[0])


# Create norm for range –0.25 to +0.25
norm = TwoSlopeNorm(vmin=-0.25, vcenter= 0, vmax=0.25)
## plotting
fig, axes = plt.subplots(2, 1, figsize=(6,6), sharey=True)

for ix, ss in enumerate(['NREM', 'REM']):
    # Column names per state
    x = 'grouped'
    y = f'Diff_{ss}_P'
    color = f'Norm_{ss}_Mag'
    size = 'Sig'
    # Apply categorical ordering
    df_plot['grouped'] = pd.Categorical(df_plot['grouped'], categories=group_order, ordered=True)
    # Create bubble / scatter plot
    sc = sns.scatterplot(
        data=df_plot,
        x=x,
        y=y,
        size=size,
        sizes=size_range,
        hue=color,
        palette=seismic_gray,
        hue_norm=norm,  # zero-centered
        ax=axes[ix],
        alpha=0.7,
        edgecolor='k',
        marker='o',
        legend=False
    )

    # Add colorbar based on scatter object
    # Create a ScalarMappable for the colorbar
    sm = mpl.cm.ScalarMappable(cmap=seismic_gray, norm=norm)
    sm.set_array([])  # Dummy data for colorbar

    # Add colorbar
    cbar = plt.colorbar(sm, ax=axes[ix])
    cbar.set_label('Normalized Magnitude')


    axes[ix].set_title(f"{ss} - Connectivity Changes", fontsize=10)
    axes[ix].set_ylabel('Delta Probability [%]')
    axes[ix].set_xlabel('')
    axes[ix].tick_params(axis='x', labelrotation=45)
    axes[ix].set_ylim([-0.25, 0.25])

    # Bubble size legend
    # Legend marker sizes for chosen values
    legend_values = [0.25, 0.5, 0.75, 1]
    legend_sizes = [map_size(v) for v in legend_values]

    # Create handles for size legend
    size_handles = [plt.scatter([], [], s=s, color='gray', alpha=0.7, edgecolor='k') for s in legend_sizes]
    size_labels = [str(v) for v in legend_values]
    legend = Legend(axes[ix], size_handles, size_labels, title='Sig', loc='upper left', bbox_to_anchor=(1.2, 1))
    axes[ix].add_artist(legend)

plt.tight_layout()
plt.savefig(os.path.join(PATH_fig, 'Sleep_modulation_scatter.svg'))
plt.show()
print('test')