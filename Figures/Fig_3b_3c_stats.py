# Import necessary libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from scipy.stats import wilcoxon
# ==== CONFIGURATION ====
# Define paths for data and resources
PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
# Load data from CSV files
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))

# ==== PREPROCESSING ====
# Extract the regions and their colors
regions_all = labels.drop_duplicates('region').sort_values('plot_order')['region'].values
regions_MT = ['Mesiotemporal', 'Amygdala', 'Hippocampus']
regions_CX = ['Orbitofrontal', 'Operculo-insular', 'Dorsofrontal', 'Central',
              'Cingular', 'Parietal', 'Insulo-temporal', 'Inferotemporal', 'Occipital']


# ==== DATA FILTERING ====
# Filter out unwanted connections (local, inter-hemispheric, or unknown regions)
con_filtere = con_all[
            (con_all.Hemi != 'B') & (con_all.Group != 'local')].reset_index(drop=True)

# Prepare reverse DataFrame to merge for comparison
con_filtere_reverse = con_filtere[['Subj','Stim', 'Chan', 'Num_trial']].reset_index(drop=True)
con_filtere_reverse = con_filtere_reverse.rename(columns={'Stim':'Chan', 'Chan':'Stim', 'Num_trial':'Num_trial_in'})
con_filtere = con_filtere.merge(con_filtere_reverse, on=['Subj','Stim', 'Chan'], how='left')

# Merge signal data for comparison
con_filtere_reverse = con_filtere[['Subj','Stim', 'Chan', 'Sig']].reset_index(drop=True)
con_filtere_reverse = con_filtere_reverse.rename(columns={'Stim':'Chan', 'Chan':'Stim', 'Sig':'Sig_in'})
con_filtere = con_filtere.merge(con_filtere_reverse, on=['Subj','Stim', 'Chan'], how='left')

# Calculate signal difference
con_filtere['Sig_diff'] = con_filtere['Sig'] - con_filtere['Sig_in']

# Handle NaN values when both signals are zero
con_filtere.loc[(con_filtere.Sig == 0) & (con_filtere.Sig_in == 0), 'Sig_diff'] = np.nan

# Apply trial number threshold to remove unreliable data
num_threshold = 10
con_filtere.loc[(con_filtere.Num_trial_in < num_threshold) | (con_filtere.Num_trial < num_threshold), ['Sig_diff', 'DI']] = np.nan

# ==== DATA SPLITTING ====
# Separate data based on region categories (Mesiotemporal and Cortical regions)
data_DI_Subj_MT = con_filtere[~np.isnan(con_filtere.Sig_diff)&~np.isin(con_filtere.ChanR, regions_MT)].groupby(['Subj', 'StimR'], as_index=False)[['d_inf', 'Sig', 'DI']].mean()
data_DI_Node_MT = con_filtere[~np.isnan(con_filtere.Sig_diff)&~np.isin(con_filtere.ChanR, regions_MT)].groupby(['Subj', 'StimR', 'ChanR'], as_index=False)[['d_inf', 'Sig', 'DI']].mean()

data_DI_Subj_CXT = con_filtere[~np.isnan(con_filtere.Sig_diff)&~np.isin(con_filtere.StimR, regions_MT)&~np.isin(con_filtere.ChanR, regions_MT)].groupby(['Subj', 'StimR'], as_index=False)[['d_inf', 'Sig', 'DI']].mean()
data_DI_Node_CXT = con_filtere[~np.isnan(con_filtere.Sig_diff)&~np.isin(con_filtere.StimR, regions_MT)&~np.isin(con_filtere.ChanR, regions_MT)].groupby(['Subj', 'StimR', 'ChanR'], as_index=False)[['d_inf', 'Sig', 'DI']].mean()

arr_mean = []
df_con = con_filtere[~np.isnan(con_filtere.Sig_diff) & ~np.isin(con_filtere.ChanR, regions_MT)].reset_index(drop=True)
df_con.loc[df_con.Sig == 0, 'Sig'] = np.nan
df_con = df_con.groupby(['Subj', 'StimR', 'ChanR'], as_index=False)[['d_inf', 'Sig', 'DI', 'Sig_diff']].mean()

for region in regions_all:
    df_region = df_con[df_con.StimR == region].reset_index(drop=True)
    md = np.median(df_region.DI)
    mn = np.mean(df_region.DI)
    IQR_low = np.quantile(df_region.DI, 0.25)
    IQR_up = np.quantile(df_region.DI, 0.75)
    md_Sig = np.nanmedian(df_region.Sig)
    IQR_low_Sig = np.nanquantile(df_region.Sig, 0.25)
    IQR_up_Sig = np.nanquantile(df_region.Sig, 0.75)
    n = len(df_region.DI)
    n_pat = len(np.unique(df_region['Subj']))

    # stats
    res = wilcoxon(df_region.Sig_diff, alternative='two-sided')
    arr_mean.append([region, md, mn, f'{md:0.2f} [{IQR_low:0.2f},{IQR_up:0.2f}]',
                     f'{md_Sig:0.2f} [{IQR_low_Sig:0.2f},{IQR_up_Sig:0.2f}]', n, n_pat, res.pvalue])

# Convert the list to a DataFrame
df_region_mean = pd.DataFrame(arr_mean, columns=['Region', 'Median', 'Mean', 'DI Median [IQR]', 'Prob Median[IQR]',
                                                 'N_connection', 'N_pat', 'p_value'])

# FDR correction
rej, pvals_corrected, _, _ = multipletests(df_region_mean['p_value'], method='fdr_bh')
df_region_mean['p_value_corrected'] = pvals_corrected
df_region_mean['rejected'] = rej

PATH_fig  ='/Users/ellenvanmaren/Desktop/Insel/PhD_Projects/EL_experiment/Codes/DATA/Figures'
df_region_mean.to_excel(os.path.join(PATH_fig, 'DI_region_NCX_stats.xlsx'))
# Save the results to a CSV file
#  df_region_mean.to_csv(os.path.join(PATH_fig, 'DI_AreaMean_violin_all.csv'), header=True, index=False)
