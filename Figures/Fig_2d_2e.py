import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import scipy.stats as stats
from pathlib import Path
import itertools
# ==== CONFIGURATION ====
# Define the path to the Connecto software directory
PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
PATH_output ='/Users/ellenvanmaren/Desktop/Insel/PhD_Projects/EL_experiment/Codes/DATA/Figures'
# ==== LOAD DATA ====
# Load the labels and connection data from CSV files
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))

# Define group labels for the plot
group_label = ['local', 'direct', 'indirect']

# Filter the data for plotting: Remove NaN values, exclude 'B' hemisphere, and keep specific groups
df_plot = con_all[~np.isnan(con_all.ExI) & (con_all.Hemi != 'B') & np.isin(con_all.Group, group_label)].reset_index(drop=True)

# ==== CREATE PLOT ====
# Set up the figure and axis for the violin plot
fig, ax = plt.subplots(figsize=(4 / 2.54, 5 / 2.54))

# Create the violin plot
sns.violinplot(
    x='Group', y='ExI', data=df_plot, inner="quart", ax=ax,
    order=group_label, density_norm='width', cut=0, linecolor='black', linewidth=0.75
)

# ==== ANNOTATE GROUP COUNTS ====
# Calculate and annotate N (sample size) for each group
group_counts = df_plot['Group'].value_counts().reindex(group_label)

# Set x-ticks to show group counts as labels
ax.set_xticks(range(len(group_label)))
ax.set_xticklabels(group_counts.values, fontsize=6)

# Add group labels as x-ticks on top of the plot
ax.set_xticks(range(len(group_label)))
ax.set_xticklabels(group_label, fontsize=6)
ax.xaxis.set_tick_params(pad=5)  # Adjust padding for clarity

# ==== SET AXIS LABELS AND TICKS ====
# Set the y-axis label and limits
ax.set_xlabel("")  # Remove x-axis label
ax.set_ylabel("Excitability Index ExI", fontsize=8)
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([0, 0.5, 1], fontsize=6)
ax.set_ylim([-0.01, 1.01])

# Tighten layout to ensure proper spacing
plt.tight_layout()
# Show the plot
plt.show()
print('Fig 2d')
## STATISTICS
# ---- PAIRWISE WILCOXON (MANN–WHITNEY U) ----
results = []
pairs = list(itertools.combinations(group_label, 2))

for g1, g2 in pairs:
    data1 = df_plot[df_plot["Group"] == g1]["ExI"].dropna()
    data2 = df_plot[df_plot["Group"] == g2]["ExI"].dropna()

    stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')

    results.append({
        "Group1": g1,
        "Group2": g2,
        "U_statistic": stat,
        "p_value": p
    })

df_results = pd.DataFrame(results)

# ---- SUMMARY TABLE ----
summary_rows = []

for g in group_label:
    data = df_plot.loc[df_plot["Group"] == g, "ExI"].dropna()

    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    N = len(data)

    summary_rows.append({
        "Group": g,
        "Median [IQR]": f"{median:.2f} [{q1:.2f}, {q3:.2f}]",
        "N": N
    })

summary_table = pd.DataFrame(summary_rows)

# ---- WRITE BOTH SHEETS TO ONE EXCEL FILE ----
output_file = os.path.join(PATH_output, 'Fig_2d.xlsx')

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='stats', index=False)
    summary_table.to_excel(writer, sheet_name='summary', index=False)

print("Saved:", output_file)


# ==== FILTER DATA FOR SCATTER PLOT ====
# Filter the data for the scatter plot: Remove NaN values and ensure valid ranges for 'Sig' and 'AUC'
df_plot_scatter = con_all[~np.isnan(con_all.ExI) & (con_all.Sig > 0) & (con_all.Hemi != 'B') & np.isin(con_all.Group, ['direct', 'indirect', 'local'])].reset_index(drop=True)

# ==== CREATE SCATTER PLOT WITH LINEAR REGRESSION ====
# Set up the figure and axis for the scatter plot
fig, ax = plt.subplots(figsize=(3.4 / 2.54, 5.4 / 2.54))

# Scatter plot
x = df_plot_scatter['Sig']
y = df_plot_scatter['ExI']
sns.scatterplot(x=x, y=y, data=df_plot_scatter, ax=ax, s=1, color='darkgray')

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Plot the regression line
x_vals = np.linspace(np.min(x), np.max(x), 100)
y_vals = slope * x_vals + intercept
ax.plot(x_vals, y_vals, color='red', linewidth=1, label='Fit')

# Compute Pearson correlation coefficient (rho) and p-value
rho, pearson_p_value = pearsonr(x, y)

# Annotate with R², slope, and p-value
r_squared = r_value ** 2
ax.text(0.05, 0.95, f'$R^2$: {r_squared:.3f}\nSlope: {slope:.3f}\nP-value: {p_value:.3g}',
        ha='left', va='top', transform=ax.transAxes, fontsize=6, bbox=dict(facecolor='white', alpha=0.5))

# ==== SET AXIS LABELS AND TICKS FOR SCATTER PLOT ====
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.01, 1.01])
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([0, 0.5, 1], fontsize=6)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels([0, 50, 100], fontsize=6)
ax.set_ylabel("Excitability Index ExI", fontsize=8)
ax.set_xlabel("Response probability [%]", fontsize=8)
ax.legend(fontsize=6, loc='lower right')
plt.tight_layout()
plt.show()
print('Fig 2e')