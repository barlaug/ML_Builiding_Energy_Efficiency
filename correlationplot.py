# Statistic analysis of the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the csv, make df and rename cols
filename = 'ENB2012_data.csv'
df = pd.read_csv(filename)
df.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']

# Correlation
sns.set_theme(style="white")
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool)) # For making correlation matrix appear lower triangular
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True) # Custom palette
# Create heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})
#plt.show()
plt.savefig("correlation2.png", bbox_inches="tight", dpi=400)