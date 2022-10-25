# Statistic analysis of the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm
import seaborn as sns

# Load the csv data using the Pandas library
filename = 'ENB2012_data.csv'
df = pd.read_csv(filename)
# Rename the attribute
df.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']
nbVal, _ = df.shape
nbAttributes = 8
nbPrediction = 2

# Total number of missing values
print("Number of missing values", df.isnull().sum().sum())

# Basic summary statistics
print(df.describe())

# Plot the histogram and the Kernel density estimation for every attribute

# Plot the histogram, the Kernel density estimation and the normal distribution of the input attribute
def addAttributeToPlot(attributeName, ax, ylimKDE, color, bins=10):
    # Plot the histogram
    ax.hist(df[attributeName], bins=bins)
    # Plot the KDE
    # ax.twinx() instantiate a second axes that shares the same x-axis
    df[attributeName].plot.kde(ax=ax.twinx(), ylim=(0, ylimKDE), c=color, linewidth=3, label="Kernel density estimation").axis('off')
    # Plot the gaussian distribution
    mu, std = norm.fit(df[attributeName]) 
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax2 = ax.twinx()
    ax2.plot(x, p, '--', c='grey', linewidth=2, label="Gaussian distribution")
    ax2.axis('off')
    ax.set_xlabel(attributeName)

fig, axs = plt.subplots(2, 5, figsize=(16, 8))
addAttributeToPlot('relative_compactness', axs[0, 0], 3.5, 'orange')
addAttributeToPlot('surface_area', axs[0, 1], 0.0045, 'green')
addAttributeToPlot('wall_area', axs[0, 2], 0.011, 'red', 8)
addAttributeToPlot('roof_area', axs[0, 3], 0.0175, 'purple')
addAttributeToPlot('overall_height', axs[0, 4], 0.45, 'brown')
addAttributeToPlot('orientation', axs[1, 0], 0.35, 'pink')
addAttributeToPlot('glazing_area', axs[1, 1], 3.65, 'gray')
addAttributeToPlot('glazing_area_distribution', axs[1, 2], 0.21, 'olive')
addAttributeToPlot('heating_load', axs[1, 3], 0.058, 'dodgerblue')
addAttributeToPlot('cooling_load', axs[1, 4], 0.061, 'peru')
plt.figure()


# Boxplot
df.plot(kind='box', subplots=True, layout=(2, 5), figsize=(28, 12), fontsize=14, sharex=False, sharey=False)
plt.figure()


#Discretize the HL
classNames = range(4)
nbClassHL = 4
discrete_HL = []
for i, row in df.iterrows():
    hl = row['heating_load']
    if hl <= 10:
        discrete_HL.append(1)
    elif hl <= 20:
        discrete_HL.append(2)
    elif hl <= 30:
        discrete_HL.append(3)
    else:
        discrete_HL.append(4)
        
#Discretize the CL
nbClassCL = 4
discrete_CL = []
for i, row in df.iterrows():
    cl = row['cooling_load']
    if cl <= 20:
        discrete_CL.append(1)
    elif cl <= 30:
        discrete_CL.append(2)
    elif cl <= 40:
        discrete_CL.append(3)
    else:
        discrete_CL.append(4)

#Pair plot for the heating load
colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
sns.set_palette(sns.color_palette(colors))

hl_df = df.iloc[: , :8]
hl_df['discrete_heating_load'] = discrete_HL
sns.pairplot(hl_df, hue='discrete_heating_load')  
plt.show()

#Pair plot for the cooling load
cl_df = df.iloc[: , :8]
cl_df['discrete_cooling_load'] = discrete_CL
sns.pairplot(cl_df, hue='discrete_cooling_load')  #, palette='vlag'
plt.show()




#Plot different attributes compared to the average HL and CL

def plotEnergyLoadVsAttribute(attributeName, ax, legendLoc = 'lower right'):
    newDf = df.groupby([attributeName]).mean()
    ax.plot(newDf.index, newDf['cooling_load'], label='Avg CL')
    ax.plot(newDf.index, newDf['heating_load'], label='Avg HL')
    ax.legend(loc=legendLoc)
    ax.set_xlabel (' '.join([word for word in attributeName.capitalize().split('_')]))#To get a clean display of the attribute
    ax.set_ylabel ("Energy load")

fig, axs = plt.subplots(2, 4, figsize=(22, 10))
plotEnergyLoadVsAttribute('relative_compactness', axs[0,0])
plotEnergyLoadVsAttribute('surface_area', axs[0,1], 'upper right')
plotEnergyLoadVsAttribute('wall_area', axs[0,2], 'upper left')
plotEnergyLoadVsAttribute('roof_area', axs[0,3], 'upper right')

plotEnergyLoadVsAttribute('overall_height', axs[1, 0])
plotEnergyLoadVsAttribute('orientation', axs[1, 1], 'center right')
plotEnergyLoadVsAttribute('glazing_area', axs[1, 2])
plotEnergyLoadVsAttribute('glazing_area_distribution', axs[1, 3])

plt.figure()
