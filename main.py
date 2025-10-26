from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#load data set
data_set = pd.read_csv("C:/Users/yeai2_6rsknlh/OneDrive/Visual/D600 Task 2/D600 Task 2 Dataset 1 Housing Information.csv")

# Descriptive stats
# Define variables for logistic regression
dep_var = "IsLuxury"
indep_vars = ["Price", "SquareFootage", "SchoolRating", "RenovationQuality", "Garage", "CrimeRate", "LocalAmenities"]

#New dataframe
analysis_data_set = data_set[[dep_var] + indep_vars]

#Descriptive stats
desc_stats = analysis_data_set.describe(include="all").transpose()

#Mode and Range for table
desc_stats["mode"] = analysis_data_set.mode().iloc[0]
desc_stats["range"] = desc_stats["max"] - desc_stats["min"]

#Cleaner presentation
final_stats = desc_stats[["count", "mean", "mode", "std", "min", "max", "range"]]

#Display
print(final_stats.round(2))

print(f"\n{analysis_data_set['IsLuxury'].value_counts()}")
print(f"\n{analysis_data_set['Garage'].value_counts()}")

#Visual style
sns.set_theme(style="whitegrid")

# Univariate: Distribution of ALL variables including IsLuxury
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()
vars_to_plot = ["IsLuxury", "Price", "SquareFootage", "SchoolRating", "RenovationQuality", "CrimeRate", "LocalAmenities"]

for i, var in enumerate(vars_to_plot):
    if i < len(axes):
        analysis_data_set[var].hist(bins=30, ax=axes[i], edgecolor="black")
        axes[i].set_title(f"Distribution of {var}")

for j in range(len(vars_to_plot), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()