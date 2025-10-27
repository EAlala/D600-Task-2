import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

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

# Bivariate: Relationship between EACH independent variable and IsLuxury
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
bivariate_vars = ["Price", "SquareFootage", "SchoolRating", "RenovationQuality", "CrimeRate", "LocalAmenities"]

for i, var in enumerate(bivariate_vars):
    if i < len(axes):
        sns.boxplot(x="IsLuxury", y=var, data=analysis_data_set, ax=axes[i])
        axes[i].set_title(f"{var} vs IsLuxury")

plt.tight_layout()
plt.show()

#Categorical visuals
plt.figure(figsize=(8, 5))
sns.countplot(x="Garage", hue="IsLuxury", data=analysis_data_set)
plt.title("Garage vs IsLuxury")
plt.show()

#Split data
#Convert categorical variables to numerical 
analysis_data_encoded = analysis_data_set.copy()
analysis_data_encoded["Garage"] = analysis_data_encoded["Garage"].map({"Yes": 1, "No": 0})

