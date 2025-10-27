import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

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

#Define X (features) and Y (target)
X = analysis_data_encoded.drop("IsLuxury", axis=1)
Y = analysis_data_encoded["IsLuxury"]

#Split data: 80% training, 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

#Split display 
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Luxury homes in training: {Y_train.sum()} ({Y_train.mean()*100:.1f}%)")
print(f"Luxury homes in test: {Y_test.sum()} ({Y_test.mean()*100:.1f}%)")

#Save the datasets
X_train.to_csv('training_features.csv', index=False)
Y_train.to_csv('training_target.csv', index=False)
X_test.to_csv('test_features.csv', index=False)
Y_test.to_csv('test_target.csv', index=False)

print("Training and test datasets saved to CSV files\n")

#Logistic Regression Model with Backward Elimination
features = list(X_train.columns)
optimal_features = features.copy()

#Backward elimination
for i in range(len(features)):
    X_temp = X_train[optimal_features]
    X_temp = sm.add_constant(X_temp)
    model = sm.Logit(Y_train, X_temp).fit(disp=False)

    #Find features with highest P-values
    p_values = model.pvalues[1:]
    max_p = p_values.max()

    #Remove P values less than 0.05
    if max_p > 0.05:
        worst_feature = p_values.idxmax()
        optimal_features.remove(worst_feature)
        print(f"Removed {worst_feature} (p-value: {max_p:.4f})")
    else:
        break

#Optimized model
X_optimal = sm.add_constant(X_train[optimal_features])
final_model = sm.Logit(Y_train, X_optimal).fit()

#Display 
print(f"\nFinal optimal features: {optimal_features}")
print(final_model.summary())

#Key metrics
print("\nEXTRACTED MODEL PARAMETERS:")
print(f"AIC: {final_model.aic:.4f}")
print(f"BIC: {final_model.bic:.4f}")
print(f"Pseudo R-squared: {final_model.prsquared:.4f}")

#Coefficients and p-values
coef_df = pd.DataFrame({
    'Variable': final_model.params.index[1:],  # Skip const
    'Coefficient': final_model.params[1:],
    'P-value': final_model.pvalues[1:]
})
print("\nCOEFFICIENTS AND P-VALUES:")
print(coef_df.round(4))