import pandas as pd
import numpy as np

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

