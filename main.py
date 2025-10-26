import pandas as pd
import numpy as np

#load data set
data_set = pd.read_csv('C:/Users/yeai2_6rsknlh/OneDrive/Visual/D600 Task 2/D600 Task 2 Dataset 1 Housing Information.csv')

# Descriptive stats
# Define variables for logistic regression
dep_var = "IsLuxury"
indep_vars = ["Price", "SquareFootage", "SchoolRating", "RenovationQuality", "Garage", "CrimeRate", "LocalAmenities"]

