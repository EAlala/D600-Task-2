import pandas as pd
import numpy as np

#load data set
data_set = pd.read_csv("C:/Users/yeai2_6rsknlh/OneDrive/Visual/D600 Task 2/D600 Task 2 Dataset 1 Housing Information.csv")

#Variables for C2
variables = ['IsLuxury', 'Price', 'SquareFootage', 'SchoolRating', 'RenovationQuality','Garage','CrimeRate', 'LocalAmenities']
analysis_data = data_set[variables]

#Descriptive stats
desc_stats = analysis_data.describe()

#
