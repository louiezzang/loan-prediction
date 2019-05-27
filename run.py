import sys
import os
import numpy as np
import pandas as pd

from prediction_model import LoanPredictionModel

model = PredictionModel(debug=True)

df_data = pd.read_csv('./input/DR_Demo_Lending_Club_reduced.csv')

X = df_data.drop('is_bad', axis=1)
y = df_data['is_bad'].values

model.fit(X, y)
