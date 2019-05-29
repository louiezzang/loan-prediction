""" Runs the loan prediction.
author: Younggue Bae
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from prediction_model import PredictionModel
from preprocessing import preprocess, check_input_validation

file_path = './input/DR_Demo_Lending_Club_reduced.csv'
df_raw_data = pd.read_csv(file_path)
df_raw_data.head()


####################################
# Data Preprocessing
####################################
print('\nData preprocessing..................')

# ordinal_categorical_fields_mapping = {
#     "pymnt_plan": {"n": 0, "y": 1},
#     "initial_list_status": {"f": 0, "m": 1},
#     "home_ownership": {"NONE": 1, "OTHER": 2, "MORTGAGE": 3, "RENT": 4, "OWN":  5},
#     "policy_code": {"PC1": 1, "PC2": 2, "PC3": 3, "PC4": 4, "PC5":  5},
#     "verification_status": {"not verified": 0, "VERIFIED - income": 1, "VERIFIED - income source": 1},
# }
ordinal_categorical_fields_mapping = {}
nominal_categorical_fields = [
    "pymnt_plan",
    "initial_list_status",
    "home_ownership",
    "policy_code",
    "verification_status",
    "purpose_cat",
    "addr_state",
    "zip_code",
]

drop_fields = [
    "Id",
    # "mths_since_last_delinq",
    # "mths_since_last_record",
    # "addr_state",
    # "zip_code",
]

df_data = preprocess(data=df_raw_data,
                     ordinal_categorical_fields_mapping=ordinal_categorical_fields_mapping,
                     nominal_categorical_fields=nominal_categorical_fields,
                     drop_fields=drop_fields
                     )

# Check input data validation.
validated = check_input_validation(df_data)
if validated:
    print('Input validation check result: OK')


####################################
# Partitioning a dataset in training and test sets
####################################

X = df_data.drop('is_bad', axis=1)
y = df_data['is_bad'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


####################################
# Training a logistic regression model
####################################

# solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
model = PredictionModel(solver='liblinear',
                        penalty='l1',
                        max_iter=1000,
                        C=1000,
                        debug=False)
model.fit(X_train, y_train)

# Training accuracy
eval_train = model.evaluate(X_train, y_train)
print('\n\nTraining evaluation:', eval_train)


####################################
# Evaluate model
####################################

# Test accuracy
eval_test = model.evaluate(X_test, y_test)
print('\n\nTest evaluation:', eval_test)


####################################
# Predict
####################################

model.predict(X_test[0:50])

model.predict_proba(X_test[0:50])


####################################
# Tune hyperparameters
####################################

# The evaluation of all possible parameter combinations is computationally very expensive.
# Therefore, used the only 200 training data set here.
# best_params = model.tune_parameters(X_train, y_train)
best_params = model.tune_parameters(X_train[:200], y_train[:200])
print('\n\nTuning parameters:', best_params)

