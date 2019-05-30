""" This provides the utilities for pre-processing the input data.
author: Younggue Bae
"""

import numpy as np
import pandas as pd


def preprocess(data,
               ordinal_categorical_fields_mapping=None,
               nominal_categorical_fields=None,
               drop_fields=None
               ):
    """
    Pre-processes the input data.
    :param data:
    :param ordinal_categorical_fields_mapping: for example
        ordinal_categorical_fields_mapping = {
            "grade": {"bad": 0, "normal": 1, "good": 2},
            ...
        }
    :param nominal_categorical_fields: for example
        ["gender", "color"]
    :param drop_fields: for example
        ["Id"]
    :return:
    """
    __print_missing_values_summary(data)
    df_data = data

    # Convert categorical values.
    __print_categorical_unique_values(df_data, ordinal_categorical_fields_mapping, nominal_categorical_fields)
    df_data = __convert_categorical_values(df_data, ordinal_categorical_fields_mapping, nominal_categorical_fields)

    # Impute the missing values.
    df_data = __impute_missing_values(df_data)

    # Drop the useless fields(features)
    df_data = __drop_useless_fields(df_data, drop_fields)

    return df_data


def check_input_validation(df):
    """
    Checks the validation whether the data still have the missing values or non numerical values.
    If the validation succeeds, returns true, otherwise throws exception.
    :param df:
    :return:
    """
    print("\nChecking the input data validation.............................")
    invalid_fields = []

    isnull_sum = df.isnull().sum()
    for index, val in isnull_sum.iteritems():
        if val > 0:
            invalid_fields.append(index)
    if len(invalid_fields) > 0:
        raise ValueError("The NaN missing values still exist in fields: " + str(invalid_fields))

    # TODO: Why not working properly??
    isreal_sum = df.applymap(np.isreal).sum()
    for index, val in isreal_sum.iteritems():
        if val < len(df):
            invalid_fields.append(index)
    # if len(invalid_fields) > 0:
    #     raise ValueError("The non-numerical values still exist in fields: " + str(invalid_fields))

    return True


def __drop_useless_fields(df, drop_fields):
    """
    Drops useless fields as a feature selection.
    :param df:
    :return:
    """
    # `Id` is the database row ID of the loan applicant.
    # This value is not very important. Therefore we need to delete this value.
    if drop_fields is not None:
        df_data = df.drop(drop_fields, axis=1)
    return df_data


def __convert_categorical_values(df,
                                 ordinal_categorical_fields_mapping,
                                 nominal_categorical_fields=None
                                 ):
    """
    Converts the categorical values with numerical values or one-hot encoded fields.
    :param df:
    :param ordinal_categorical_fields_mapping:
    :param nominal_categorical_fields:
    :return:
    """

    """
    addr_state_mapping = {
        label: idx for idx, label in
        enumerate(np.unique(df['addr_state']))
    }

    zip_code_mapping = {
        label: idx for idx, label in
        enumerate(np.unique(df['zip_code']))
    }

    purpose_cat_mapping = {
        label: idx for idx, label in
        enumerate(np.unique(df['purpose_cat']))
    }
    """

    # Convert ordinal categorical values to the numerical values
    if ordinal_categorical_fields_mapping is not None:
        df.replace(ordinal_categorical_fields_mapping, inplace=True)

    # df.replace(addr_state_mapping, inplace=True)
    # df.replace(zip_code_mapping, inplace=True)
    # df.replace(purpose_cat_mapping, inplace=True)

    # Convert nominal categorical values to the one-hot encoded fields
    for field_name in nominal_categorical_fields:
        dummies = pd.get_dummies(df[field_name]).rename(columns=lambda x: 'is_' + field_name + '_' + str(x))
        df = pd.concat([df, dummies], axis=1)
        df = df.drop([field_name], axis=1)

    return df


def __impute_missing_values(df):
    """
    TODO: Need to be improved in order to generalize the cases
    Imputes the missing values.
    :param df:
    :return:
    """
    df['mths_since_last_delinq'].fillna(0, inplace=True)
    df['mths_since_last_record'].fillna(0, inplace=True)

    # Drop rows that have not at least 5 non-NnN values.
    df.dropna(thresh=5)

    df.fillna(df.mean(), inplace=True)
    df['emp_length'] = df.emp_length.apply(lambda x: x if str(x).isnumeric() else 0)

    return df


def __print_categorical_unique_values(df, ordinal_categorical_fields_mapping, nominal_categorical_fields):
    categorical_fields = []
    if nominal_categorical_fields is not None:
        categorical_fields = nominal_categorical_fields
    if ordinal_categorical_fields_mapping is not None:
        categorical_fields.extend(ordinal_categorical_fields_mapping.keys())

    print('\nCategorical values:')
    print('------------------------------------')
    for field_name in categorical_fields:
        print('\n' + field_name + ':')
        print(df[field_name].unique())


def __print_missing_values_summary(df):
    """
    Checks the missing values.
    :param df:
    :return:
    """
    print("\nMissing values:")
    print('------------------------------------')
    print(df.isnull().sum())
    print('\n')
    print(len(df))

