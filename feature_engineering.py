#!/usr/bin/env python3
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import matplotlib.pyplot as plt
import pandapower.plotting as ppl
from sklearn.preprocessing import StandardScaler
# -*- coding: utf-8 -*-
# To install scipy I went at https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy




"""
Created on Sun Dec 12 18:35:16 2021

@author: Demetris-Ch, nkpanda97

This file contains all the functions used for pre-data processing
"""


def plot_feature_properties(Input):
    """
    This function should do te following things:
        1. Print number of features, instances, and number of missing data per features
            A plot can be used to show number of missing features/ feature
        2. Plot the mean, variance and standard deviation of all features
        
        3. Plot the chi2 test score for all features and  
            sort it. Other appropiate tests can also be used.
        
        4. Plot corelation of various features and also show the features which have constant data

    Parameters
    ----------
    Input : Pandas dataframe
        Povided dataset

    Returns
    -------
    None.

    """
    return


def split_input_output(input_df):
    x = input_df.iloc[:, :-1]
    y = input_df.iloc[:, -1]
    return x, y


def create_input_output_for_all(input_dict):
    new_dict = {}
    for key in input_dict:
        input_df = input_dict[key]
        x, y = split_input_output(input_df)
        new_dict[key] = {'X': x, 'Y': y}
    return new_dict


def standardization_preprocessing(input_dict):
    new_dict = {}
    for key in input_dict:
        xy_dict = input_dict[key]
        x = xy_dict['X']
        new_dict[key] = {'X': standardize(x), 'Y': xy_dict['Y']}
    return new_dict


def create_classification_dict(xy_dict):
    classification_dict = {}
    for key in xy_dict:
        x = xy_dict[key]['X']
        y = xy_dict[key]['Y']
        new_y_list = []
        for value in y:
            if value < 0.1:
                y_new = 0
            elif value < 0.35:
                y_new = 1
            elif value < 0.7:
                y_new = 2
            else:
                y_new = 3
            new_y_list.append(y_new)
            new_y_df = pd.DataFrame(new_y_list, columns=['Risk State'])
        classification_dict[key] = {'X': x, 'Y': new_y_df}
    return classification_dict


def standardize(x):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(x), columns=x.columns)


def empty_value_preprocessing(input_df, types=['reduction'], unreduced_df=None):
    """
    Parameters
    ----------
    input_df : Pandas dataframe
        provided dataset without 0 variance features
    types: List
        chosen empty value preprocessing methods
    unreduced_df: None of dataframe
        dataset with 0 variance features, only useful for the domain knowledge method
    Returns
    -------
    new_dataset_dict: dict
        dictionary with new datasets depending on the methods
    """
    boolean_df = input_df.isnull().any(axis=1)
    no_of_empty = boolean_df.sum()

    print(f"Found {no_of_empty} data points with at least 1 empty feature")
    new_dataset_dict = {}
    if 'reduction' in types:
        non_empty_list = []
        for index, row in input_df.iterrows():
            if not boolean_df.loc[index]:
                non_empty_list.append(row)
        reduced_df = pd.DataFrame(non_empty_list, columns=input_df.columns)
        reduced_df = reduced_df.drop([input_df.columns[0]], axis=1)
        new_dataset_dict['reduction'] = reduced_df
    if 'domain' in types:
        net = pn.case9()
        # net buses are L3: bus 6 (load 1)
        #               L2: bus 8? (load 2)
        #               L1: bus 4? (load 0)
        #               G2: bus 2? (gen 1)
        #               G3: bus 1? (gen 0)

        print(net.gen.loc[0]['p_mw'], unreduced_df.loc[0][' P generation-1'])
        # PV Buses
        net.gen.at[0, 'p_mw'] = unreduced_df.loc[0][' P generation-3']
        net.gen.at[0, 'vm_pu'] = unreduced_df.loc[0]['Bus Voltage-3']

        net.gen.at[1, 'p_mw'] = unreduced_df.loc[0][' P generation-2']
        net.gen.at[1, 'vm_pu'] = unreduced_df.loc[0]['Bus Voltage-2']

        # Slack bus
        net.ext_grid.at[0, 'vm_pu'] = unreduced_df.loc[0]['Bus Voltage-1']

        # PQ buses
        print(net.load)
        print(net.load.loc[0])
        net.load.at[0, 'p_mw'] = unreduced_df.loc[0]['P demand-5']
        net.load.at[0, 'q_mvar'] = unreduced_df.loc[0]['Q demand-5']

        net.load.at[1, 'p_mw'] = unreduced_df.loc[0]['P demand-8']
        net.load.at[1, 'q_mvar'] = unreduced_df.loc[0]['Q demand-8']

        net.load.at[2, 'p_mw'] = unreduced_df.loc[0]['P demand-6']
        net.load.at[2, 'q_mvar'] = unreduced_df.loc[0]['Q demand-6']

        #ppl.simple_plot(net, plot_loads=True, plot_sgens=True)
        #ppl.simple_plotly(net)
        empty_list = []
        for index, row in input_df.iterrows():
            if boolean_df.loc[index]:
                empty_list.append(row)
    if 'statistical' in types:
        non_empty_list = []
        for index, row in input_df.iterrows():
            if not boolean_df.loc[index]:
                non_empty_list.append(row)
        reduced_df = pd.DataFrame(non_empty_list, columns=input_df.columns)
        reduced_df = reduced_df.drop([input_df.columns[0]], axis=1)
        input_df = input_df.drop([input_df.columns[0]], axis=1)
        mean_filled_list = []
        for index, row in input_df.iterrows():
            if not boolean_df.loc[index]:
                mean_filled_list.append(row)
            else:
                new_row = row.copy()
                row_bool = row.isnull()
                for jdex in range(0, len(row_bool)):
                    if row_bool[jdex]:
                        new_row.at[input_df.columns[jdex]] = np.mean(reduced_df.iloc[:, jdex])
                mean_filled_list.append(new_row)
        statistical_df = pd.DataFrame(mean_filled_list, columns=input_df.columns)
        new_dataset_dict['statistical'] = statistical_df
    return new_dataset_dict


def remove_features(input_df):
    new_df = input_df.copy()
    variances = input_df.var(axis=0)
    dropping_columns = []
    for i in range(1, len(input_df.columns)):
        # ignore first column because it is the index
        if variances.iloc[i] < 0.000000001:
            dropping_columns.append(input_df.columns[i])
            new_df = new_df.drop([input_df.columns[i]], axis=1)
    print(f"Dropping {len(dropping_columns)} columns {dropping_columns}")
    print(new_df.shape)
    return new_df


def plot_features_variance(input_df):
    variances = input_df.var(axis=0)
    thresholds = [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    y = []
    for threshold in thresholds:
        drop = 0

        for i in range(1, len(input_df.columns)):
            # ignore first column because it is the index
            if variances.iloc[i] < threshold:
                drop = drop + 1
        y.append(drop)
    fig = plt.figure()
    plt.rcParams['axes.axisbelow'] = True
    thresholds_str = [str(thr) for thr in thresholds ]
    plt.bar(thresholds_str, y)
    plt.ylabel('Amount of features Dropped')
    plt.xlabel('Variance Threshold')
    plt.title('Feature Removal per Variance Threshold')
    plt.ylim(top= max(y)+5, bottom= min(y)-5)
    plt.grid()
    fig.savefig('feature_variance.png', dpi=600)
    return


def save_datasets(dataset_dict, task_type='not_given'):
    for key in dataset_dict:
        dataset_dict[key]['X'].to_pickle(f'datasets/{task_type}_x_{key}')
        dataset_dict[key]['Y'].to_pickle(f'datasets/{task_type}_y_{key}')
    return


def add_feature(input_df):
    """

    Parameters
    ----------
    input_df : Pandas dataframe
        Povided dataset

    Returns
    -------
    new_dataframe : pandas dataframe
        Data with new features combined

    """

    return


def feature_addition(input_df, new_column_list, new_column_name):
    return input_df.insert(0, new_column_name, new_column_list)


def full_preprocessing_pipeline(input_data_df):
    # Remove features with 0 variance
    reduced_data_df = remove_features(input_data_df)
    # Plot amount of removed features per variance threshold
    plot_features_variance(input_data_df)
    # Create a dictionary with 2 datasets, 1 with removed datapoints whenever a feature has no value and 1 were
    # all missing values are replaced with the mean of that feature
    data_dict = empty_value_preprocessing(reduced_data_df, types=['reduction', 'statistical'],
                                          unreduced_df=input_data_df)

    # split inputs and outputs per set
    xy_dict = create_input_output_for_all(data_dict)
    # standardize input per dataset
    standardized_xy_dict = standardization_preprocessing(xy_dict)
    # print(standardized_xy_dict)
    final_regression_dict = standardized_xy_dict
    final_classification_dict = create_classification_dict(standardized_xy_dict)
    save_datasets(final_regression_dict, task_type='regression')
    save_datasets(final_classification_dict, task_type='classification')
    return final_regression_dict, final_classification_dict
