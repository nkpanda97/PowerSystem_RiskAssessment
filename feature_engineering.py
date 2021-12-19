#!/usr/bin/env python3
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandapower.plotting as ppl
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
                        #print(new_row.at[input_df.columns[jdex]])
                        new_row.at[input_df.columns[jdex]] = np.mean(reduced_df.iloc[:, jdex])
                        #print(new_row.at[input_df.columns[jdex]])
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
    print(f"Dropping columns {dropping_columns}")
    print(new_df.shape)
    return new_df


def add_feature(input):
    """
    

    Parameters
    ----------
    Input : Pandas dataframe
        Povided dataset

    Returns
    -------
    new_data : Tpandas dataframe or np array
        Data with new features combined

    """

    return
    

