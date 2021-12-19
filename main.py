import pandas as pd
import numpy as np
from feature_engineering import empty_value_preprocessing, add_feature, remove_features


if __name__ == "__main__":
    input_data_df = pd.read_csv(r'Project_InputData.csv')
    reduced_data_df = remove_features(input_data_df)
    empty_value_preprocessing(reduced_data_df, types=['reduction', 'statistical'], unreduced_df=input_data_df)
