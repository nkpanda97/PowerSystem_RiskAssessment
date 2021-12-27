import pandas as pd
from feature_engineering import full_preprocessing_pipeline


if __name__ == "__main__":
    input_data_df = pd.read_csv(r'Project_InputData.csv')
    final_regression_dict, final_classification_dict = full_preprocessing_pipeline(input_data_df)


