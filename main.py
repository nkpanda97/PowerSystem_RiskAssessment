import pandas as pd
from feature_engineering import full_preprocessing_pipeline
from classification import classification_pipeline


if __name__ == "__main__":
    input_data_df = pd.read_csv(r'Project_InputData.csv')
    final_regression_dict, final_classification_dict = full_preprocessing_pipeline(input_data_df)
    classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=True,
                            hyperparameters={'hidden_layer_sizes': (1024, 1024, 256, 256, 128, 128)},
                            method='MLP NN')
    classification_pipeline(dataset_type='statistical', test_size=0.2, choose_model=True,
                            hyperparameters={'hidden_layer_sizes': (1024, 512, 256, 256, 128, 128)},
                            method='MLP NN')