import pandas as pd
from feature_engineering import full_preprocessing_pipeline
from classification import classification_pipeline


if __name__ == "__main__":
    input_data_df = pd.read_csv(r'Project_InputData.csv')
    final_regression_dict, final_classification_dict = full_preprocessing_pipeline(input_data_df)
    #classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=True, hyperparameters={},
    #                        method='Extra Trees')
    #classification_pipeline(dataset_type='statistical', test_size=0.2, choose_model=True, hyperparameters={},
    #                        method='Extra Trees')
    #classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=True, hyperparameters={},
    #                        method='Random Forests')
    classification_pipeline(dataset_type='statistical', test_size=0.2, choose_model=False, hyperparameters={},
                            method='MLP NN')
    #classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=True, hyperparameters={},
    #                        method='SVC Linear')
    #classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=False, hyperparameters=
    #                        {'criterion': 'entropy', 'n_estimators': 5000}, method='Extra Trees')
