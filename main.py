import pandas as pd
from feature_engineering import full_preprocessing_pipeline
from classification import classification_pipeline
from regression import compare_regression_script_, \
    regression_pipeline, regression_best_model_scores
import pickle
import time

if __name__ == "__main__":
    start = time.time()
    # input_data_df = pd.read_csv(r'Project_InputData.csv')
    # final_regression_dict, final_classification_dict = full_preprocessing_pipeline(input_data_df)
    #classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=True, hyperparameters={},
    #                        method='Extra Trees')
    #classification_pipeline(dataset_type='statistical', test_size=0.2, choose_model=True, hyperparameters={},
    #                        method='Extra Trees')
    #classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=True, hyperparameters={},
    #                        method='Random Forests')
    # classification_pipeline(dataset_type='statistical', test_size=0.2, choose_model=False, hyperparameters={},
    #                         method='MLP NN')
    #classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=True, hyperparameters={},
    #                        method='SVC Linear')
    #classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=False, hyperparameters=
    #                        {'criterion': 'entropy', 'n_estimators': 5000}, method='Extra Trees')

    ###### TASK-2 REGRESSION----------------------------------------------------------------------------------------------------------

    # compare_regression_script_() # Run comparasion for regression performance and plot group barplots
    # ----------- Run grid search on the neural netwoork model and save it as pickle
    # grid_search_results, best_params, trained_reg_model = regression_pipeline(dataset_type='statistical', choose_model=True, hyperparameters={})
    # pickle.dump({'GridSearchResults': grid_search_results, 'BestParameters': best_params,
    #              'TrainedModel':trained_reg_model}
    #             , open('Regression_reults_statistical.pkl', 'wb'))

    # Analyse best models
    scores_reduction = regression_best_model_scores(datatyperead='reduction')
    scores_statistical = regression_best_model_scores(datatyperead='statistical')
    print(scores_reduction)
    print(scores_statistical)
    stop = time.time()
    print('Performance scores of regressor and regressor as classifier')
    print('Time taken to train and grid search is %.2f',(stop-start))
