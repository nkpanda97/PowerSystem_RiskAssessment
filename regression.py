import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, accuracy_score, \
    f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neural_network import MLPRegressor
import pandas as pd
import pickle


def read_dataset(dataset_type='reduction'):
    """

    Parameters
    ----------
    dataset_type = 'reduction' or 'statistical' default = 'reduction'

    Returns
    -------
    X,y = Features. target

    """
    X = pd.read_pickle(f'./datasets/regression_x_{dataset_type}')
    y = pd.read_pickle(f'./datasets/regression_y_{dataset_type}')
    return X, y


def create_base_nn(input_shape):
    # Neual network parameters
    acti_fn = 'relu'
    loss_fn = keras.losses.MeanSquaredError()
    # Create model
    model_tf = keras.Sequential()

    # Input layer
    model_tf.add(layers.Dense(128, activation=acti_fn, input_dim=input_shape))

    # Hidden layers
    model_tf.add(layers.Dense(256, activation=acti_fn))
    model_tf.add(layers.Dense(256, activation=acti_fn))
    model_tf.add(layers.Dense(256, activation=acti_fn))
    model_tf.add(layers.Dense(256, activation=acti_fn))

    # Output layer
    model_tf.add(layers.Dense(1, activation='linear'))

    # Compile the neural network
    model_tf.compile(optimizer='adam', metrics=['MSE'], loss=loss_fn)
    model_tf.summary()
    return model_tf


def create_comparasion_models(x_train, y_train):
    # Split train and test data
    input_shape = np.shape(x_train)[1]
    # -------- Models without Hyper parameter --------------------------------------------------------
    # ----------------------Non- Neural Network Model ------------------------------------------------------

    # Creat Regression Models
    # Linear regression model
    model_linreg = LinearRegression().fit(x_train, y_train)
    # Lasso regression model
    model_lasso = Lasso(1).fit(x_train, y_train)
    # Ridge regression model
    model_ridge = Ridge(1).fit(x_train, y_train)
    # Linear SVM Model
    model_svmlin = SVR(kernel='linear').fit(x_train, y_train)
    # Polynomial SVM
    model_svmpoly = SVR(kernel='poly').fit(x_train, y_train)
    # Gaussian SVM
    model_svmgauss = SVR(kernel='rbf').fit(x_train, y_train)
    # ---------------------- Neural Network Models ----------------------------------------------------------
    model_tf = create_base_nn(input_shape)
    # Fit the model
    model_tf.fit(x_train, y_train, epochs=50, batch_size=25, validation_split=0.25)
    return model_linreg, model_lasso, model_ridge, model_svmlin, model_svmpoly, model_svmgauss, model_tf


# --------------- Testing the ML Models of Regression ----------------------------------
def test_model(all_models, model_names, xtests, ytests, datasettypename):
    """

    Parameters
    ----------
    models: List of all ML models for both dataset
    model_names: List of all model names
    xtest: Test data
    ytest: Test results

    Returns
    all_results: Lists of dictonar containing Name, MSE, R2, MAPE for all models provided for each dataset

    """
    export_results = []
    for n, models in enumerate(all_models):

        dict_dataset = []
        for i, model in enumerate(models):
            prediction = model.predict(xtests[n])
            mape = mean_absolute_error(ytests[n], prediction)
            mse = mean_squared_error(ytests[n], prediction)
            evs = explained_variance_score(ytests[n], prediction)
            r2 = r2_score(ytests[n], prediction)

            dict_res = {'Name': model_names[i],
                        'Predicted_values': prediction,
                        'MAPE': mape,
                        'MSE': mse,
                        'R2': r2,
                        'EVS': evs}
            dict_dataset.append(dict_res)
        all_results_datasets = {'Dataset': datasettypename[n],
                                'Results': dict_dataset}

        export_results.append(all_results_datasets)

    return export_results


def compare_regression_script_(datatype=['reduction', 'statistical']):
    trained_models = []
    x_tests = []
    y_tests = []
    for l, datatyperead in enumerate(datatype):
        x, y = read_dataset(datatyperead)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
        x_tests.append(x_test)
        y_tests.append(y_test)
        model_linreg, model_lasso, model_ridge, model_svmlin, model_svmpoly, model_svmgauss, model_tf = create_comparasion_models(
            x_train, y_train)
        all_models = [model_lasso, model_ridge, model_svmlin, model_svmpoly, model_svmgauss, model_tf]
        trained_models.append(all_models)
    models_reduction_dataset = trained_models[0]
    models_statistical_dataset = trained_models[1]

    all_model_names = ['Lasso reg',
                       'Ridge reg',
                       'Lin svm',
                       'Poly svm',
                       'Rbf svm',
                       'Neural Net']
    all_models = [models_reduction_dataset, models_statistical_dataset]
    Model_performance_scores = test_model(all_models, all_model_names, x_tests, y_tests, ['Reduction', 'Statistical'])
    all_mse_statistical = []
    all_mape_statistical = []
    all_r2_statistical = []
    all_evs_statistical = []
    all_mse_reduction = []
    all_mape_reduction = []
    all_r2_reduction = []
    all_evs_reduction = []
    for i, value in enumerate(Model_performance_scores[0]['Results']):
        all_mse_reduction.append(value['MSE'])
        all_mape_reduction.append(value['MAPE'])
        all_r2_reduction.append(value['R2'])
        all_evs_reduction.append(value['EVS'])
    for i, value in enumerate(Model_performance_scores[1]['Results']):
        all_mse_statistical.append(value['MSE'])
        all_mape_statistical.append(value['MAPE'])
        all_r2_statistical.append(value['R2'])
        all_evs_statistical.append(value['EVS'])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    x = np.arange(1, 2 * len(all_models[0]), 2)
    width = 0.4
    ax1.bar(x - 0.2, all_mse_reduction, width, label='reduction_dataset')
    ax1.bar(x + 0.2, all_mse_statistical, width, label='statistical_dataset')
    ax1.set_ylabel('Mean squared error')
    ax1.legend()
    ax1.grid()

    ax2.bar(x - 0.2, all_mape_reduction, width, label='reduction_dataset')
    ax2.bar(x + 0.2, all_mape_statistical, width, label='statistical_dataset')
    ax2.set_ylabel('Mean abs % error')
    ax2.legend()
    ax2.grid()

    ax3.bar(x - 0.2, all_r2_reduction, width, label='reduction_dataset')
    ax3.bar(x + 0.2, all_r2_statistical, width, label='statistical_dataset')
    ax3.set_ylabel('R2 score')
    ax3.legend()
    ax3.grid()

    ax4.bar(x - 0.2, all_evs_reduction, width, label='reduction_dataset')
    ax4.bar(x + 0.2, all_evs_statistical, width, label='statistical_dataset')
    ax4.set_ylabel('Explained var. score')
    ax4.grid()
    ax4.legend()
    ax4.set_xticks(x, all_model_names)

    fig.savefig('ComparasionReg_new.png', dpi=600)
    fig.show()
    return trained_models


def nn_regressor(X_train, X_test, y_train, y_test, hyperparameter_dict={}):
    hyperparameter_dict = {}
    hidden_layer_sizes = hyperparameter_dict.get('hidden_layer_sizes', (128, 256, 256, 256, 256))
    activation = hyperparameter_dict.get('activation', 'relu')
    solver = hyperparameter_dict.get('solver', 'adam')
    alpha = hyperparameter_dict.get('alpha', 0.001)
    batch_size = hyperparameter_dict.get('batch_size', 'auto')
    learning_rate = hyperparameter_dict.get('learning_rate', 'invscaling')
    learning_rate_init = hyperparameter_dict.get('learning_rate_init', 0.01)
    tol = hyperparameter_dict.get('tol', 1e-6)
    momentum = hyperparameter_dict.get('momentum', 0.8)

    reg = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                       batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                       tol=tol, momentum=momentum, max_iter=200, random_state=4720)
    reg.fit(X_train, y_train)
    test_score = reg.score(X_test, y_test)
    y_predict = reg.predict(X_test)
    return reg, y_predict, test_score, {'hidden_layer_sizes': hidden_layer_sizes, 'activation': activation,
                                        'solver': solver,
                                        'alpha': alpha, 'batch_size': batch_size, 'learning_rate': learning_rate,
                                        'learning_rate_init': learning_rate_init, 'tol': tol, 'momentum': momentum}


def nn_reg_grid_search(X_train, y_train):
    # parameters = {'hidden_layer_sizes': [(128, 256, 256, 256, 256)], 'activation': ['relu'], 'solver': ['adam'],
    #               'alpha': [0.001],
    #               'batch_size': [25], 'learning_rate': ['adaptive'],
    #               'learning_rate_init': [0.001], 'momentum': [0.9]}
    parameters = {'hidden_layer_sizes': [(128, 256, 256, 256, 256)], 'activation': ['logistic'
        , 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [0.001, 0.01, 0.1],
                  'batch_size': ['auto', 32, 64], 'learning_rate': ['invscaling', 'adaptive'],
                  'learning_rate_init': [0.001, 0.01], 'momentum': [0.9, 0.5]}

    mlp = MLPRegressor(random_state=4720, max_iter=200)
    grid_search = GridSearchCV(mlp, parameters, cv=5, verbose=10, scoring=['r2', 'neg_mean_squared_error'], refit='r2')
    grid_search.fit(X_train, y_train)
    best_regressor = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_regressor, best_params, grid_search.cv_results_, grid_search.cv_results_['params'], grid_search


def regression_pipeline(dataset_type='reduction', choose_model=True, hyperparameters={}):
    x, y = read_dataset(dataset_type)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    if choose_model:
        best_model, best_params, test_scores, case_parameters, grid_search_results = nn_reg_grid_search(x_train,
                                                                                                        y_train)
        trained_model, test_score, params = nn_regressor(x_train, x_test, y_train, y_test,
                                                         hyperparameter_dict=best_params)
    else:
        trained_model, test_score, params = nn_regressor(x_train, x_test, y_train, y_test,
                                                         hyperparameter_dict=hyperparameters)
    return grid_search_results, best_params, trained_model


def regression_best_model_scores(datatyperead='reduction'):
    best_param = pickle.load(open('Regression_reults_' + datatyperead + '_dataset.pkl', 'rb'))['BestParameters']
    x, y = read_dataset(datatyperead)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    best_regressor, prediction, _, _ = nn_regressor(x_train, x_test, y_train, y_test, hyperparameter_dict=best_param)

    mape = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    evs = explained_variance_score(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    # scores as classifiers
    # Calculate true class
    # 0:safe , 1: low risk, 2: moderate risk, 3: high risk
    true_class = []
    predicted_class = []
    for i, obs in enumerate(np.asarray(y_test)):
        if obs < 0.1:
            true_class.append(0)
        elif obs >= 0.1 and obs < 0.35:
            true_class.append(1)
        elif obs >= 0.35 and obs < .7:
            true_class.append(2)
        else:
            true_class.append(3)

    for i, obs in enumerate(np.asarray(prediction)):
        if obs < 0.1:
            predicted_class.append(0)
        elif obs >= 0.1 and obs < 0.35:
            predicted_class.append(1)
        elif obs >= 0.35 and obs < .7:
            predicted_class.append(2)
        else:
            predicted_class.append(3)

    accu = accuracy_score(true_class, predicted_class)
    peci = precision_score(true_class, predicted_class, average='micro')
    recall = recall_score(true_class, predicted_class, average='micro')
    f1 = f1_score(true_class, predicted_class, average='micro')

    return {'Dataset': datatyperead, 'MSE': mse, 'MAPE': mape, 'EVS': evs, 'R2': r2,
            'Accuracy': accu, 'Precision': peci, 'Recall': recall, 'F1': f1}


