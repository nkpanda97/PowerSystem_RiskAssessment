import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.svm import SVR
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neural_network import MLPRegressor
import pandas as pd

def read_dataset(dataset_type='reduction'):
    """

    Parameters
    ----------
    dataset_type = 'reduction' or 'statistical' default = 'reduction'

    Returns
    -------
    X,y = Features. target

    """
    X = np.asarray(pd.read_pickle(f'./datasets/classification_x_{dataset_type}'))
    y = np.asarray(pd.read_pickle(f'./datasets/classification_y_{dataset_type}'))
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


def create_comparasion_models(x_train,y_train):
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
def test_model(models, model_names, xtest, ytest, datatypes):
    """

    Parameters
    ----------
    models: List of all ML models
    model_names: List of all model names
    xtest: Test data
    ytest: Test results

    Returns
    all_results: Lists of dictonar containing Name, MSE, R2, MAPE for all models provided

    """

    all_results = []
    all_mape = []
    all_mse = []
    all_r2 = []
    all_evs = []
    all_err = []
    all_pred = []
    for i, model in enumerate(models):
        prediction = model.predict(xtest)
        mape = mean_absolute_error(ytest, prediction)
        mse = mean_squared_error(ytest, prediction)
        evs = explained_variance_score(ytest, prediction)
        r2 = r2_score(ytest, prediction)

        error_cal = (ytest - prediction) ** 2
        all_pred.append(prediction)

        all_err.append(error_cal)
        dict_res = {'Name': model_names[i],
                    'Predicted_values': prediction,
                    'MAPE': mape,
                    'MSE': mse,
                    'R2': r2,
                    'EVS': evs}
        all_mse.append(mse)
        all_mape.append(mape)
        all_r2.append(r2)
        all_evs.append(evs)
        all_results.append(dict_res)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    x = np.arange(len(models))
    width = 0.4
    ax1.bar(x, all_mse, width)
    ax1.set_ylabel('Mean squaed error')
    ax1.grid()

    ax2.bar(x, all_mape, width)
    ax2.set_ylabel('Mean abs % error')
    ax2.grid()

    ax3.bar(x, all_r2, width)
    ax3.set_ylabel('R2 score')
    ax3.grid()

    ax4.bar(x, all_evs, width)
    ax4.set_ylabel('Explained var. score')
    ax4.grid()
    ax4.set_xticks(x, model_names)

    fig.title('Regression performance comparison for '+datatypes)
    fig.show()



def compare_regression_script_(datatype='reduction'):
    x, y = read_dataset(datatype)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    model_linreg, model_lasso, model_ridge, model_svmlin, model_svmpoly, model_svmgauss, model_tf = create_comparasion_models(x_train,y_train)
    all_models = [model_linreg, model_lasso, model_ridge, model_svmlin, model_svmpoly, model_svmgauss, model_tf]
    all_model_names = ['Lin reg',
                       'Lasso reg',
                       'Ridge reg',
                       'Lin svm',
                       'Poly svm',
                       'Rbf svm',
                       'Neural Net']

    test_model(all_models, all_model_names, x_test, y_test, datatype)



# ------------------------   GRID SEACH FOR NN -----------

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
    return reg, test_score, {'hidden_layer_sizes': hidden_layer_sizes, 'activation': activation, 'solver': solver,
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

    mlp = MLPRegressor(random_state=4720, max_iter=100)
    grid_search = GridSearchCV(mlp, parameters, cv=5,  verbose=10, scoring=['r2', 'neg_mean_squared_error'], refit='r2')
    grid_search.fit(X_train, y_train)
    best_regressor = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_regressor, best_params, grid_search.cv_results_, grid_search.cv_results_['params'], grid_search


def regression_pipeline(dataset_type='reduction', choose_model=True, hyperparameters={}):
    x, y = read_dataset(dataset_type)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    if choose_model:
            best_model, best_params, test_scores, case_parameters, grid_search_results = nn_reg_grid_search(x_train, y_train)
            trained_model, test_score, params = nn_regressor(x_train, x_test, y_train, y_test,hyperparameter_dict=best_params)
    else:
            trained_model, test_score, params = nn_regressor(x_train, x_test, y_train, y_test,hyperparameter_dict=hyperparameters)
    return grid_search_results, best_params, trained_model


