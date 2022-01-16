from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support


def read_dataset(dataset_type='reduction'):
    X = pd.read_pickle(f'datasets/classification_x_{dataset_type}')
    y = pd.read_pickle(f'datasets/classification_y_{dataset_type}')
    return X, y


def create_training_and_test_sets(X, Y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=4720)
    return X_train, X_test, y_train, y_test


def extra_trees_classification(X_train, X_test, y_train, y_test, hyperparameter_dict={}):
    criterion = hyperparameter_dict.get('criterion', 'gini')
    n_estimators = hyperparameter_dict.get('n_estimators', 100)
    clf = ExtraTreesClassifier(criterion=criterion, n_estimators=n_estimators, random_state=4720)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f'Training precision recall fscore results {precision_recall_fscore_support(y_train, clf.predict(X_train))}')
    print(f'Test precision recall fscore results {precision_recall_fscore_support(y_test, clf.predict(X_test))}')
    return clf, test_score, {'criterion': criterion, 'n_estimators': n_estimators}


def sv_classification(X_train, X_test, y_train, y_test, hyperparameter_dict={}, type='rbf'):
    C = hyperparameter_dict.get('C', 1.0)
    degree = hyperparameter_dict.get('degree', 3)
    gamma = hyperparameter_dict.get('gamma', 'scale')
    clf = SVC(C=C, degree=degree, gamma=gamma, random_state=4720, kernel=type)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f'Training precision recall fscore results {precision_recall_fscore_support(y_train, clf.predict(X_train))}')
    print(f'Test precision recall fscore results {precision_recall_fscore_support(y_test, clf.predict(X_test))}')
    return clf, test_score, {'C': C, 'degree': degree, 'gamma': gamma}


def random_forests_classification(X_train, X_test, y_train, y_test, hyperparameter_dict={}):
    criterion = hyperparameter_dict.get('criterion', 'gini')
    n_estimators = hyperparameter_dict.get('n_estimators', 100)
    clf = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, random_state=4720)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f'Training precision recall fscore results {precision_recall_fscore_support(y_train, clf.predict(X_train))}')
    print(f'Test precision recall fscore results {precision_recall_fscore_support(y_test, clf.predict(X_test))}')
    return clf, test_score, {'criterion': criterion, 'n_estimators': n_estimators}


def nn_mlp_classification(X_train, X_test, y_train, y_test, hyperparameter_dict={}):
    hyperparameter_dict = {}
    hidden_layer_sizes = hyperparameter_dict.get('hidden_layer_sizes', (1024, 1024, 256, 256, 128, 128))
    activation = hyperparameter_dict.get('activation', 'relu')
    solver = hyperparameter_dict.get('solver', 'adam')
    alpha = hyperparameter_dict.get('alpha', 0.001)
    batch_size = hyperparameter_dict.get('batch_size', 'auto')
    learning_rate = hyperparameter_dict.get('learning_rate', 'invscaling')
    learning_rate_init = hyperparameter_dict.get('learning_rate_init', 0.01)
    tol = hyperparameter_dict.get('tol', 1e-6)
    momentum = hyperparameter_dict.get('momentum', 0.8)

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                        batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                        tol=tol, momentum=momentum, max_iter=200, random_state=4720)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f'Training precision recall fscore results {precision_recall_fscore_support(y_train, clf.predict(X_train))}')
    print(f'Test precision recall fscore results {precision_recall_fscore_support(y_test, clf.predict(X_test))}')
    return clf, test_score, {'hidden_layer_sizes': hidden_layer_sizes, 'activation': activation, 'solver': solver,
                             'alpha': alpha, 'batch_size': batch_size, 'learning_rate': learning_rate,
                             'learning_rate_init': learning_rate_init, 'tol': tol, 'momentum': momentum}


def extra_trees_grid_search(X_train, y_train):
    parameters = {'n_estimators': [1000, 5000], 'criterion': ('gini', 'entropy')}
    etc = ExtraTreesClassifier(random_state=4720)
    grid_search = GridSearchCV(etc, parameters, cv=10)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return clf, best_params, grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']


def svc_grid_search(X_train, y_train, type='rbf'):
    parameters = {'C': [0.1, 0.5, 1], 'degree': [3], 'gamma': ('scale', 'auto')}
    svc_class = SVC(random_state=4720, kernel=type)
    grid_search = GridSearchCV(svc_class, parameters, cv=10)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return clf, best_params, grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']


def nn_mlp_grid_search(X_train, y_train):
    parameters = {'hidden_layer_sizes': [(1024, 1024, 256, 256, 128, 128)], 'activation': ['relu'], 'solver': ['adam'],
                  'alpha': [0.001], 'batch_size': ['auto'], 'learning_rate': ['invscaling'],
                  'learning_rate_init': [0.01]}
    mlp = MLPClassifier(random_state=4720, max_iter=20)
    grid_search = GridSearchCV(mlp, parameters, cv=10)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return clf, best_params, grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']


def random_forests_grid_search(X_train, y_train):
    parameters = {'n_estimators': [1000], 'criterion': ('gini', 'entropy')}
    rfc = RandomForestClassifier(random_state=4720)
    grid_search = GridSearchCV(rfc, parameters, cv=10)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return clf, best_params, grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']


def classification_pipeline(dataset_type='reduction', test_size=0.2, choose_model=True, hyperparameters={}, method='Extra Trees'):
    X, y = read_dataset(dataset_type)
    X_train, X_test, y_train, y_test = create_training_and_test_sets(X, y, test_size)
    print(f"X train shape is {X_train.shape}")
    if choose_model:
        if method == 'Extra Trees':
            best_model, best_params, test_scores, case_parameters = extra_trees_grid_search(X_train, y_train)
            trained_model, test_score, params = extra_trees_classification(X_train, X_test, y_train, y_test,
                                                                           hyperparameter_dict=best_params)
        elif method == 'Random Forests':
            best_model, best_params, test_scores, case_parameters = random_forests_grid_search(X_train, y_train)
            trained_model, test_score, params = random_forests_classification(X_train, X_test, y_train, y_test,
                                                                              hyperparameter_dict=best_params)
        elif method == 'MLP NN':
            best_model, best_params, test_scores, case_parameters = nn_mlp_grid_search(X_train, y_train)
            trained_model, test_score, params = nn_mlp_classification(X_train, X_test, y_train, y_test,
                                                                      hyperparameter_dict=best_params)
        elif method == 'SVC RBF':
            best_model, best_params, test_scores, case_parameters = svc_grid_search(X_train, y_train, type='rbf')
            trained_model, test_score, params = sv_classification(X_train, X_test, y_train, y_test,
                                                                  hyperparameter_dict=best_params, type='rbf')
        elif method == 'SVC Poly':
            best_model, best_params, test_scores, case_parameters = svc_grid_search(X_train, y_train, type='poly')
            trained_model, test_score, params = sv_classification(X_train, X_test, y_train, y_test,
                                                                  hyperparameter_dict=best_params, type='poly')

        elif method == 'SVC Linear':
            best_model, best_params, test_scores, case_parameters = svc_grid_search(X_train, y_train, type='linear')
            trained_model, test_score, params = sv_classification(X_train, X_test, y_train, y_test,
                                                                  hyperparameter_dict=best_params, type='linear')

    else:
        if method == 'Extra Trees':
            trained_model, test_score, params = extra_trees_classification(X_train, X_test, y_train, y_test,
                                                                           hyperparameter_dict=hyperparameters)
        elif method == 'Random Forests':
            trained_model, test_score, params = random_forests_classification(X_train, X_test, y_train, y_test,
                                                                              hyperparameter_dict=hyperparameters)
        elif method == 'MLP NN':
            trained_model, test_score, params = nn_mlp_classification(X_train, X_test, y_train, y_test,
                                                                      hyperparameter_dict=hyperparameters)
    print(f"For dataset type {dataset_type}, the model type {method}, and parameters {params}, the test score is {test_score}")
    if choose_model:
        print(f"The rest of the info is test: {test_scores}, parameters {case_parameters}")
    return trained_model


