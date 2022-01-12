import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow import keras
from tensorflow.keras import layers

# Import data
x_data = './datasets/regression_x_reduction'
y_data = './datasets/regression_y_reduction'
x = pickle.load(open(x_data, 'rb'))
y = pickle.load(open(y_data, 'rb'))

# Split train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
input_shape = np.shape(x_train)[1]
# ----------------------Non- Neural Network Model ------------------------------------------------------
# Feature scaling
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

print("Mean of x =...", np.mean(x))
print("Mean of scaled x =...", np.mean(x_scaled))
print("Mean of scaled x_train = ...", np.mean(x_train))
print("Mean of scaled x_test = ...", np.mean(x_test))

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

# Fit the model
model_tf.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.23)


# --------------- Testing the ML Models of Regression ----------------------------------
def test_model(models, model_names, xtest, ytest):
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
    for i, model in enumerate(models):
        prediction = model.predict(xtest)
        mape = mean_absolute_error(ytest, prediction)
        mse = mean_squared_error(ytest, prediction)
        r2 = r2_score(ytest, prediction)
        dict_res = {'Name': model_names[i], 'Predicted_values': prediction, 'MAPE': mape, 'MSE': mse, 'R2': r2}
        all_mse.append(mse)
        all_mape.append(mape)
        all_r2.append(2)
        all_results.append(dict_res)
    plt.figure(figsize=(10,10))
    x = np.arange(len(models))
    width = 0.4
    # plt.bar(x+0.4,all_mse, width)
    plt.bar(x, all_mape, width)
    plt.bar(x - 0.4, all_r2, width)
    plt.xticks(x,model_names)
    plt.ylabel('Performance scores')
    plt.legend(['MSE','MAPE','R2'])
    plt.grid()
    plt.show()

    return all_results


all_models = [model_linreg, model_lasso, model_ridge, model_svmlin, model_svmpoly, model_svmgauss, model_tf]
all_model_names = ['Linear regression',
                   'Lasso Rigression',
                   'Ridge Regression',
                   'Linear SVM',
                   'Poly SVM',
                   'Gaussian SVM',
                   'Neural Netwoks']

model_test_results = test_model(all_models, all_model_names, x_test, y_test)
