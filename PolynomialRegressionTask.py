import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import os

data = pd.read_csv("Data/Task1 - dataset - pol_regression.csv")
data_x = data['x'].values
data_y = data['y'].values

train_x, test_x, train_y, test_y = ms.train_test_split(data_x, data_y, test_size=0.3 )

new_order = np.lexsort([train_y, train_x])
train_x = train_x[new_order]
train_y = train_y[new_order]

new_order = np.lexsort([test_y, test_x])
test_x = test_x[new_order]
test_y = test_y[new_order]

degrees_collection = [0, 1, 2, 3, 6, 10]
colours = ['r', 'g', 'm', 'c', 'y', 'k']


def expand_features(features_train, degree):
    X = np.ones(features_train.shape)
    for i in range(1, degree + 1):
        X = np.column_stack((X, features_train ** i))
    return X

def pol_regression(features_train, y_train, degree):
    features_expanded = expand_features(features_train, degree)
    t_values_expanded = features_expanded.transpose().dot(features_expanded)

    if degree == 0:
        return t_values_expanded

    parameters = np.linalg.solve(t_values_expanded, features_expanded.transpose().dot(y_train))
    return parameters

def eval_pol_regression(parameters, x, y, degree):
    expanded_x = expand_features(x, degree)
    regression_values = expanded_x.dot(parameters)

    difference_total = 0

    for index, value in enumerate(y):
        difference_total += abs(regression_values[index] -  value)

    mean_squared_error = (difference_total / len(y))**2

    rmse = np.sqrt(mean_squared_error)
    return rmse


def get_least_squares_solution():
    least_squares = -1
    least_squares_degree = -1

    for degree in degrees_collection:
        errors = np.zeros(train_x.shape)
        parameters = pol_regression(train_x, train_y, degree)
        rmse = eval_pol_regression(parameters, test_x, test_y, degree)
        if rmse < least_squares or least_squares == -1:
            least_squares = rmse
            least_squares_degree = degree
    return least_squares_degree

plt.figure()
plt.plot(train_x, train_y, 'bo')
plt.plot(test_x, test_y, 'ro')

axes = plt.gca()
axes.set_xlim([-5, 5])

for index, degree in enumerate(degrees_collection):
    coefficient = pol_regression(train_x, train_y, degree)
    expanded_x = expand_features(train_x, degree)
    plt.plot(train_x, expanded_x.dot(coefficient), colours[index])

plt.legend(('Training Data', 'Test Data', '$x^0$', '$x$', '$x^2$', '$x^3$', '$x^6$', '$x^{{{}}}$'.format('10')), loc = 'lower right')
plt.savefig('polynomial.png')
os.startfile('polynomial.png')


print("Least Squared Solution (degree): " + str(get_least_squares_solution()))