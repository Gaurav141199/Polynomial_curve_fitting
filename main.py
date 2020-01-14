import numpy as np
import pandas as pd
import csv
import poly
from matplotlib import pyplot as plt

# def PolyCoefficients(x, coeffs):
#     """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.
#
#     The coefficients must be in ascending order (``x**0`` to ``x**o``).
#     """
#     o = len(coeffs)
#     print(f'# This is a polynomial of order {ord}.')
#     y = 0
#     for i in range(o):
#         y += coeffs[i]*x**i
#     return y

def LinearRegression(x, y, m, normal):
    lbd = 0
    regularize = False
    n = x.shape[0]

    X = np.ones((n, 1), dtype = 'int')
    for i in range(1, m + 1):
        temp = x ** i
        temp = poly.normalize(temp)
        X = np.append(X, temp, axis = 1)

    theta = np.zeros((m + 1, 1))

    if regularize:

        lbd_errors = []
        newList = [10** i for i in range(-15,3,1)]

        for i in range(len(newList)):
            [lcost, ltheta] = poly.normal_equation(X, y, theta, newList[i])
            lbd_errors.append(lcost)

        # print(lbd_errors)
        min_error = min(lbd_errors)
        # print(min_error)
        lbd = newList[lbd_errors.index(min_error)]

    # print(lbd)

    # lbd = 5
    if normal:
        [tempCost, tempTheta] = poly.normal_equation(X, y, theta, lbd)

    else:
        [tempCost, tempTheta] = poly.gradientDescent(X, y, lbd)

    # print(X.shape)
    # print(tempCost)
    return [tempCost, tempTheta, X]

def main():

    data_path = 'Gaussian_noise.csv'
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # get header from first row
        headers = next(reader)
        # get all the rows as a list
        data = list(reader)
        # transform data into numpy array
        data = np.array(data).astype(float)

    # print(headers)
    # print(data.shape)
    # print(data[:3])
    x = data[:,:-1]
    y = data[:,-1]
    n = x.shape[0]
    # print(x.shape)
    m = 20 #max_degree

    costs = []
    thetas = []

    normal = False
    for i in range(1, m + 1):
        [cost_i, theta_i, X] = LinearRegression(x, y, i, normal)
        # print(cost_m)
        # print(theta_m)
        costs.append(cost_i)
        thetas.append(theta_i)

    # print(costs)
    min_cost = min(costs)
    degree = costs.index(min_cost) + 1
    W = thetas[degree - 1]
    # print(X.shape)
    print("Degree of Polynomial: " + str(degree))
    # print(W)
    var = poly.variance(X[:,: degree + 1], y, W)

    print("Variance Estimate: " + str(var))
    # print(1)
    # x_range = np.linspace(np.amin(x), np.max(x), 10)
    # x_range = np.linspace(0, 9, 100)
    # # print(2)
    # plt.plot(x_range, PolyCoefficients(x, W))
    # plt.show()
    # # print(1)


if __name__ == "__main__":
    main()
