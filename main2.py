import numpy as np
import pandas as pd
import csv
import poly
from matplotlib import pyplot as plt
from random import randrange
from random import seed

def cross_validation_split(dataset, folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def designMatrix(x, m):
    n = x.shape[0]
    # print(type(x[0]))
    X = np.ones((n, 1), dtype = 'int')
    for i in range(1, m + 1):
        temp = x ** i
        temp = poly.normalize(temp)
        temp = temp.reshape(n, -1)
        X = np.append(X, temp, axis = 1)

        return X

def LinearRegression(train_setX, train_setY, test_setX, test_setY, i, normal, m):
    lbd = 0
    regularize = False
    x = np.array(train_setX)
    y = np.array(train_setY)
    ty = np.array(test_setY)
    tx = np.array(test_setX)
    X = designMatrix(x, m)
    tX = designMatrix(tx, m)

    if regularize:

        lbd_errors = []
        newList = [10** i for i in range(-15,3,1)]

        for i in range(len(newList)):
            [lcost, ltheta] = poly.normal_equation(X, y, newList[i])
            lbd_errors.append(lcost)

        min_error = min(lbd_errors)
        lbd = newList[lbd_errors.index(min_error)]


    if normal:
        [tempCost, tempTheta] = poly.normal_equation(X, y, lbd)

    else:
        [tempCost, tempTheta] = poly.gradientDescent(X, y, lbd)


    K_cost = poly.computeCost(tX, ty, tempTheta)

    return [tempCost, tempTheta, X]

def main():

    data_path1 = 'train.csv'
    data_path2 = 'test.csv'
    with open(data_path1, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data1 = list(reader)
        data1 = np.array(data1)

    with open(data_path2, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data2 = list(reader)
        data2 = np.array(data2)



    trainx = data1[:,0]
    trainy = [float(i[1]) for i in data1]
    n = trainx.shape[0]
    testx = data2[:,0]
    trainx_months = [int((i.split('/'))[0]) for i in trainx]
    testx_months = [int((i.split('/'))[0]) for i in testx]

    costs = []
    thetas = []

    normal = True

    total_folds = 10
    fold_size = n/total_folds
    zipped = np.array(pd.concat([pd.DataFrame(trainx_months),pd.DataFrame(trainy)],axis=1));
    m = 20 #max_degree

    for i in range(1,m + 1):

        seed(1)
        folds = cross_validation_split(zipped, total_folds)
        running_cost = 0
        for j in range(total_folds):
            copySet1 = folds
            copySet2 = folds
            train_set = np.array(pd.DataFrame(folds).drop([j]))
            train_setX=[]
            train_setY=[]
            for c in train_set :
                for d in c:
                    train_setX.append(d[0])
                    train_setY.append(d[1])
            train_setX=np.array(train_setX)
            train_setY=np.array(train_setY)
            test_set = copySet2[j]
            test_setX = [i[0] for i in test_set]
            test_setY = [i[1] for i in test_set]


            [cost_i, theta_i, X] = LinearRegression(train_setX, train_setY, test_setX, test_setY, i, normal, m)
            running_cost += cost_i
            running_cost /= total_folds
            # print(cost_m)
            print(theta_i)
            costs.append(running_cost)
            thetas.append(theta_i)

    min_cost = min(costs)
    degree = costs.index(min_cost) + 1
    W = thetas[degree - 1]
    print("Degree of Polynomial: " + str(degree))
    print(W)





if __name__ == "__main__":
    main()
