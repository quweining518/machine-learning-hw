# -*- coding: utf-8 -*-
# Question 5
# Author: Weining Qu (wq2155)

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

class KNNClassifier(object):
    def __init__(self, k, norm):
        """

        :param k: number of nearest neighbors
        :param norm: distance metrics
        """
        self.k = k
        self.norm = norm

    # def l2_distance(self, X_i):
    #     X_i = X_i.reshape(1, -1)
    #     X_train = self.trainX.reshape(self.trainX.shape[0], -1)
    #     distances = np.power(np.tile(X_i, (X_train.shape[0], 1)) - X_train, 2).sum(axis=1)
    #     return distances

    def predict(self, trainX, trainY, testX):
        trainY = trainY.values
        trainX = trainX.values
        testX = testX.values

        N_tr = trainX.shape[0]
        N_te = testX.shape[0]

        dist_mat = np.zeros((N_te, N_tr))
        for i in range(N_te):
            for j in range(N_tr):
                dist_mat[i][j] = self.dist(testX[i],trainX[j])
        predY = []
        for i in range(N_te):
            dist_k_near = np.argsort(dist_mat[i])[:self.k]
            Y_k_near = trainY[dist_k_near]
            predY_i = np.argmax(np.bincount(Y_k_near.tolist()))
            predY.append(predY_i)
        predY = np.array(predY)
        return predY

    def dist(self, x, y):
        if self.norm == 'L1':
            dist = np.linalg.norm(x - y, 1)
        elif self.norm == 'L2':
            dist = np.linalg.norm(x - y, 2)
        elif self.norm == 'Linf':
            dist = np.linalg.norm(x - y, np.inf)
        return dist


def train_test_split(X, y, test_size=0.1):
    if X.shape[0] != y.shape[0]:
        return "Unmatched sample size between X %s and y %s" % (X.shape[0], y.shape[0])
    N = X.shape[0]
    ls = np.arange(0,N)
    np.random.shuffle(ls)
    test_ls = ls[:int(N*test_size)].tolist()
    train_ls = ls[int(N*test_size):].tolist()
    train_X = X.iloc[train_ls,:]
    test_X = X.iloc[test_ls,:]
    train_y = y.iloc[train_ls]
    test_y = y.iloc[test_ls]

    return train_X, test_X, train_y, test_y


def Accuracy(pred, true):
    return sum(pred == true) / len(pred)

def Test_trainsize(traindata, testdata, target, k, norm):
    accu_pair = []
    for s in range(1,11):
        N = int(traindata.shape[0] * s/10)
        tr_part = traindata.iloc[:N,:]
        testY = testdata[target]
        testX = testdata[[col for col in testdata.columns if col != target]]
        predY = KNNClassifier(traindata, testX, target, k, norm)
        accuracy_test = Accuracy(predY, testY)
        accu_pair.append((N, accuracy_test))

    x, y = zip(*accu_pair)
    y_max = np.argmax(y)
    plt.figure()
    plt.plot(x,y)
    plt.scatter(x[y_max],y[y_max], c='red')
    plt.xlabel("Training sample size")
    plt.ylabel("Accuracy")
    plt.title("KNN")
    plt.savefig("./Fig/KNN_trainsizetest.png")
    plt.show()
    return accu_pair



if __name__ == '__main__':
    digits = loadmat('digits.mat')
    X = pd.DataFrame(digits['X'])
    X_float = X.astype(np.float32) / 255  # Transform uint8 to float32
    Y = pd.DataFrame(digits['Y']).astype(np.int).iloc[:, 0]  # Transform uint8 to int

    coverrate = (X_float != 0).sum() / X_float.shape[0]
    X_cover = X_float.loc[:, coverrate[coverrate > 0.05].index]  # Cover rate > 5%

    var_x_sort = X_cover.var(axis=0).sort_values(ascending=False)
    X_reduce = X_cover[var_x_sort.index[:150].tolist()]  # top variance variables
    testsize = 0.1

    # Start
    train_X, test_X, train_y, test_y = train_test_split(X_reduce, Y, test_size=testsize)
    norm = 'L2'
    accuracies = np.zeros((20,1))

    # Test 20 different values of K and 3 types of norm (L1, L2, Linf)
    for k in range(10,30):
        print(k)
        KNN = KNNClassifier(k, norm)
        predY = KNN.predict(train_X, train_y, test_X)
        accu = Accuracy(predY, test_y)
        print(accu)
        accuracies[k-10] = accu
    accu_max = accuracies.max()
    accu_max_param = np.unravel_index(np.argmax(accuracies), accuracies.shape)  # K=9, L1-norm
    print("Testdata accuracy: {:.4f}".format(accu_max))

    # k1 = 10
    # KNN1 = KNNClassifier(k1, norm)
    # predY1 = KNN1.predict(train_X, train_y, test_X)
    # result = np.array([predY1, test_y.values]).T
    # df_result = pd.DataFrame(result, columns=['ypred','ytrue'])
    # df_result.to_csv("testresult_KNN.csv",index=False)
