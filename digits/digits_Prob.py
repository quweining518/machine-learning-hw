# -*- coding: utf-8 -*-
# Question 5
# Author: Weining Qu (wq2155)

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

class MLE_Classifier(object):
    def __init__(self):
        self.X_byclass = {}
        self.Y_byclass = {}
        self.phi = {}
        self.mu = {}
        self.Sigma = {}

    def fit(self, traindata, target):
        self.tr_Y = target
        self.tr_X = traindata
        self.cls = np.unique(self.tr_Y)  # predict labels

        for i in np.unique(self.tr_Y):
            print(i)
            self.Y_byclass[i] = self.tr_Y[self.tr_Y == i]
            self.X_byclass[i] = self.tr_X.loc[self.Y_byclass[i].index,:]
            self.phi[i] = (len(self.Y_byclass[i]) + 1) / (len(self.tr_Y) + 2)
            self.mu[i] = np.sum(self.X_byclass[i], axis=0) / len(self.Y_byclass[i])  # Param mean of X for X|y=1 (Multivariate Gaussian)
            X_submu = self.X_byclass[i] - self.mu[i]

            Sigma_all = []
            for s in range(X_submu.shape[0]):
                X_i = X_submu.iloc[s,:].values.reshape(X_submu.shape[1], 1)
                sigma_i = X_i.dot(X_i.T)
                Sigma_all.append(sigma_i)
            Sigma_all = np.array(Sigma_all)
            self.Sigma[i] = np.sum(Sigma_all, axis=0) / len(self.Y_byclass[i])  # Param - Covariance matrix (Multivariate Gaussian)


    def GaussianProb(self, x, mu, Sigma):
        col = np.shape(Sigma)[0]
        Sigma_det = np.linalg.det(Sigma + np.eye(col) * 0.001)
        Sigma_inv = np.linalg.inv(Sigma + np.eye(col) * 0.001)
        x_mu = (x - mu).values.reshape((1, col))
        prob = 1.0 / (np.power(np.abs(Sigma_det), 0.5)) * np.exp(-0.5 * x_mu.dot(Sigma_inv).dot(x_mu.T))[0][0]
        return prob


    def predict(self, testX):
        testX = testX.values

        predY = []
        for i in range(testX.shape[0]):
            prob_all = pd.Series()
            for cls in self.cls:
                prob_i = self.GaussianProb(testX[i], self.mu[cls] * self.phi[cls],self.Sigma[cls])
                prob_all.loc[prob_i] = cls
            prob_max = prob_all.index.max()
            predY.append(prob_all.loc[prob_max])  # return class with highest probability
        predY = np.array(predY)
        return predY


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

def Test_trainsize(X,y,testsize):
    accu_pair = []
    for s in range(1,11):
        N = int(X.shape[0] * s/10)
        tr_part = X.iloc[:N,:]
        y_part = y.iloc[:N]
        train_X, test_X, train_y, test_y = train_test_split(tr_part, y_part,
                        test_size=testsize)  # train test set split (test size 0.2)

        MLE = MLE_Classifier()
        MLE.fit(train_X, train_y)
        predY = MLE.predict(test_X)

        accuracy_test = Accuracy(predY, test_y)
        accu_pair.append((len(train_y), accuracy_test))
    x, y = zip(*accu_pair)
    y_max = np.argmax(y)
    plt.figure()
    plt.plot(x,y)
    plt.scatter(x[y_max],y[y_max], c='red')
    plt.xlabel("Training sample size")
    plt.ylabel("Accuracy")
    plt.title("MLE")
    plt.savefig("./Fig/MLE_trainsizetest.png")
    plt.show()

    return accu_pair



if __name__ == '__main__':
    digits = loadmat('digits.mat')
    X = pd.DataFrame(digits['X'])
    X_float = X.astype(np.float32) / 255  # Transform uint8 to float32
    Y = pd.DataFrame(digits['Y']).astype(np.int).iloc[:,0]  # Transform uint8 to int

    coverrate = (X_float != 0).sum() / X_float.shape[0]
    X_cover = X_float.loc[:, coverrate[coverrate > 0.05].index]  # Cover rate > 5%

    var_x_sort = X_cover.var(axis=0).sort_values(ascending=False)
    X_reduce = X_cover[var_x_sort.index[:150].tolist()]  # top variance variables
    testsize = 0.1

    # Start
    train_X, test_X, train_y, test_y = train_test_split(X_reduce, Y, test_size=testsize)  # train test set split (test size 0.2)
    MLE = MLE_Classifier()
    MLE.fit(train_X,train_y)
    predY = MLE.predict(test_X)
    accu_full = Accuracy(predY, test_y)
    print("Testdata accuracy: {:.4f}".format(accu_full))

    result = np.array([predY,test_y.values]).T
    # df_result = pd.DataFrame(result, columns=['ypred','ytrue'])
    # df_result.to_csv("testresult_MLE.csv",index=False)

    #################### Test sample size ##############################################################################
    accu_diffsize = Test_trainsize(X_reduce, Y, testsize)  # Test training sample size effect
