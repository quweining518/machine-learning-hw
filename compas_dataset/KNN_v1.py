# -*- coding: utf-8 -*-
# Question 4
# Author: Weining Qu (wq2155)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import *

def KNNClassifier(traindata, testX, target, k, norm='L2'):
    def dist(x, y):
        if norm == 'L1':
            dist = np.linalg.norm(x-y, 1)
        elif norm == 'L2':
            dist = np.linalg.norm(x-y, 2)
        elif norm == 'Linf':
            dist = np.linalg.norm(x-y, np.inf)
        return dist

    trainY = traindata[target].values
    trainX = traindata[[col for col in traindata.columns if col != target]].values
    testX = testX.values

    N_tr = trainX.shape[0]
    N_te = testX.shape[0]

    dist_mat = np.zeros((N_te, N_tr))
    for i in range(N_te):
        for j in range(N_tr):
            dist_mat[i][j] = dist(testX[i],trainX[j])
    predY = []
    for i in range(N_te):
        dist_k_near = np.argsort(dist_mat[i])[:k]
        Y_k_near = trainY[dist_k_near]
        predY_i = np.argmax(np.bincount(Y_k_near.tolist()))
        predY.append(predY_i)
    predY = np.array(predY)
    return predY


def Accuracy(pred, true):
    return sum(pred == true) / len(pred)

def Test_trainsize(traindata, testdata, target, k, norm):
    accu_pair = []
    for s in range(1,11):
        N = int(traindata.shape[0] * s/10)
        tr_part = traindata.iloc[:N,:]
        testY = testdata[target]
        testX = testdata[[col for col in testdata.columns if col != target]]
        predY = KNNClassifier(tr_part, testX, target, k, norm)
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
    traindata = pd.read_csv("propublicaTrain.csv", sep=',')
    testdata = pd.read_csv("propublicaTest.csv", sep=',')
    target = 'two_year_recid'
    testY = testdata[target]
    testX = testdata[[col for col in testdata.columns if col != target]]

    norms = ['L1','L2','Linf']
    accuracies = np.zeros((20,3))

    # Test 20 different values of K and 3 types of norm (L1, L2, Linf)
    for k in range(18,19):
        for n in range(len(norms)):
            predY = KNNClassifier(traindata, testX, target, k, norms[n])
            accu = Accuracy(predY, testY)
            accuracies[k-1][n] = accu
    accu_max = accuracies.max()
    accu_max_param = np.unravel_index(np.argmax(accuracies), accuracies.shape)  # K=18, L1-norm
    print("Testdata accuracy: {:.4f}".format(accu_max))


    ############## Fairness analysis ################################################################
    df = pd.concat([testX['race'], pd.Series(predY), testY], axis=1)
    df.columns = ['race', 'ypred', 'ytrue']
    # Demographic parity
    df_race0 = df[df['race'] == 0]
    df_race1 = df[df['race'] == 1]
    p_race0_0 = (df_race0.shape[0] - df_race0['ypred'].sum()) / df_race0.shape[0]
    p_race0_1 = df_race0['ypred'].sum() / df_race0.shape[0]
    p_race1_0 = (df_race1.shape[0] - df_race1['ypred'].sum()) / df_race1.shape[0]
    p_race1_1 = df_race1['ypred'].sum() / df_race1.shape[0]

    # Equalized Odds (EO)
    df_race0_y0 = df_race0[df_race0['ytrue'] == 0]
    eo_race0_00 = (df_race0_y0.shape[0] - df_race0_y0['ypred'].sum()) / df_race0_y0.shape[0]
    df_race0_y1 = df_race0[df_race0['ytrue'] == 1]
    eo_race0_11 = df_race0_y1['ypred'].sum() / df_race0_y1.shape[0]
    df_race1_y0 = df_race1[df_race1['ytrue'] == 0]
    eo_race1_00 = (df_race1_y0.shape[0] - df_race1_y0['ypred'].sum()) / df_race1_y0.shape[0]
    df_race1_y1 = df_race1[df_race1['ytrue'] == 1]
    eo_race1_11 = df_race1_y1['ypred'].sum() / df_race1_y1.shape[0]

    # Predictive Parity (PP)
    df_race0_hy0 = df_race0[df_race0['ypred'] == 0]
    pp_race0_00 = (df_race0_hy0.shape[0] - df_race0_hy0['ytrue'].sum()) / df_race0_hy0.shape[0]
    df_race0_hy1 = df_race0[df_race0['ypred'] == 1]
    pp_race0_11 = df_race0_hy1['ytrue'].sum() / df_race0_hy1.shape[0]
    df_race1_hy0 = df_race1[df_race1['ypred'] == 0]
    pp_race1_00 = (df_race1_hy0.shape[0] - df_race1_hy0['ytrue'].sum()) / df_race1_hy0.shape[0]
    df_race1_hy1 = df_race1[df_race1['ypred'] == 1]
    pp_race1_11 = df_race1_y1['ytrue'].sum() / df_race1_y1.shape[0]