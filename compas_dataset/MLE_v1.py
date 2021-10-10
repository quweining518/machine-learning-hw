# -*- coding: utf-8 -*-
# Question 4
# Author: Weining Qu (wq2155)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import *


def GaussianTrain(data, target):
    """
    :param data: Input data
    :param mu: mean of X|y
    :param Sigma: covariance matrix of X|y
    :return: prob of X
    """

    Y = data[target].values
    X_all = data[[col for col in data.columns if col != target]]
    X_pos = X_all[Y == 1].values
    X_neg = X_all[Y == 0].values

    phi = (Y.sum()+1) / (len(Y)+2)  # Laplace Param phi for Bernoulli distribution of y (Prior)
    mu_pos = np.sum(X_pos, axis=0) / Y.sum()  # Param mean of X for X|y=1 (Multivariate Gaussian)
    mu_neg = np.sum(X_neg, axis=0) / (len(Y)-Y.sum())  # Param mean of X for X|y=0 (Multivariate Gaussian)

    X_pos_submu = X_pos - mu_pos
    X_neg_submu = X_neg - mu_neg

    Sigma_all = []
    for X_i in X_pos_submu:
        X_i = X_i.reshape(X_pos_submu.shape[1], 1)
        sigma_i = X_i.dot(X_i.T)
        Sigma_all.append(sigma_i)
    for X_i in X_neg_submu:
        X_i = X_i.reshape(X_neg_submu.shape[1], 1)
        sigma_i = X_i.dot(X_i.T)
        Sigma_all.append(sigma_i)
    Sigma_all = np.array(Sigma_all)
    Sigma = np.sum(Sigma_all, axis=0) / len(Y)  # Param - Covariance matrix (Multivariate Gaussian)

    return phi, mu_pos, mu_neg, Sigma

def GaussianProb(x, mu, Sigma):

    col = np.shape(Sigma)[0]
    Sigma_det = np.linalg.det(Sigma + np.eye(col) * 0.001)
    Sigma_inv = np.linalg.inv(Sigma + np.eye(col) * 0.001)
    x_mu = (x - mu).reshape((1, col))
    # P(X|y) follows Gaussian distribution
    prob = 1.0 / (np.power(np.power(2 * np.pi, col) * np.abs(Sigma_det), 0.5)) * \
           np.exp(-0.5 * x_mu.dot(Sigma_inv).dot(x_mu.T))[0][0]
    return prob

def GaussianPredict(testX, phi, mu_pos, mu_neg, Sigma):
    """

    :param data: testdata (only features)
    :param phi: P(Y=1)
    :param mu_pos: mean of X|y=1
    :param mu_neg: mean of X|y=0
    :param Sigma: Covariance matrix
    :return: predicted label array
    """
    testX = testX.values
    predY = []
    predY_prob = []
    for i in range(testX.shape[0]):
        prob_pos = GaussianProb(testX[i], mu_pos, Sigma) * phi
        prob_neg = GaussianProb(testX[i], mu_neg, Sigma) * (1-phi)
        if prob_pos >= prob_neg:
            predY.append(1)
            predY_prob.append(1-prob_pos)
        else:
            predY.append(0)
            predY_prob.append(prob_neg)
    predY = np.array(predY)
    predY_prob = np.array(predY_prob)
    return predY, predY_prob


################ Evaluation ###########################################################
def Accuracy(pred, true):
        return sum(pred==true) / len(pred)

def Test_trainsize(traindata, testdata, target):
    accu_pair = []
    for s in range(1,11):
        N = int(traindata.shape[0] * s/10)
        tr_part = traindata.iloc[:N,:]
        # print(tr_part)
        phi, mean_pos, mean_neg, Sigma = GaussianTrain(tr_part, target)
        testX = testdata[[col for col in testdata.columns if col != target]]
        testY = testdata[target]
        predY, probY = GaussianPredict(testX, phi, mean_pos, mean_neg, Sigma)
        accuracy_test = Accuracy(predY, testY)
        accu_pair.append((N, accuracy_test))
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
    traindata = pd.read_csv("propublicaTrain.csv", sep=',')
    testdata = pd.read_csv("propublicaTest.csv", sep=',')
    target = 'two_year_recid'

    phi, mean_pos, mean_neg, Sigma = GaussianTrain(traindata, target)
    testX = testdata[[col for col in testdata.columns if col != target]]
    testY = testdata[target]

    predY, probY = GaussianPredict(testX, phi, mean_pos, mean_neg, Sigma)
    accuracy_test = Accuracy(predY, testY)
    print("Testdata accuracy: {:.4f}".format(accuracy_test))

    plot_roc(testY, probY, "MLE")  # ROC Curve and AUC

    accu_diffsize = Test_trainsize(traindata, testdata, target)  # Test training sample size effect


    ############## Fairness analysis ################################################################
    df = pd.concat([testX['race'],pd.Series(predY), testY],axis=1)
    df.columns = ['race','ypred','ytrue']
    # Demographic parity
    df_race0 = df[df['race']==0]
    df_race1 = df[df['race']==1]
    p_race0_0 = (df_race0.shape[0]-df_race0['ypred'].sum())/df_race0.shape[0]
    p_race0_1 = df_race0['ypred'].sum() / df_race0.shape[0]
    p_race1_0 = (df_race1.shape[0]-df_race1['ypred'].sum())/df_race1.shape[0]
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