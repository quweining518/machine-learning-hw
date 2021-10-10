# -*- coding: utf-8 -*-
# Question 4
# Author: Weining Qu (wq2155)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import *

def NBClassifier(data, target):
    Y = data[target].values
    X_all = data[[col for col in data.columns if col != target]]

    # Categorize data by target class
    X_byclass = {}
    Y_byclass = {}
    phi = {}
    for i in np.unique(Y):
        X_byclass[i] = X_all[Y == i]
        Y_byclass[i] = Y[Y == i]
        phi[i] = (len(Y_byclass[i])+1)/ (len(Y)+2)

    # Calculate count probabilities of features
    prob_X = {}
    for i in np.unique(Y):
        prob_X[i] = {}
        for j in range(X_all.shape[1]):
            X_j = X_byclass[i].iloc[:,j]
            prob_X[i][j] = X_j.value_counts()/len(X_j)

        # Model "age" feature as continuous variable following Gaussian distribution
        prob_X[i][1] = [np.mean(X_byclass[i].iloc[:, 1]), np.var(X_byclass[i].iloc[:, 1])]
    return phi, prob_X


def NBpredict(testX, phi, prob_X):
    def GaussianProb(x, mean, var):
        # (np.exp(-(x-mean)**2/(2*var)))*(1/np.sqrt(2*np.pi*var))
        gaussian_prob = np.exp(-(x-mean)**2/(2*var)) * (1/np.sqrt(2*np.pi*var))
        return gaussian_prob

    predY = []
    predY_prob = []
    for i in range(testX.shape[0]):
        X_i = testX.iloc[i,:].values
        Cond_pos = 1
        for s in range(len(X_i)):
            if (s != 1) and (X_i[s] in prob_X[1][s].index.tolist()):
                Cond_pos = Cond_pos * prob_X[1][s][X_i[s]]
            else:  # "age"
                Cond_pos = Cond_pos * GaussianProb(X_i[s],prob_X[1][s][0], prob_X[1][s][1])
        prob_pos = phi[1] * Cond_pos

        Cond_neg = 1
        for s in range(len(X_i)):
            if (s != 1) and (X_i[s] in prob_X[0][s].index.tolist()):
                Cond_neg = Cond_neg * prob_X[0][s][X_i[s]]
            else:  # "age"
                Cond_neg = Cond_neg * GaussianProb(X_i[s], prob_X[1][s][0], prob_X[1][s][1])
        prob_neg = phi[0] * Cond_neg

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
    return sum(pred == true) / len(pred)

def Test_trainsize(traindata, testdata, target):
    accu_pair = []
    for s in range(1,11):
        N = int(traindata.shape[0] * s/10)
        tr_part = traindata.iloc[:N,:]
        # print(tr_part)
        phi, prob_X = NBClassifier(tr_part, target)
        testX = testdata[[col for col in testdata.columns if col != target]]
        testY = testdata[target]
        predY, predprobY = NBpredict(testX, phi, prob_X)
        accuracy_test = Accuracy(predY, testY)
        accu_pair.append((N, accuracy_test))
    x, y = zip(*accu_pair)
    y_max = np.argmax(y)
    plt.figure()
    plt.plot(x,y)
    plt.scatter(x[y_max],y[y_max], c='red')
    plt.xlabel("Training sample size")
    plt.ylabel("Accuracy")
    plt.title("Naive Bayes")
    plt.savefig("./Fig/NB_trainsizetest.png")
    plt.show()

    return accu_pair


if __name__ == "__main__":
    traindata = pd.read_csv("propublicaTrain.csv", sep=',')
    testdata = pd.read_csv("propublicaTest.csv", sep=',')
    target = 'two_year_recid'

    phi, prob_X = NBClassifier(traindata, target)
    testX = testdata[[col for col in testdata.columns if col != target]]
    testY = testdata[target]

    predY, predprobY = NBpredict(testX, phi, prob_X)
    accuracy_test = Accuracy(predY, testY)
    print("Testdata accuracy: {:.4f}".format(accuracy_test))

    plot_roc(testY, predprobY, "NB")  # ROC Curve and AUC

    accu_diffsize = Test_trainsize(traindata, testdata, target)  # Test training sample size effect


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