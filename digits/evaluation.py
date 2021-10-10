# -*- coding: utf-8 -*-
# Question 5
# Author: Weining Qu (wq2155)

from matplotlib import pylab
from pylab import *
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score


def get_accuracy(y_pred, y_true):
    return accuracy_score(y_true, y_pred)

def get_roc_auc_score(y_true, y_pred):
    if y_true.nunique() != 2:
        return np.nan
    else:
        return roc_auc_score(y_true, y_pred)

def get_precision(y_true, y_pred):
    if y_true.nunique() == 2:
        return precision_score(y_true, y_pred, average='binary')
    else:
        return precision_score(y_true, y_pred, average='macro')

# recall
def get_recall(y_true, y_pred):
    if y_true.nunique() == 2:
        return recall_score(y_true, y_pred, average='binary')
    else:
        return precision_score(y_true, y_pred, average='macro')

# f1
def get_f1(y_true, y_pred):
    if y_true.nunique() == 2:
        return f1_score(y_true, y_pred, average='binary')
    else:
        return f1_score(y_true, y_pred, average='macro')


def plot_roc(y_true, y_prob, predictor):
    plt.figure(figsize=(5, 5))
    title = predictor + '_ROC'

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    pylab.plot(fpr, tpr, color='darkorange', label='auc')
    pylab.plot([0, 1], [0, 1], color='navy', linestyle='--')
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.05])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title(title)
    plt.legend(['AUC = %0.4f' % auc])
    plt.tight_layout()
    plt.savefig('./Fig/' + title + '.png', bbox_inches='tight')
    plt.show()
    return
