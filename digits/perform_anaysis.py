from evaluation import *
import pandas as pd

result_mle = pd.read_csv("testresult_MLE.csv")
result_KNN = pd.read_csv("testresult_KNN.csv")

accuracy_mle = get_accuracy(result_mle['ytrue'], result_mle['ypred'])
print("Accuracy MLE: ", round(accuracy_mle,4))
precision_mle = get_precision(result_mle['ytrue'], result_mle['ypred'])
print("Precision MLE: ", round(precision_mle,4))
recall_mle = get_recall(result_mle['ytrue'], result_mle['ypred'])
print("Recall MLE: ", round(recall_mle,4))
f1_mle = get_f1(result_mle['ytrue'], result_mle['ypred'])
print("f1 score MLE: ", round(f1_mle,4))

accuracy_KNN = get_accuracy(result_KNN['ytrue'], result_KNN['ypred'])
print("Accuracy KNN: ", round(accuracy_KNN,4))
precision_KNN = get_precision(result_KNN['ytrue'], result_KNN['ypred'])
print("Precision KNN: ", round(precision_KNN,4))
recall_KNN = get_recall(result_KNN['ytrue'], result_KNN['ypred'])
print("Recall KNN: ", round(recall_KNN,4))
f1_KNN= get_f1(result_KNN['ytrue'], result_KNN['ypred'])
print("f1 score KNN: ", round(f1_KNN,4))