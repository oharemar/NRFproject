import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from NeuralRandomForest_parallel import NeuralRandomForest
from sklearn.metrics import average_precision_score
import statistics
from sklearn.model_selection import KFold
from helpful_functions import *
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import time

if __name__ == '__main__':


    df, y = load_datasets('messidor') # messidor je binární

    n_classes = 2
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)  # split to train and test dataset
    rf = RandomForestClassifier(n_estimators=10,max_depth=6,max_features='auto')
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    scores_lr = lr.decision_function(X_test)
    fpr_lr,tpr_lr,thr = metrics.roc_curve(y_test,scores_lr,pos_label=1)

    precision_lr,recall_lr,t = metrics.precision_recall_curve(y_test,scores_lr,pos_label=1)

    rf.fit(X_train,y_train)
    nrf = NeuralRandomForest(rf, 'NRF_basic_adam', X_train, y_train, output_func='softmax',
                                       cost_func='CrossEntropy',
                                       gamma_output=1.5, gamma=[1, 1])
    nrf.get_NRF_ensemble(30, 10, 0.002, 0.02)
    scores_nrf = nrf.predict_averaging_loss(X_test)[:,1]
    fpr_nrf, tpr_nrf, thresholds_nrf = metrics.roc_curve(y_test, scores_nrf, pos_label=1)
    precision_nrf,recall_nrf,tnrf = metrics.precision_recall_curve(y_test,scores_nrf,pos_label=1)


    scores = rf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    precision_rf,recall_rf,trf = metrics.precision_recall_curve(y_test,scores,pos_label=1)


    roc_auc_rf = metrics.auc(fpr, tpr)
    roc_auc_nrf = metrics.auc(fpr_nrf, tpr_nrf)
    roc_auc_lr = metrics.auc(fpr_lr,tpr_lr)

    avg_prec_rf = average_precision_score(y_test,scores)
    avg_prec_lr = average_precision_score(y_test,scores_lr)
    avg_prec_nrf = average_precision_score(y_test,scores_nrf)

    auc_rf = metrics.auc(recall_rf,precision_rf)
    auc_nrf = metrics.auc(recall_nrf,precision_nrf)
    auc_lr = metrics.auc(recall_lr,precision_lr)
    '''
    plt.figure()
    lw = 2
    plt.plot(recall_lr, precision_lr, color='darkorange',
             lw=lw, label='PR curve LR (AP = {})(AUC = {})'.format(round(avg_prec_lr,2),round(auc_lr,2)))
    plt.plot(recall_nrf, precision_nrf, color='red',
             lw=lw, label='PR curve NRF (AP = {})(AUC = {})'.format(round(avg_prec_nrf,2),round(auc_nrf,2)))
    plt.plot(recall_rf, precision_rf, color='blue',
             lw=lw, label='PR curve RF (AP = {})(AUC = {})'.format(round(avg_prec_rf,2),round(auc_rf,2)))
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

'''
    plt.figure()
    lw = 2
    plt.plot(fpr_lr, tpr_lr, color='darkorange',
             lw=lw, label='ROC curve LR (area = %0.2f)' % roc_auc_lr)
    plt.plot(fpr_nrf, tpr_nrf, color='red',
             lw=lw, label='ROC curve NRF (area = %0.2f)' % roc_auc_nrf)
    plt.plot(fpr, tpr, color='blue',
             lw=lw, label='ROC curve NRF (area = %0.2f)' % roc_auc_rf)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

