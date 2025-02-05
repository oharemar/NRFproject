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
import statistics
from sklearn.model_selection import KFold
from helpful_functions import *
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import seaborn as sns

import time



# LOAD DATASET (bank_marketing, cars, vehicle_silhouette,diabetes,messidor)
if __name__ == '__main__':

    df, y = load_datasets('vehicle_silhouette')

    #epochs = [5,10,15,20,25,30,35,40,45,50,55,60,65]
    beta1 = [1,3,5,7,10,12]
    beta2 = [1,3,5,7,10,12]
    n = 10 # number of experiments (2 times 5 fold cross val)

    accuracy_nrfdw = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)
    accuracy_nrf = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)
    accuracy_nrfeldw = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)
    accuracy_nrfeldw_ultra = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)
    accuracy_rf = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)


    f1_nrfdw = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)
    f1_nrf = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)
    f1_nrfeldw = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)
    f1_nrfeldw_ultra = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)
    f1_rf = np.zeros((len(beta2), len(beta1), n), dtype=np.float64)

    i = 0
    kf = KFold(n_splits=5)
    for _ in range(2):
        for train_index, test_index in kf.split(df):
            X_train, X_test = df[train_index], df[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # y_train_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
            # y_test_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
            # y_train_keras = y_train_keras.astype('int64')
            # y_test_keras = y_test_keras.astype('int64')
            y_train = y_train.astype('int64')
            y_test = y_test.astype('int64')
            rf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=6,
                                        max_features='auto')
            rf.fit(X_train, y_train)  # zde trénink random forestu
            predictions_rf = rf.predict(X_test)
            results_rf = classification_report(y_test, predictions_rf, output_dict=True)
            for beta2_number, j in zip(beta2, range(len(beta2))):
                for beta1_number, k in zip(beta1, range(len(beta1))):
                    accuracy_rf[j, k, i] = results_rf['accuracy']
                    f1_rf[j, k, i] = results_rf['macro avg']['f1-score']
                    nrfdw = NeuralRandomForest(rf, 'NRF_analyticWeights_adam', X_train, y_train, output_func='softmax',
                                               cost_func='CrossEntropy',
                                               gamma_output=1, gamma=[beta1_number, beta2_number])
                    nrfdw.get_NRF_ensemble(30, 10, 0.0035, 0.02)
                    predictions_test_nrfdw = nrfdw.predict(X_test)
                    results_test_nrfdw = classification_report(y_test, predictions_test_nrfdw, output_dict=True)
                    accuracy_nrfdw[j,k,i] = results_test_nrfdw['accuracy']
                    f1_nrfdw[j,k,i] = results_test_nrfdw['macro avg']['f1-score']

                    nrf = NeuralRandomForest(rf, 'NRF_basic_adam', X_train, y_train, output_func='softmax',
                                             cost_func='CrossEntropy',
                                             gamma_output=1, gamma=[beta1_number, beta2_number])  # zde změna, gamma_output je 1
                    nrf.get_NRF_ensemble(30, 10, 0.002, 0.02)
                    predictions_test_nrf = nrf.predict(X_test)
                    results_test_nrf = classification_report(y_test, predictions_test_nrf, output_dict=True)
                    accuracy_nrf[j, k, i] = results_test_nrf['accuracy']
                    f1_nrf[j, k, i] = results_test_nrf['macro avg']['f1-score']

                    nrf_eldw = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_adam', X_train, y_train,
                                                  output_func='softmax',
                                                  cost_func='CrossEntropy',
                                                  gamma_output=1, gamma=[beta1_number, beta2_number])  # zde změna, gamma_output je 1
                    nrf_eldw.get_NRF_ensemble(30, 10, 0.005, 0.02)
                    predictions_test_nrf_eldw = nrf_eldw.predict(X_test)
                    results_test_nrf_eldw = classification_report(y_test, predictions_test_nrf_eldw, output_dict=True)
                    accuracy_nrfeldw[j, k, i] = results_test_nrf_eldw['accuracy']
                    f1_nrfeldw[j, k, i] = results_test_nrf_eldw['macro avg']['f1-score']

                    nrf_eldw_ultra = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_ultra_adam', X_train, y_train,
                                                  output_func='softmax',
                                                  cost_func='CrossEntropy',
                                                  gamma_output=1, gamma=[beta1_number, beta2_number])  # zde změna, gamma_output je 1
                    nrf_eldw_ultra.get_NRF_ensemble(30, 10, 0.0045, 0.02)
                    predictions_test_nrf_eldw_ultra = nrf_eldw_ultra.predict(X_test)
                    results_test_nrf_eldw_ultra = classification_report(y_test, predictions_test_nrf_eldw_ultra, output_dict=True)
                    accuracy_nrfeldw_ultra[j, k, i] = results_test_nrf_eldw_ultra['accuracy']
                    f1_nrfeldw_ultra[j, k, i] = results_test_nrf_eldw_ultra['macro avg']['f1-score']

            i = i + 1



    accuracy_nrf_mean = np.mean(accuracy_nrf,axis=2)
    accuracy_nrf_std = np.std(accuracy_nrf,axis=2)
    f1_nrf_mean = np.mean(f1_nrf, axis=2)
    f1_nrf_std = np.std(f1_nrf, axis=2)

    accuracy_nrfdw_mean = np.mean(accuracy_nrfdw, axis=2)
    accuracy_nrfdw_std = np.std(accuracy_nrfdw, axis=2)
    f1_nrfdw_mean = np.mean(f1_nrfdw, axis=2)
    f1_nrfdw_std = np.std(f1_nrfdw, axis=2)

    accuracy_nrfeldw_mean = np.mean(accuracy_nrfeldw, axis=2)
    accuracy_nrfeldw_std = np.std(accuracy_nrfeldw, axis=2)
    f1_nrfeldw_mean = np.mean(f1_nrfeldw, axis=2)
    f1_nrfeldw_std = np.std(f1_nrfeldw, axis=2)

    accuracy_nrfeldw_ultra_mean = np.mean(accuracy_nrfeldw_ultra, axis=2)
    accuracy_nrfeldw_ultra_std = np.std(accuracy_nrfeldw_ultra, axis=2)
    f1_nrfeldw_ultra_mean = np.mean(f1_nrfeldw_ultra, axis=2)
    f1_nrfeldw_ultra_std = np.std(f1_nrfeldw_ultra, axis=2)

    accuracy_rf_mean = np.mean(accuracy_rf, axis=2)
    accuracy_rf_std = np.std(accuracy_rf, axis=2)
    f1_rf_mean = np.mean(f1_rf, axis=2)
    f1_rf_std = np.std(f1_rf, axis=2)

    min_acc = min(np.amin(accuracy_nrf_mean),np.amin(accuracy_nrfdw_mean),np.amin(accuracy_nrfeldw_mean),np.amin(accuracy_nrfeldw_ultra_mean),
                  np.amin(accuracy_rf_mean))
    max_acc = max(np.amax(accuracy_nrf_mean),np.amax(accuracy_nrfdw_mean),np.amax(accuracy_nrfeldw_mean),np.amax(accuracy_nrfeldw_ultra_mean),
                  np.amax(accuracy_rf_mean))
    min_f1 = min(np.amin(f1_nrf_mean), np.amin(f1_nrfdw_mean), np.amin(f1_nrfeldw_mean),
                  np.amin(f1_nrfeldw_ultra_mean),
                  np.amin(f1_rf_mean))
    max_f1 = max(np.amax(f1_nrf_mean), np.amax(f1_nrfdw_mean), np.amax(f1_nrfeldw_mean),
                 np.amax(f1_nrfeldw_ultra_mean),
                 np.amax(f1_rf_mean))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(accuracy_nrf_mean, vmin=min_acc, vmax=max_acc)
    plt.colorbar(im, label='Accuracy')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('NRF')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j, u'{:.3f}\n\u00B1\n{:.3f}'.format(accuracy_nrf_mean[j, k], accuracy_nrf_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('NRF_beta1_beta2_acc.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(f1_nrf_mean, vmin=min_f1, vmax=max_f1)
    plt.colorbar(im, label='F1-score')
    #plt.clim(min_f1, max_f1)
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('NRF')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j, u'{:.3f}\n\u00B1\n{:.3f}'.format(f1_nrf_mean[j, k], f1_nrf_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('NRF_beta1_beta2_f1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(accuracy_nrfdw_mean, vmin=min_acc, vmax=max_acc)
    plt.colorbar(im, label='Accuracy')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('NRF_DW')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j, u'{:.3f}\n\u00B1\n{:.3f}'.format(accuracy_nrfdw_mean[j, k], accuracy_nrfdw_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('NRF_DW_beta1_beta2_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(f1_nrfdw_mean, vmin=min_f1, vmax=max_f1)
    plt.colorbar(im, label='F1-score')
    # plt.clim(min_f1, max_f1)
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('NRF_DW')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j, u'{:.3f}\n\u00B1\n{:.3f}'.format(f1_nrfdw_mean[j, k], f1_nrfdw_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('NRF_DW_beta1_beta2_f1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(accuracy_nrfeldw_mean, vmin=min_acc, vmax=max_acc)
    plt.colorbar(im, label='Accuracy')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('NRF_EL_DW')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j, u'{:.3f}\n\u00B1\n{:.3f}'.format(accuracy_nrfeldw_mean[j, k], accuracy_nrfeldw_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('NRF_EL_DW_beta1_beta2_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(f1_nrfeldw_mean, vmin=min_f1, vmax=max_f1)
    plt.colorbar(im, label='F1-score')
    # plt.clim(min_f1, max_f1)
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('NRF_EL_DW')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j, u'{:.3f}\n\u00B1\n{:.3f}'.format(f1_nrfeldw_mean[j, k], f1_nrfeldw_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('NRF_EL_DW_beta1_beta2_f1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(accuracy_nrfeldw_ultra_mean, vmin=min_acc, vmax=max_acc)
    plt.colorbar(im, label='Accuracy')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('NRF_EL_DW_identity')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j,
                           u'{:.3f}\n\u00B1\n{:.3f}'.format(accuracy_nrfeldw_ultra_mean[j, k], accuracy_nrfeldw_ultra_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('NRF_EL_DW_identity_beta1_beta2_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(f1_nrfeldw_ultra_mean, vmin=min_f1, vmax=max_f1)
    plt.colorbar(im, label='F1-score')
    # plt.clim(min_f1, max_f1)
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('NRF_EL_DW_identity')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j, u'{:.3f}\n\u00B1\n{:.3f}'.format(f1_nrfeldw_ultra_mean[j, k], f1_nrfeldw_ultra_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('NRF_EL_DW_identity_beta1_beta2_f1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(accuracy_rf_mean, vmin=min_acc, vmax=max_acc)
    plt.colorbar(im, label='Accuracy')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('RF')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j,
                           u'{:.3f}\n\u00B1\n{:.3f}'.format(accuracy_rf_mean[j, k],
                                                            accuracy_rf_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('RF_beta1_beta2_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(len(beta1)), labels=beta1)
    plt.yticks(np.arange(len(beta2)), labels=beta2)
    im = ax.imshow(f1_rf_mean, vmin=min_f1, vmax=max_f1)
    plt.colorbar(im, label='F1-score')
    # plt.clim(min_f1, max_f1)
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title('RF')
    for k in range(len(beta1)):
        for j in range(len(beta2)):
            text = ax.text(k, j,
                           u'{:.3f}\n\u00B1\n{:.3f}'.format(f1_rf_mean[j, k], f1_rf_std[j, k]),
                           ha="center", va="center", color="red", fontsize=7)
    fig.savefig('RF_beta1_beta2_f1.png')



