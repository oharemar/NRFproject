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
import time

# LOAD DATASET (bank_marketing, cars, vehicle_silhouette,diabetes,messidor)
if __name__ == '__main__':

    df, y = load_datasets('diabetes')

    learn_rates_nrfdw = list(np.linspace(0.0002, 0.008, num=15))
    learn_rates_nrf = list(np.linspace(0.0002, 0.008, num=15))
    learn_rates_nrfel = list(np.linspace(0.001, 0.015, num=15))
    learn_rates_nrfeldw = list(np.linspace(0.001, 0.015, num=15))
    learn_rates_nrfeldw_ultra = list(np.linspace(0.001, 0.015, num=15))


    accuracy_nrfdw = []
    accuracy_nrf = []
    accuracy_nrfel = []
    accuracy_nrfeldw = []
    accuracy_nrfeldw_ultra = []


    f_nrfdw = []
    f_nrf = []
    f_nrfel = []
    f_nrfeldw = []
    f_nrfeldw_ultra = []


    kf = KFold(n_splits=5)

    final_results_vals = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                          'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
    final_results_stds = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                          'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
    for _ in range(2):
        for train_index, test_index in kf.split(df):
            rf = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=6, max_features='auto')
            X_train, X_test = df[train_index], df[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # y_train_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
            # y_test_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
            # y_train_keras = y_train_keras.astype('int64')
            # y_test_keras = y_test_keras.astype('int64')
            y_train = y_train.astype('int64')
            y_test = y_test.astype('int64')
            rf.fit(X_train, y_train)  # zde trénink random forestu

            accuracy_temp_nrfdw = []
            accuracy_temp_nrf = []
            accuracy_temp_nrfel = []
            accuracy_temp_nrfeldw = []
            accuracy_temp_nrfeldw_ultra = []


            f_temp_nrfdw = []
            f_temp_nrf = []
            f_temp_nrfel = []
            f_temp_nrfeldw = []
            f_temp_nrfeldw_ultra = []


            for eta, eta_nrfdw, eta_nrfel, eta_nrfeldw, eta_ultra in zip(learn_rates_nrf, learn_rates_nrfdw, learn_rates_nrfel,
                                                              learn_rates_nrfeldw,learn_rates_nrfeldw_ultra):
                nrfdw = NeuralRandomForest(rf, 'NRF_analyticWeights_adam', X_train, y_train, output_func='softmax',
                                           cost_func='CrossEntropy',
                                           gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                nrfdw.get_NRF_ensemble(30, 10, eta_nrfdw, 0.02)
                predictions_nrfdw = nrfdw.predict(X_test)
                results_nrfdw = classification_report(y_test, predictions_nrfdw, output_dict=True)
                accuracy_temp_nrfdw.append(results_nrfdw['accuracy'])
                f_temp_nrfdw.append(results_nrfdw['macro avg']['f1-score'])

                nrf = NeuralRandomForest(rf, 'NRF_basic_adam', X_train, y_train, output_func='softmax',
                                         cost_func='CrossEntropy',
                                         gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                nrf.get_NRF_ensemble(30, 10, eta, 0.02)
                predictions_nrf = nrf.predict(X_test)
                results_nrf = classification_report(y_test, predictions_nrf, output_dict=True)
                accuracy_temp_nrf.append(results_nrf['accuracy'])
                f_temp_nrf.append(results_nrf['macro avg']['f1-score'])
                '''
                nrf_el = NeuralRandomForest(rf, 'NRF_extraLayer_adam', X_train, y_train, output_func='softmax',
                                            cost_func='CrossEntropy',
                                            gamma_output=1, gamma=[1.3, 1.3])  # zde změna, gamma_output je 1
                nrf_el.get_NRF_ensemble(30, 10, eta_nrfel, 0.02)
                predictions_nrf_el = nrf_el.predict(X_test)
                results_nrf_el = classification_report(y_test, predictions_nrf_el, output_dict=True)
                accuracy_temp_nrfel.append(results_nrf_el['accuracy'])
                f_temp_nrfel.append(results_nrf_el['macro avg']['f1-score'])
                '''

                nrf_eldw = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_adam', X_train, y_train,
                                              output_func='softmax',
                                              cost_func='CrossEntropy',
                                              gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                nrf_eldw.get_NRF_ensemble(30, 10, eta_nrfeldw, 0.02)
                predictions_nrf_eldw = nrf_eldw.predict(X_test)
                results_nrf_eldw = classification_report(y_test, predictions_nrf_eldw, output_dict=True)
                accuracy_temp_nrfeldw.append(results_nrf_eldw['accuracy'])
                f_temp_nrfeldw.append(results_nrf_eldw['macro avg']['f1-score'])

                nrf_eldw_ultra = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_ultra_adam', X_train, y_train,
                                              output_func='softmax',
                                              cost_func='CrossEntropy',
                                              gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                nrf_eldw_ultra.get_NRF_ensemble(30, 10, eta_ultra, 0.02)
                predictions_nrf_eldw_ultra = nrf_eldw_ultra.predict(X_test)
                results_nrf_eldw_ultra = classification_report(y_test, predictions_nrf_eldw_ultra, output_dict=True)
                accuracy_temp_nrfeldw_ultra.append(results_nrf_eldw_ultra['accuracy'])
                f_temp_nrfeldw_ultra.append(results_nrf_eldw_ultra['macro avg']['f1-score'])

            accuracy_nrfdw.append(accuracy_temp_nrfdw)
            accuracy_nrf.append(accuracy_temp_nrf)
            accuracy_nrfeldw.append(accuracy_temp_nrfeldw)
            accuracy_nrfeldw_ultra.append(accuracy_temp_nrfeldw_ultra)

            #accuracy_nrfel.append(accuracy_temp_nrfel)
            f_nrfdw.append(f_temp_nrfdw)
            f_nrf.append(f_temp_nrf)
            f_nrfeldw.append(f_temp_nrfeldw)
            f_nrfeldw_ultra.append(f_temp_nrfeldw_ultra)
            #f_nrfel.append(f_temp_nrfel)

    accuracy_nrfdw_temp = np.array(accuracy_nrfdw, dtype=np.float64)
    for j in range(accuracy_nrfdw_temp.shape[0]):
        accuracy_nrfdw_temp[j,:] = accuracy_nrfdw_temp[j,:]/np.amax(accuracy_nrfdw_temp[j,:])
    accuracy_nrfdw = list(np.mean(accuracy_nrfdw_temp, axis=0))
    accuracy_nrfdw_std = list(np.std(accuracy_nrfdw_temp, axis=0))
    accuracy_nrf_temp = np.array(accuracy_nrf, dtype=np.float64)
    for j in range(accuracy_nrf_temp.shape[0]):
        accuracy_nrf_temp[j,:] = accuracy_nrf_temp[j,:]/np.amax(accuracy_nrf_temp[j,:])
    accuracy_nrf = list(np.mean(accuracy_nrf_temp, axis=0))
    accuracy_nrf_std = list(np.std(accuracy_nrf_temp, axis=0))
    accuracy_nrfeldw_temp = np.array(accuracy_nrfeldw, dtype=np.float64)
    for j in range(accuracy_nrfeldw_temp.shape[0]):
        accuracy_nrfeldw_temp[j,:] = accuracy_nrfeldw_temp[j,:]/np.amax(accuracy_nrfeldw_temp[j,:])
    accuracy_nrfeldw = list(np.mean(accuracy_nrfeldw_temp, axis=0))
    accuracy_nrfeldw_std = list(np.std(accuracy_nrfeldw_temp, axis=0))
    accuracy_nrfeldw_ultra_temp = np.array(accuracy_nrfeldw_ultra, dtype=np.float64)
    for j in range(accuracy_nrfeldw_ultra_temp.shape[0]):
        accuracy_nrfeldw_ultra_temp[j,:] = accuracy_nrfeldw_ultra_temp[j,:]/np.amax(accuracy_nrfeldw_ultra_temp[j,:])
    accuracy_nrfeldw_ultra = list(np.mean(accuracy_nrfeldw_ultra_temp, axis=0))
    accuracy_nrfeldw_ultra_std = list(np.std(accuracy_nrfeldw_ultra_temp, axis=0))

    #accuracy_nrfel = np.array(accuracy_nrfel, dtype=np.float64)
    #accuracy_nrfel = list(np.mean(accuracy_nrfel, axis=0))
    f_nrfdw_temp = np.array(f_nrfdw, dtype=np.float64)
    for j in range(f_nrfdw_temp.shape[0]):
        f_nrfdw_temp[j, :] = f_nrfdw_temp[j,:]/np.amax(f_nrfdw_temp[j,:])
    f_nrfdw = list(np.mean(f_nrfdw_temp, axis=0))
    f_nrfdw_std = list(np.std(f_nrfdw_temp, axis=0))

    f_nrf_temp = np.array(f_nrf, dtype=np.float64)
    for j in range(f_nrfdw_temp.shape[0]):
        f_nrf_temp[j, :] = f_nrf_temp[j,:]/np.amax(f_nrf_temp[j,:])
    f_nrf = list(np.mean(f_nrf_temp, axis=0))
    f_nrf_std = list(np.std(f_nrf_temp, axis=0))

    f_nrfeldw_temp = np.array(f_nrfeldw, dtype=np.float64)
    for j in range(f_nrfeldw_temp.shape[0]):
        f_nrfeldw_temp[j, :] = f_nrfeldw_temp[j,:]/np.amax(f_nrfeldw_temp[j,:])
    f_nrfeldw = list(np.mean(f_nrfeldw_temp, axis=0))
    f_nrfeldw_std = list(np.std(f_nrfeldw_temp, axis=0))

    f_nrfeldw_ultra_temp = np.array(f_nrfeldw_ultra, dtype=np.float64)
    for j in range(f_nrfeldw_ultra_temp.shape[0]):
        f_nrfeldw_ultra_temp[j, :] = f_nrfeldw_ultra_temp[j,:]/np.amax(f_nrfeldw_ultra_temp[j,:])
    f_nrfeldw_ultra = list(np.mean(f_nrfeldw_ultra_temp, axis=0))
    f_nrfeldw_ultra_std = list(np.std(f_nrfeldw_ultra_temp, axis=0))

    #f_nrfel = np.array(f_nrfel, dtype=np.float64)
    #f_nrfel = list(np.mean(f_nrfel, axis=0))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.bar(learn_rates_nrf, accuracy_nrf, width=0.8 * (learn_rates_nrf[1] - learn_rates_nrf[0]))
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrf) - (learn_rates_nrf[1] - learn_rates_nrf[0]),
              max(learn_rates_nrf) + (learn_rates_nrf[1] - learn_rates_nrf[0])])
    for j in range(accuracy_nrf_temp.shape[0]):
        ax.errorbar(learn_rates_nrf, accuracy_nrf_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(learn_rates_nrf, accuracy_nrf,yerr = accuracy_nrf_std,ecolor='darkorange',marker = 'o')
    plt.title('NRF')
    plt.ylim([np.amin(accuracy_nrf_temp) - 0.05, np.amax(accuracy_nrf_temp) + 0.05])
    fig.savefig('NRF_learning_rate_acc_norm.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.bar(learn_rates_nrf, f_nrf, width=0.8 * (learn_rates_nrf[1] - learn_rates_nrf[0]))
    plt.xlabel('Learning rate')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrf) - (learn_rates_nrf[1] - learn_rates_nrf[0]),
              max(learn_rates_nrf) + (learn_rates_nrf[1] - learn_rates_nrf[0])])
    for j in range(f_nrf_temp.shape[0]):
        ax.errorbar(learn_rates_nrf, f_nrf_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(learn_rates_nrf, f_nrf,yerr = f_nrf_std,ecolor='darkorange',marker = 'o')
    plt.title('NRF')
    plt.ylim([np.amin(f_nrf_temp) - 0.05, np.amax(f_nrf_temp) + 0.05])
    fig.savefig('NRF_learning_rate_F1_norm.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.bar(learn_rates_nrfdw, accuracy_nrfdw, width=0.8 * (learn_rates_nrfdw[1] - learn_rates_nrfdw[0]))
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfdw) - (learn_rates_nrfdw[1] - learn_rates_nrfdw[0]),
              max(learn_rates_nrfdw) + (learn_rates_nrfdw[1] - learn_rates_nrfdw[0])])
    for j in range(accuracy_nrfdw_temp.shape[0]):
        ax.errorbar(learn_rates_nrfdw, accuracy_nrfdw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(learn_rates_nrfdw, accuracy_nrfdw,yerr = accuracy_nrfdw_std,ecolor='darkorange',marker = 'o')
    plt.title('NRF_DW')
    plt.ylim([np.amin(accuracy_nrfdw_temp) - 0.05, np.amax(accuracy_nrfdw_temp) + 0.05])
    fig.savefig('NRF_DW_learning_rate_acc_norm.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.bar(learn_rates_nrfdw, f_nrfdw, width=0.8 * (learn_rates_nrfdw[1] - learn_rates_nrfdw[0]))
    plt.xlabel('Learning rate')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrfdw) - (learn_rates_nrfdw[1] - learn_rates_nrfdw[0]),
              max(learn_rates_nrfdw) + (learn_rates_nrfdw[1] - learn_rates_nrfdw[0])])
    for j in range(f_nrfdw_temp.shape[0]):
        ax.errorbar(learn_rates_nrfdw, f_nrfdw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(learn_rates_nrfdw, f_nrfdw,yerr = f_nrfdw_std,ecolor='darkorange',marker = 'o')
    plt.title('NRF_DW')
    plt.ylim([np.amin(f_nrfdw_temp) - 0.05, np.amax(f_nrfdw_temp) + 0.05])
    fig.savefig('NRF_DW_learning_rate_F1_norm.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.bar(learn_rates_nrfeldw, accuracy_nrfeldw, width=0.8 * (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]))
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfeldw) - (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]),
              max(learn_rates_nrfeldw) + (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0])])
    for j in range(accuracy_nrfeldw_temp.shape[0]):
        ax.errorbar(learn_rates_nrfeldw, accuracy_nrfeldw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(learn_rates_nrfeldw, accuracy_nrfeldw,yerr = accuracy_nrfeldw_std,ecolor='darkorange',marker = 'o')
    plt.title('NRF_EL_DW')
    plt.ylim([np.amin(accuracy_nrfeldw_temp) - 0.05, np.amax(accuracy_nrfeldw_temp) + 0.05])
    fig.savefig('NRF_EL_DW_learning_rate_acc_norm.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.bar(learn_rates_nrfeldw, f_nrfeldw, width=0.8 * (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]))
    plt.xlabel('Learning rate')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrfeldw) - (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]),
              max(learn_rates_nrfeldw) + (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0])])
    for j in range(f_nrfeldw_temp.shape[0]):
        ax.errorbar(learn_rates_nrfeldw, f_nrfeldw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(learn_rates_nrfeldw, f_nrfeldw,yerr = f_nrfeldw_std,ecolor='darkorange',marker = 'o')
    plt.title('NRF_EL_DW')
    plt.ylim([np.amin(f_nrfeldw_temp) - 0.05, np.amax(f_nrfeldw_temp) + 0.05])
    fig.savefig('NRF_EL_DW_learning_rate_F1_norm.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.bar(learn_rates_nrfeldw, accuracy_nrfeldw, width=0.8 * (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]))
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfeldw_ultra) - (learn_rates_nrfeldw_ultra[1] - learn_rates_nrfeldw_ultra[0]),
              max(learn_rates_nrfeldw_ultra) + (learn_rates_nrfeldw_ultra[1] - learn_rates_nrfeldw_ultra[0])])
    for j in range(accuracy_nrfeldw_ultra_temp.shape[0]):
        ax.errorbar(learn_rates_nrfeldw_ultra, accuracy_nrfeldw_ultra_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(learn_rates_nrfeldw_ultra, accuracy_nrfeldw_ultra, yerr=accuracy_nrfeldw_ultra_std, ecolor='darkorange', marker='o')
    plt.title('NRF_EL_DW_identity')
    plt.ylim([np.amin(accuracy_nrfeldw_ultra_temp) - 0.05, np.amax(accuracy_nrfeldw_ultra_temp) + 0.05])
    fig.savefig('NRF_EL_DW_identity_learning_rate_acc_norm.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.bar(learn_rates_nrfeldw, f_nrfeldw, width=0.8 * (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]))
    plt.xlabel('Learning rate')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrfeldw_ultra) - (learn_rates_nrfeldw_ultra[1] - learn_rates_nrfeldw_ultra[0]),
              max(learn_rates_nrfeldw_ultra) + (learn_rates_nrfeldw_ultra[1] - learn_rates_nrfeldw_ultra[0])])
    for j in range(f_nrfeldw_ultra_temp.shape[0]):
        ax.errorbar(learn_rates_nrfeldw_ultra, f_nrfeldw_ultra_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(learn_rates_nrfeldw_ultra, f_nrfeldw_ultra, yerr=f_nrfeldw_ultra_std, ecolor='darkorange', marker='o')
    plt.title('NRF_EL_DW_identity')
    plt.ylim([np.amin(f_nrfeldw_ultra_temp) - 0.05, np.amax(f_nrfeldw_ultra_temp) + 0.05])
    fig.savefig('NRF_EL_DW_identity_learning_rate_F1_norm.png')