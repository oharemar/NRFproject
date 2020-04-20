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


"""TEST PRO BETA1 parametr v první skryté vrstvě, BETA2 bude zatím nastaven na 1"""

# LOAD DATASET (bank_marketing, cars, vehicle_silhouette,diabetes,messidor)
if __name__ == '__main__':

    df, y = load_datasets('diabetes')

    learn_rates_nrfdw = list(np.linspace(0.25, 10, num=20))
    learn_rates_nrf = list(np.linspace(0.25, 10, num=20))
    learn_rates_nrfel = list(np.linspace(0.25, 10, num=20))
    learn_rates_nrfeldw = list(np.linspace(0.25, 10, num=20))

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

    rf_acc = []
    rf_f1 = []

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
            preds = rf.predict(X_test)
            res = classification_report(y_test, preds, output_dict=True)
            rf_acc.append(res['accuracy'])
            rf_f1.append(res['macro avg']['f1-score'])

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


            for eta, eta_nrfdw, eta_nrfel, eta_nrfeldw in zip(learn_rates_nrf, learn_rates_nrfdw, learn_rates_nrfel,
                                                              learn_rates_nrfeldw):
                nrfdw = NeuralRandomForest(rf, 'NRF_analyticWeights_adam', X_train, y_train, output_func='softmax',
                                           cost_func='CrossEntropy',
                                           gamma_output=1, gamma=[eta, 1])
                nrfdw.get_NRF_ensemble(30, 10, 0.0035, 0.02)
                predictions_nrfdw = nrfdw.predict(X_test)
                results_nrfdw = classification_report(y_test, predictions_nrfdw, output_dict=True)
                accuracy_temp_nrfdw.append(results_nrfdw['accuracy'])
                f_temp_nrfdw.append(results_nrfdw['macro avg']['f1-score'])

                nrf = NeuralRandomForest(rf, 'NRF_basic_adam', X_train, y_train, output_func='softmax',
                                         cost_func='CrossEntropy',
                                         gamma_output=1, gamma=[eta, 1])  # zde změna, gamma_output je 1
                nrf.get_NRF_ensemble(30, 10, 0.002, 0.02)
                predictions_nrf = nrf.predict(X_test)
                results_nrf = classification_report(y_test, predictions_nrf, output_dict=True)
                accuracy_temp_nrf.append(results_nrf['accuracy'])
                f_temp_nrf.append(results_nrf['macro avg']['f1-score'])


                nrf_eldw = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_adam', X_train, y_train,
                                              output_func='softmax',
                                              cost_func='CrossEntropy',
                                              gamma_output=1, gamma=[eta, 1])  # zde změna, gamma_output je 1
                nrf_eldw.get_NRF_ensemble(30, 10, 0.005, 0.02)
                predictions_nrf_eldw = nrf_eldw.predict(X_test)
                results_nrf_eldw = classification_report(y_test, predictions_nrf_eldw, output_dict=True)
                accuracy_temp_nrfeldw.append(results_nrf_eldw['accuracy'])
                f_temp_nrfeldw.append(results_nrf_eldw['macro avg']['f1-score'])

                nrf_eldw_ultra = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_ultra_adam', X_train, y_train,
                                              output_func='softmax',
                                              cost_func='CrossEntropy',
                                              gamma_output=1, gamma=[eta, 1])  # zde změna, gamma_output je 1
                nrf_eldw_ultra.get_NRF_ensemble(30, 10, 0.0045, 0.02)
                predictions_nrf_eldw_ultra = nrf_eldw_ultra.predict(X_test)
                results_nrf_eldw_ultra = classification_report(y_test, predictions_nrf_eldw_ultra, output_dict=True)
                accuracy_temp_nrfeldw_ultra.append(results_nrf_eldw_ultra['accuracy'])
                f_temp_nrfeldw_ultra.append(results_nrf_eldw_ultra['macro avg']['f1-score'])

            accuracy_nrfdw.append(accuracy_temp_nrfdw)
            accuracy_nrf.append(accuracy_temp_nrf)
            accuracy_nrfeldw.append(accuracy_temp_nrfeldw)
            accuracy_nrfeldw_ultra.append(accuracy_temp_nrfeldw_ultra)


            f_nrfdw.append(f_temp_nrfdw)
            f_nrf.append(f_temp_nrf)
            f_nrfeldw.append(f_temp_nrfeldw)
            f_nrfeldw_ultra.append(f_temp_nrfeldw_ultra)


    accuracy_nrfdw_temp = np.array(accuracy_nrfdw, dtype=np.float64)
    accuracy_nrfdw = list(np.mean(accuracy_nrfdw_temp, axis=0))
    accuracy_nrfdw_std = list(np.std(accuracy_nrfdw_temp, axis=0))
    accuracy_nrf_temp = np.array(accuracy_nrf, dtype=np.float64)
    accuracy_nrf = list(np.mean(accuracy_nrf_temp, axis=0))
    accuracy_nrf_std = list(np.std(accuracy_nrf_temp, axis=0))
    accuracy_nrfeldw_temp = np.array(accuracy_nrfeldw, dtype=np.float64)
    accuracy_nrfeldw = list(np.mean(accuracy_nrfeldw_temp, axis=0))
    accuracy_nrfeldw_std = list(np.std(accuracy_nrfeldw_temp, axis=0))
    accuracy_nrfeldw_ultra_temp = np.array(accuracy_nrfeldw_ultra, dtype=np.float64)
    accuracy_nrfeldw_ultra = list(np.mean(accuracy_nrfeldw_ultra_temp, axis=0))
    accuracy_nrfeldw_ultra_std = list(np.std(accuracy_nrfeldw_ultra_temp, axis=0))

    f_nrfdw_temp = np.array(f_nrfdw, dtype=np.float64)
    f_nrfdw = list(np.mean(f_nrfdw_temp, axis=0))
    f_nrfdw_std = list(np.std(f_nrfdw_temp, axis=0))

    f_nrf_temp = np.array(f_nrf, dtype=np.float64)
    f_nrf = list(np.mean(f_nrf_temp, axis=0))
    f_nrf_std = list(np.std(f_nrf_temp, axis=0))

    f_nrfeldw_temp = np.array(f_nrfeldw, dtype=np.float64)
    f_nrfeldw = list(np.mean(f_nrfeldw_temp, axis=0))
    f_nrfeldw_std = list(np.std(f_nrfeldw_temp, axis=0))

    f_nrfeldw_ultra_temp = np.array(f_nrfeldw_ultra, dtype=np.float64)
    f_nrfeldw_ultra = list(np.mean(f_nrfeldw_ultra_temp, axis=0))
    f_nrfeldw_ultra_std = list(np.std(f_nrfeldw_ultra_temp, axis=0))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrf, accuracy_nrf,yerr = accuracy_nrf_std,ecolor='darkorange',marker = 'o')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrf) - (learn_rates_nrf[1] - learn_rates_nrf[0]),
              max(learn_rates_nrf) + (learn_rates_nrf[1] - learn_rates_nrf[0])])
    plt.title('NRF')
    plt.ylim([min(accuracy_nrf) - 0.05, max(accuracy_nrf) + 0.05])
    fig.savefig('NRF_beta1_acc.png')

    fig = plt.figure()
    rf_vals = [statistics.mean(rf_acc) for _ in range(len(learn_rates_nrf))]
    rf_std = [statistics.stdev(rf_acc) for _ in range(len(learn_rates_nrf))]
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrf, accuracy_nrf, yerr=accuracy_nrf_std, ecolor='darkorange', marker='o',label = 'NRF accuracy')
    ax.errorbar(learn_rates_nrf, rf_vals, yerr=rf_std, color = 'red',ecolor='green', marker='o',label = 'RF accuracy')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrf) - (learn_rates_nrf[1] - learn_rates_nrf[0]),
              max(learn_rates_nrf) + (learn_rates_nrf[1] - learn_rates_nrf[0])])
    plt.title('NRF')
    plt.ylim([min([min(accuracy_nrf),min(rf_vals)]) - 0.05, max([max(accuracy_nrf),max(rf_vals)]) + 0.05])
    plt.legend(loc='upper right')
    fig.savefig('NRF_beta1_acc_with_RF.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrf, f_nrf,yerr = f_nrf_std,ecolor='darkorange',marker = 'o')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrf) - (learn_rates_nrf[1] - learn_rates_nrf[0]),
              max(learn_rates_nrf) + (learn_rates_nrf[1] - learn_rates_nrf[0])])
    plt.title('NRF')
    plt.ylim([min(f_nrf) - 0.05, max(f_nrf) + 0.05])
    fig.savefig('NRF_beta1_F1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrfdw, accuracy_nrfdw,yerr = accuracy_nrfdw_std,ecolor='darkorange',marker = 'o')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfdw) - (learn_rates_nrfdw[1] - learn_rates_nrfdw[0]),
              max(learn_rates_nrfdw) + (learn_rates_nrfdw[1] - learn_rates_nrfdw[0])])
    plt.title('NRF_DW')
    plt.ylim([min(accuracy_nrfdw) - 0.05, max(accuracy_nrfdw) + 0.05])
    fig.savefig('NRF_DW_beta1_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrfdw, accuracy_nrfdw, yerr=accuracy_nrfdw_std, ecolor='darkorange', marker='o',label='NRF_DW_accuracy')
    ax.errorbar(learn_rates_nrf, rf_vals, yerr=rf_std, color = 'red',ecolor='green', marker='o',label = 'RF accuracy')

    plt.xlabel(r'$\beta_1$')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfdw) - (learn_rates_nrfdw[1] - learn_rates_nrfdw[0]),
              max(learn_rates_nrfdw) + (learn_rates_nrfdw[1] - learn_rates_nrfdw[0])])
    plt.title('NRF_DW')
    plt.legend(loc='upper right')
    plt.ylim([min([min(accuracy_nrfdw),min(rf_vals)]) - 0.05, max([max(accuracy_nrfdw),max(rf_vals)]) + 0.05])
    fig.savefig('NRF_DW_beta1_acc_with_RF.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrfdw, f_nrfdw,yerr = f_nrfdw_std,ecolor='darkorange',marker = 'o')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrfdw) - (learn_rates_nrfdw[1] - learn_rates_nrfdw[0]),
              max(learn_rates_nrfdw) + (learn_rates_nrfdw[1] - learn_rates_nrfdw[0])])
    plt.title('NRF_DW')
    plt.ylim([min(f_nrfdw) - 0.05, max(f_nrfdw) + 0.05])
    fig.savefig('NRF_DW_beta1_F1.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrfeldw, accuracy_nrfeldw,yerr = accuracy_nrfeldw_std,ecolor='darkorange',marker = 'o')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfeldw) - (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]),
              max(learn_rates_nrfeldw) + (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0])])
    plt.title('NRF_EL_DW')
    plt.ylim([min(accuracy_nrfeldw) - 0.05, max(accuracy_nrfeldw) + 0.05])
    fig.savefig('NRF_EL_DW_beta1_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrfeldw, accuracy_nrfeldw, yerr=accuracy_nrfeldw_std, ecolor='darkorange', marker='o',label='NRF_EL_DW_accuracy')
    ax.errorbar(learn_rates_nrf, rf_vals, yerr=rf_std, color = 'red',ecolor='green', marker='o',label = 'RF accuracy')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfeldw) - (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]),
              max(learn_rates_nrfeldw) + (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0])])
    plt.title('NRF_EL_DW')
    plt.legend(loc='upper right')
    plt.ylim([min([min(accuracy_nrfeldw),min(rf_vals)]) - 0.05, max([max(accuracy_nrfeldw),max(rf_vals)]) + 0.05])
    fig.savefig('NRF_EL_DW_beta1_acc_with_RF.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrfeldw, f_nrfeldw,yerr = f_nrfeldw_std,ecolor='darkorange',marker = 'o')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrfeldw) - (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]),
              max(learn_rates_nrfeldw) + (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0])])
    plt.title('NRF_EL_DW')
    plt.ylim([min(f_nrfeldw) - 0.05, max(f_nrfeldw) + 0.05])
    fig.savefig('NRF_EL_DW_beta1_F1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrfeldw, accuracy_nrfeldw_ultra, yerr=accuracy_nrfeldw_ultra_std, ecolor='darkorange', marker='o')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfeldw) - (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]),
              max(learn_rates_nrfeldw) + (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0])])
    plt.title('NRF_EL_DW_identity')
    plt.ylim([min(accuracy_nrfeldw_ultra) - 0.05, max(accuracy_nrfeldw_ultra) + 0.05])
    fig.savefig('NRF_EL_DW_identity_beta1_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrfeldw, accuracy_nrfeldw_ultra, yerr=accuracy_nrfeldw_ultra_std, ecolor='darkorange', marker='o',
                label='NRF_EL_DW_identity_accuracy')
    ax.errorbar(learn_rates_nrf, rf_vals, yerr=rf_std, color='red', ecolor='green', marker='o', label='RF accuracy')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfeldw) - (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]),
              max(learn_rates_nrfeldw) + (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0])])
    plt.title('NRF_EL_DW_identity')
    plt.legend(loc='upper right')
    plt.ylim([min([min(accuracy_nrfeldw_ultra), min(rf_vals)]) - 0.05, max([max(accuracy_nrfeldw_ultra), max(rf_vals)]) + 0.05])
    fig.savefig('NRF_EL_DW_identity_beta1_acc_with_RF.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(learn_rates_nrfeldw, f_nrfeldw_ultra, yerr=f_nrfeldw_ultra_std, ecolor='darkorange', marker='o')
    plt.xlabel(r'$\beta_1$')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrfeldw) - (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0]),
              max(learn_rates_nrfeldw) + (learn_rates_nrfeldw[1] - learn_rates_nrfeldw[0])])
    plt.title('NRF_EL_DW_idendity')
    plt.ylim([min(f_nrfeldw_ultra) - 0.05, max(f_nrfeldw_ultra) + 0.05])
    fig.savefig('NRF_EL_DW_identity_beta1_F1.png')

