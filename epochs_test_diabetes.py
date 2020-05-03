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

import time


"""budeme měnit počet EPOCHS a sledovat vývoj accuracy a loss (cross entropy)"""

# LOAD DATASET (bank_marketing, cars, vehicle_silhouette,diabetes,messidor)
if __name__ == '__main__':

    df, y = load_datasets('diabetes')

    #epochs = [5,10,15,20,25,30,35,40,45,50,55,60,65]
    epochs = [5,10,20,30,40,50,60,70,80,90,100]


    learn_rates_nrfdw = list(np.linspace(1, 5, num=10))
    learn_rates_nrf = list(np.linspace(1, 5, num=10))
    learn_rates_nrfel = list(np.linspace(1, 5, num=10))
    learn_rates_nrfeldw = list(np.linspace(1, 5, num=10))

    accuracy_train_nrfdw = []
    accuracy_train_nrf = []
    accuracy_train_nrfel = []
    accuracy_train_nrfeldw = []
    accuracy_train_nrfeldw_ultra = []


    accuracy_test_nrfdw = []
    accuracy_test_nrf = []
    accuracy_test_nrfel = []
    accuracy_test_nrfeldw = []
    accuracy_test_nrfeldw_ultra = []


    loss_train_nrfdw = []
    loss_train_nrf = []
    loss_train_nrfel = []
    loss_train_nrfeldw = []
    loss_train_nrfeldw = []
    loss_train_nrfeldw_ultra = []



    loss_test_nrfdw = []
    loss_test_nrf = []
    loss_test_nrfel = []
    loss_test_nrfeldw = []
    loss_test_nrfeldw_ultra = []


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
            rf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=6, max_features='auto')
            X_train, X_test = df[train_index], df[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # y_train_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
            # y_test_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
            # y_train_keras = y_train_keras.astype('int64')
            # y_test_keras = y_test_keras.astype('int64')
            y_train = y_train.astype('int64')
            y_test = y_test.astype('int64')
            rf.fit(X_train, y_train)  # zde trénink random forestu

            accuracy_temp_train_nrfdw = []
            accuracy_temp_train_nrf = []
            accuracy_temp_train_nrfel = []
            accuracy_temp_train_nrfeldw = []
            accuracy_temp_train_nrfeldw_ultra = []


            accuracy_temp_test_nrfdw = []
            accuracy_temp_test_nrf = []
            accuracy_temp_test_nrfel = []
            accuracy_temp_test_nrfeldw = []
            accuracy_temp_test_nrfeldw_ultra = []


            loss_temp_train_nrfdw = []
            loss_temp_train_nrf = []
            loss_temp_train_nrfel = []
            loss_temp_train_nrfeldw = []
            loss_temp_train_nrfeldw_ultra = []


            loss_temp_test_nrfdw = []
            loss_temp_test_nrf = []
            loss_temp_test_nrfel = []
            loss_temp_test_nrfeldw = []
            loss_temp_test_nrfeldw_ultra = []

            for epoch in epochs:
                nrfdw = NeuralRandomForest(rf, 'NRF_analyticWeights_adam', X_train, y_train, output_func='softmax',
                                           cost_func='CrossEntropy',
                                           gamma_output=1, gamma=[1, 1])
                nrfdw.get_NRF_ensemble(epoch, 10, 0.0035, 0.02)
                predictions_test_nrfdw = nrfdw.predict(X_test)
                predictions_test_loss_nrfdw = nrfdw.predict_averaging_loss(X_test)
                predictions_train_nrfdw = nrfdw.predict(X_train)
                predictions_train_loss_nrfdw = nrfdw.predict_averaging_loss(X_train)

                results_test_nrfdw = classification_report(y_test, predictions_test_nrfdw, output_dict=True)
                results_train_nrfdw = classification_report(y_train, predictions_train_nrfdw, output_dict=True)

                accuracy_temp_test_nrfdw.append(results_test_nrfdw['accuracy'])
                accuracy_temp_train_nrfdw.append(results_train_nrfdw['accuracy'])

                loss_temp_test_nrfdw.append(log_loss(y_test,predictions_test_loss_nrfdw))
                loss_temp_train_nrfdw.append(log_loss(y_train,predictions_train_loss_nrfdw))




                nrf = NeuralRandomForest(rf, 'NRF_basic_adam', X_train, y_train, output_func='softmax',
                                         cost_func='CrossEntropy',
                                         gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                nrf.get_NRF_ensemble(epoch, 10, 0.002, 0.02)
                predictions_test_nrf = nrf.predict(X_test)
                predictions_test_loss_nrf = nrf.predict_averaging_loss(X_test)
                predictions_train_nrf = nrf.predict(X_train)
                predictions_train_loss_nrf = nrf.predict_averaging_loss(X_train)

                results_test_nrf = classification_report(y_test, predictions_test_nrf, output_dict=True)
                results_train_nrf = classification_report(y_train, predictions_train_nrf, output_dict=True)

                accuracy_temp_test_nrf.append(results_test_nrf['accuracy'])
                accuracy_temp_train_nrf.append(results_train_nrf['accuracy'])

                loss_temp_test_nrf.append(log_loss(y_test, predictions_test_loss_nrf))
                loss_temp_train_nrf.append(log_loss(y_train, predictions_train_loss_nrf))

                nrf_eldw = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_adam', X_train, y_train,
                                              output_func='softmax',
                                              cost_func='CrossEntropy',
                                              gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                nrf_eldw.get_NRF_ensemble(epoch, 10, 0.005, 0.02)
                predictions_test_nrf_eldw = nrf_eldw.predict(X_test)
                predictions_test_loss_nrf_eldw = nrf_eldw.predict_averaging_loss(X_test)
                predictions_train_nrf_eldw = nrf_eldw.predict(X_train)
                predictions_train_loss_nrf_eldw = nrf_eldw.predict_averaging_loss(X_train)

                results_test_nrf_eldw = classification_report(y_test, predictions_test_nrf_eldw, output_dict=True)
                results_train_nrf_eldw = classification_report(y_train, predictions_train_nrf_eldw, output_dict=True)

                accuracy_temp_test_nrfeldw.append(results_test_nrf_eldw['accuracy'])
                accuracy_temp_train_nrfeldw.append(results_train_nrf_eldw['accuracy'])

                loss_temp_test_nrfeldw.append(log_loss(y_test, predictions_test_loss_nrf_eldw))
                loss_temp_train_nrfeldw.append(log_loss(y_train, predictions_train_loss_nrf_eldw))

                nrf_eldw_ultra = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_ultra_adam', X_train, y_train,
                                              output_func='softmax',
                                              cost_func='CrossEntropy',
                                              gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                nrf_eldw_ultra.get_NRF_ensemble(epoch, 10, 0.004, 0.02)
                predictions_test_nrf_eldw_ultra = nrf_eldw_ultra.predict(X_test)
                predictions_test_loss_nrf_eldw_ultra = nrf_eldw_ultra.predict_averaging_loss(X_test)
                predictions_train_nrf_eldw_ultra = nrf_eldw_ultra.predict(X_train)
                predictions_train_loss_nrf_eldw_ultra = nrf_eldw_ultra.predict_averaging_loss(X_train)

                results_test_nrf_eldw_ultra = classification_report(y_test, predictions_test_nrf_eldw_ultra, output_dict=True)
                results_train_nrf_eldw_ultra = classification_report(y_train, predictions_train_nrf_eldw_ultra, output_dict=True)

                accuracy_temp_test_nrfeldw_ultra.append(results_test_nrf_eldw_ultra['accuracy'])
                accuracy_temp_train_nrfeldw_ultra.append(results_train_nrf_eldw_ultra['accuracy'])

                loss_temp_test_nrfeldw_ultra.append(log_loss(y_test, predictions_test_loss_nrf_eldw_ultra))
                loss_temp_train_nrfeldw_ultra.append(log_loss(y_train, predictions_train_loss_nrf_eldw_ultra))

            accuracy_train_nrfdw.append(accuracy_temp_train_nrfdw)
            accuracy_train_nrf.append(accuracy_temp_train_nrf)
            accuracy_train_nrfeldw.append(accuracy_temp_train_nrfeldw)
            #accuracy_train_nrfel.append(accuracy_temp_train_nrfel)
            accuracy_train_nrfeldw_ultra.append(accuracy_temp_train_nrfeldw_ultra)


            accuracy_test_nrfdw.append(accuracy_temp_test_nrfdw)
            accuracy_test_nrf.append(accuracy_temp_test_nrf)
            accuracy_test_nrfeldw.append(accuracy_temp_test_nrfeldw)
            accuracy_test_nrfeldw_ultra.append(accuracy_temp_test_nrfeldw_ultra)
            #accuracy_test_nrfel.append(accuracy_temp_test_nrfel)

            loss_train_nrfdw.append(loss_temp_train_nrfdw)
            loss_train_nrf.append(loss_temp_train_nrf)
            loss_train_nrfeldw.append(loss_temp_train_nrfeldw)
            loss_train_nrfeldw_ultra.append(loss_temp_train_nrfeldw_ultra)
            #loss_train_nrfel.append(loss_temp_train_nrfel)

            loss_test_nrfdw.append(loss_temp_test_nrfdw)
            loss_test_nrf.append(loss_temp_test_nrf)
            loss_test_nrfeldw.append(loss_temp_test_nrfeldw)
            loss_test_nrfeldw_ultra.append(loss_temp_test_nrfeldw_ultra)
            #loss_test_nrfel.append(loss_temp_test_nrfel)

    accuracy_train_nrfdw_temp = np.array(accuracy_train_nrfdw, dtype=np.float64)
    accuracy_train_nrfdw_mean = list(np.mean(accuracy_train_nrfdw_temp, axis=0))
    accuracy_train_nrfdw_std = list(np.std(accuracy_train_nrfdw_temp, axis=0))

    accuracy_test_nrfdw_temp = np.array(accuracy_test_nrfdw, dtype=np.float64)
    accuracy_test_nrfdw_mean = list(np.mean(accuracy_test_nrfdw_temp, axis=0))
    accuracy_test_nrfdw_std = list(np.std(accuracy_test_nrfdw_temp, axis=0))

    loss_train_nrfdw_temp = np.array(loss_train_nrfdw, dtype=np.float64)
    loss_train_nrfdw_mean = list(np.mean(loss_train_nrfdw_temp, axis=0))
    loss_train_nrfdw_std = list(np.std(loss_train_nrfdw_temp, axis=0))

    loss_test_nrfdw_temp = np.array(loss_test_nrfdw, dtype=np.float64)
    loss_test_nrfdw_mean = list(np.mean(loss_test_nrfdw_temp, axis=0))
    loss_test_nrfdw_std = list(np.std(loss_test_nrfdw_temp, axis=0))


    accuracy_train_nrf_temp = np.array(accuracy_train_nrf, dtype=np.float64)
    accuracy_train_nrf_mean = list(np.mean(accuracy_train_nrf_temp, axis=0))
    accuracy_train_nrf_std = list(np.std(accuracy_train_nrf_temp, axis=0))

    accuracy_test_nrf_temp = np.array(accuracy_test_nrf, dtype=np.float64)
    accuracy_test_nrf_mean = list(np.mean(accuracy_test_nrf_temp, axis=0))
    accuracy_test_nrf_std = list(np.std(accuracy_test_nrf_temp, axis=0))

    loss_train_nrf_temp = np.array(loss_train_nrf, dtype=np.float64)
    loss_train_nrf_mean = list(np.mean(loss_train_nrf_temp, axis=0))
    loss_train_nrf_std = list(np.std(loss_train_nrf_temp, axis=0))

    loss_test_nrf_temp = np.array(loss_test_nrf, dtype=np.float64)
    loss_test_nrf_mean = list(np.mean(loss_test_nrf_temp, axis=0))
    loss_test_nrf_std = list(np.std(loss_test_nrf_temp, axis=0))

    accuracy_train_nrfeldw_temp = np.array(accuracy_train_nrfeldw, dtype=np.float64)
    accuracy_train_nrfeldw_mean = list(np.mean(accuracy_train_nrfeldw_temp, axis=0))
    accuracy_train_nrfeldw_std = list(np.std(accuracy_train_nrfeldw_temp, axis=0))

    accuracy_test_nrfeldw_temp = np.array(accuracy_test_nrfeldw, dtype=np.float64)
    accuracy_test_nrfeldw_mean = list(np.mean(accuracy_test_nrfeldw_temp, axis=0))
    accuracy_test_nrfeldw_std = list(np.std(accuracy_test_nrfeldw_temp, axis=0))

    loss_train_nrfeldw_temp = np.array(loss_train_nrfeldw, dtype=np.float64)
    loss_train_nrfeldw_mean = list(np.mean(loss_train_nrfeldw_temp, axis=0))
    loss_train_nrfeldw_std = list(np.std(loss_train_nrfeldw_temp, axis=0))

    loss_test_nrfeldw_temp = np.array(loss_test_nrfeldw, dtype=np.float64)
    loss_test_nrfeldw_mean = list(np.mean(loss_test_nrfeldw_temp, axis=0))
    loss_test_nrfeldw_std = list(np.std(loss_test_nrfeldw_temp, axis=0))

    accuracy_train_nrfeldw_ultra_temp = np.array(accuracy_train_nrfeldw_ultra, dtype=np.float64)
    accuracy_train_nrfeldw_ultra_mean = list(np.mean(accuracy_train_nrfeldw_ultra_temp, axis=0))
    accuracy_train_nrfeldw_ultra_std = list(np.std(accuracy_train_nrfeldw_ultra_temp, axis=0))

    accuracy_test_nrfeldw_ultra_temp = np.array(accuracy_test_nrfeldw_ultra, dtype=np.float64)
    accuracy_test_nrfeldw_ultra_mean = list(np.mean(accuracy_test_nrfeldw_ultra_temp, axis=0))
    accuracy_test_nrfeldw_ultra_std = list(np.std(accuracy_test_nrfeldw_ultra_temp, axis=0))

    loss_train_nrfeldw_ultra_temp = np.array(loss_train_nrfeldw_ultra, dtype=np.float64)
    loss_train_nrfeldw_ultra_mean = list(np.mean(loss_train_nrfeldw_ultra_temp, axis=0))
    loss_train_nrfeldw_ultra_std = list(np.std(loss_train_nrfeldw_ultra_temp, axis=0))

    loss_test_nrfeldw_ultra_temp = np.array(loss_test_nrfeldw_ultra, dtype=np.float64)
    loss_test_nrfeldw_ultra_mean = list(np.mean(loss_test_nrfeldw_ultra_temp, axis=0))
    loss_test_nrfeldw_ultra_std = list(np.std(loss_test_nrfeldw_ultra_temp, axis=0))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, accuracy_train_nrf_mean, yerr=accuracy_train_nrf_std, ecolor='darkorange', marker='o')
    #ax.plot(epochs, accuracy_train_nrf,color = 'red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for j in range(accuracy_train_nrf_temp.shape[0]):
        ax.errorbar(epochs, accuracy_train_nrf_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, accuracy_train_nrf_mean,yerr = accuracy_train_nrf_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0,110])
    plt.title('NRF')
    plt.ylim([np.amin(accuracy_train_nrf_temp) - 0.05, np.amax(accuracy_train_nrf_temp) + 0.05])
    fig.savefig('NRF_epochs_acc_train_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, accuracy_test_nrf_mean, yerr=accuracy_test_nrf_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for j in range(accuracy_test_nrf_temp.shape[0]):
        ax.errorbar(epochs, accuracy_test_nrf_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, accuracy_test_nrf_mean,yerr = accuracy_test_nrf_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF')
    plt.ylim([np.amin(accuracy_test_nrf_temp) - 0.05, np.amax(accuracy_test_nrf_temp) + 0.05])
    fig.savefig('NRF_epochs_acc_test_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, loss_train_nrf_mean, yerr=loss_train_nrf_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for j in range(loss_train_nrf_temp.shape[0]):
        ax.errorbar(epochs, loss_train_nrf_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, loss_train_nrf_mean,yerr = loss_train_nrf_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF')
    plt.ylim([np.amin(loss_train_nrf_temp) - 0.05, np.amax(loss_train_nrf_temp) + 0.05])
    fig.savefig('NRF_epochs_loss_train_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, loss_test_nrf_mean, yerr=loss_test_nrf_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for j in range(loss_test_nrf_temp.shape[0]):
        ax.errorbar(epochs, loss_test_nrf_mean[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, loss_test_nrf_mean,yerr = loss_test_nrf_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF')
    plt.ylim([np.amin(loss_test_nrf_temp) - 0.05, np.amax(loss_test_nrf_temp) + 0.05])
    fig.savefig('NRF_epochs_loss_test_diabetes.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, accuracy_train_nrfdw_mean, yerr=accuracy_train_nrfdw_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for j in range(accuracy_train_nrfdw_temp.shape[0]):
        ax.errorbar(epochs, accuracy_train_nrfdw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, accuracy_train_nrfdw_mean,yerr = accuracy_train_nrfdw_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0,110])
    plt.title('NRF_DW')
    plt.ylim([np.amin(accuracy_train_nrfdw_temp) - 0.05, np.amax(accuracy_train_nrfdw_temp) + 0.05])
    fig.savefig('NRF_DW_epochs_acc_train_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, accuracy_test_nrfdw_mean, yerr=accuracy_test_nrfdw_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for j in range(accuracy_test_nrfdw_temp.shape[0]):
        ax.errorbar(epochs, accuracy_test_nrfdw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, accuracy_test_nrfdw_mean,yerr = accuracy_test_nrfdw_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_DW')
    plt.ylim([np.amin(accuracy_test_nrfdw_temp) - 0.05, np.amax(accuracy_test_nrfdw_temp) + 0.05])
    fig.savefig('NRF_DW_epochs_acc_test_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, loss_train_nrfdw_mean, yerr=loss_train_nrfdw_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for j in range(loss_train_nrfdw_temp.shape[0]):
        ax.errorbar(epochs, loss_train_nrfdw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, loss_train_nrfdw_mean,yerr = loss_train_nrfdw_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_DW')
    plt.ylim([np.amin(loss_train_nrfdw_temp) - 0.05, np.amax(loss_train_nrfdw_temp) + 0.05])
    fig.savefig('NRF_DW_epochs_loss_train_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, loss_test_nrfdw_mean, yerr=loss_test_nrfdw_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for j in range(loss_test_nrfdw_temp.shape[0]):
        ax.errorbar(epochs, loss_test_nrfdw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, loss_test_nrfdw_mean,yerr = loss_test_nrfdw_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_DW')
    plt.ylim([np.amin(loss_test_nrfdw_temp) - 0.05, np.amax(loss_test_nrfdw_temp) + 0.05])
    fig.savefig('NRF_DW_epochs_loss_test_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, accuracy_train_nrfeldw_mean, yerr=accuracy_train_nrfeldw_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for j in range(accuracy_train_nrfeldw_temp.shape[0]):
        ax.errorbar(epochs, accuracy_train_nrfeldw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, accuracy_train_nrfeldw_mean,yerr = accuracy_train_nrfeldw_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0,110])
    plt.title('NRF_EL_DW')
    plt.ylim([np.amin(accuracy_train_nrfeldw_temp) - 0.05, np.amax(accuracy_train_nrfeldw_temp) + 0.05])
    fig.savefig('NRF_EL_DW_epochs_acc_train_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, accuracy_test_nrfeldw_mean, yerr=accuracy_test_nrfeldw_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for j in range(accuracy_test_nrfeldw_temp.shape[0]):
        ax.errorbar(epochs, accuracy_test_nrfeldw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, accuracy_test_nrfeldw_mean,yerr = accuracy_test_nrfeldw_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_EL_DW')
    plt.ylim([np.amin(accuracy_test_nrfeldw_temp) - 0.05, np.amax(accuracy_test_nrfeldw_temp) + 0.05])
    fig.savefig('NRF_EL_DW_epochs_acc_test_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, loss_train_nrfeldw_mean, yerr=loss_train_nrfeldw_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for j in range(loss_train_nrfeldw_temp.shape[0]):
        ax.errorbar(epochs, loss_train_nrfeldw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, loss_train_nrfeldw_mean,yerr = loss_train_nrfeldw_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_EL_DW')
    plt.ylim([np.amin(loss_train_nrfeldw_temp) - 0.05, np.amax(loss_train_nrfeldw_temp) + 0.05])
    fig.savefig('NRF_EL_DW_epochs_loss_train_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, loss_test_nrfeldw_mean, yerr=loss_test_nrfeldw_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for j in range(loss_test_nrfeldw_temp.shape[0]):
        ax.errorbar(epochs, loss_test_nrfeldw_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, loss_test_nrfeldw_mean,yerr = loss_test_nrfeldw_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_EL_DW')
    plt.ylim([np.amin(loss_test_nrfeldw_temp) - 0.05, np.amax(loss_test_nrfeldw_temp) + 0.05])
    fig.savefig('NRF_EL_DW_epochs_loss_test_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, accuracy_train_nrfeldw_ultra_mean, yerr=accuracy_train_nrfeldw_ultra_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for j in range(accuracy_train_nrfeldw_ultra_temp.shape[0]):
        ax.errorbar(epochs, accuracy_train_nrfeldw_ultra_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, accuracy_train_nrfeldw_ultra_mean,yerr = accuracy_train_nrfeldw_ultra_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_EL_DW_identity')
    plt.ylim([min(accuracy_train_nrfeldw_ultra_temp) - 0.05, max(accuracy_train_nrfeldw_ultra_temp) + 0.05])
    fig.savefig('NRF_EL_DW_identity_epochs_acc_train_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, accuracy_test_nrfeldw_ultra_mean, yerr=accuracy_test_nrfeldw_ultra_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for j in range(accuracy_test_nrfeldw_ultra_temp.shape[0]):
        ax.errorbar(epochs, accuracy_test_nrfeldw_ultra_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, accuracy_test_nrfeldw_ultra_mean,yerr = accuracy_test_nrfeldw_ultra_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_EL_DW_identity')
    plt.ylim([np.amin(accuracy_test_nrfeldw_ultra_temp) - 0.05, np.amax(accuracy_test_nrfeldw_ultra_temp) + 0.05])
    fig.savefig('NRF_EL_DW_identity_epochs_acc_test_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, loss_train_nrfeldw_ultra_mean, yerr=loss_train_nrfeldw_ultra_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for j in range(loss_train_nrfeldw_ultra_temp.shape[0]):
        ax.errorbar(epochs, loss_train_nrfeldw_ultra_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, loss_train_nrfeldw_ultra_mean,yerr = loss_train_nrfeldw_ultra_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_EL_DW_identity')
    plt.ylim([np.amin(loss_train_nrfeldw_ultra_temp) - 0.05, np.amax(loss_train_nrfeldw_ultra_temp) + 0.05])
    fig.savefig('NRF_EL_DW_identity_epochs_loss_train_diabetes.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(epochs, loss_test_nrfeldw_ultra_mean, yerr=loss_test_nrfeldw_ultra_std, ecolor='darkorange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for j in range(loss_test_nrfeldw_ultra_temp.shape[0]):
        ax.errorbar(epochs, loss_test_nrfeldw_ultra_temp[j,:], yerr=None, marker='o',color = 'lightgray',alpha = 0.5,zorder=-32)
    ax.errorbar(epochs, loss_test_nrfeldw_ultra_mean,yerr = loss_test_nrfeldw_ultra_std,ecolor='darkorange',marker = 'o')
    plt.xlim([0, 110])
    plt.title('NRF_EL_DW_identity')
    plt.ylim([np.amin(loss_test_nrfeldw_ultra_temp) - 0.05, np.amax(loss_test_nrfeldw_ultra_temp) + 0.05])
    fig.savefig('NRF_EL_DW_identity_epochs_loss_test_diabetes.png')
