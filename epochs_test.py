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

    df, y = load_datasets('vehicle_silhouette')

    epochs = [5,10,15,20,25,30,35,40,45,50,55,60,65]

    learn_rates_nrfdw = list(np.linspace(1, 5, num=10))
    learn_rates_nrf = list(np.linspace(1, 5, num=10))
    learn_rates_nrfel = list(np.linspace(1, 5, num=10))
    learn_rates_nrfeldw = list(np.linspace(1, 5, num=10))

    accuracy_train_nrfdw = []
    accuracy_train_nrf = []
    accuracy_train_nrfel = []
    accuracy_train_nrfeldw = []

    accuracy_test_nrfdw = []
    accuracy_test_nrf = []
    accuracy_test_nrfel = []
    accuracy_test_nrfeldw = []

    loss_train_nrfdw = []
    loss_train_nrf = []
    loss_train_nrfel = []
    loss_train_nrfeldw = []

    loss_test_nrfdw = []
    loss_test_nrf = []
    loss_test_nrfel = []
    loss_test_nrfeldw = []

    f_nrfdw = []
    f_nrf = []
    f_nrfel = []
    f_nrfeldw = []

    kf = KFold(n_splits=5)

    final_results_vals = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                          'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
    final_results_stds = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                          'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}

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

        accuracy_temp_train_nrfdw = []
        accuracy_temp_train_nrf = []
        accuracy_temp_train_nrfel = []
        accuracy_temp_train_nrfeldw = []

        accuracy_temp_test_nrfdw = []
        accuracy_temp_test_nrf = []
        accuracy_temp_test_nrfel = []
        accuracy_temp_test_nrfeldw = []

        loss_temp_train_nrfdw = []
        loss_temp_train_nrf = []
        loss_temp_train_nrfel = []
        loss_temp_train_nrfeldw = []

        loss_temp_test_nrfdw = []
        loss_temp_test_nrf = []
        loss_temp_test_nrfel = []
        loss_temp_test_nrfeldw = []


        for epoch in epochs:
            nrfdw = NeuralRandomForest(rf, 'NRF_analyticWeights_adam', X_train, y_train, output_func='softmax',
                                       cost_func='CrossEntropy',
                                       gamma_output=1.5, gamma=[1, 1])
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
                                     gamma_output=1.5, gamma=[1, 1])  # zde změna, gamma_output je 1
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

            nrf_el = NeuralRandomForest(rf, 'NRF_extraLayer_adam', X_train, y_train, output_func='softmax',
                                        cost_func='CrossEntropy',
                                        gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
            nrf_el.get_NRF_ensemble(epoch, 10, 0.01, 0.02)
            predictions_test_nrf_el = nrf_el.predict(X_test)
            predictions_test_loss_nrf_el = nrf_el.predict_averaging_loss(X_test)
            predictions_train_nrf_el = nrf_el.predict(X_train)
            predictions_train_loss_nrf_el = nrf_el.predict_averaging_loss(X_train)

            results_test_nrf_el = classification_report(y_test, predictions_test_nrf_el, output_dict=True)
            results_train_nrf_el = classification_report(y_train, predictions_train_nrf_el, output_dict=True)

            accuracy_temp_test_nrfel.append(results_test_nrf_el['accuracy'])
            accuracy_temp_train_nrfel.append(results_train_nrf_el['accuracy'])

            loss_temp_test_nrfel.append(log_loss(y_test, predictions_test_loss_nrf_el))
            loss_temp_train_nrfel.append(log_loss(y_train, predictions_train_loss_nrf_el))

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

        accuracy_train_nrfdw.append(accuracy_temp_train_nrfdw)
        accuracy_train_nrf.append(accuracy_temp_train_nrf)
        accuracy_train_nrfeldw.append(accuracy_temp_train_nrfeldw)
        accuracy_train_nrfel.append(accuracy_temp_train_nrfel)

        accuracy_test_nrfdw.append(accuracy_temp_test_nrfdw)
        accuracy_test_nrf.append(accuracy_temp_test_nrf)
        accuracy_test_nrfeldw.append(accuracy_temp_test_nrfeldw)
        accuracy_test_nrfel.append(accuracy_temp_test_nrfel)

        loss_train_nrfdw.append(loss_temp_train_nrfdw)
        loss_train_nrf.append(loss_temp_train_nrf)
        loss_train_nrfeldw.append(loss_temp_train_nrfeldw)
        loss_train_nrfel.append(loss_temp_train_nrfel)

        loss_test_nrfdw.append(loss_temp_test_nrfdw)
        loss_test_nrf.append(loss_temp_test_nrf)
        loss_test_nrfeldw.append(loss_temp_test_nrfeldw)
        loss_test_nrfel.append(loss_temp_test_nrfel)


    accuracy_train_nrfdw = list(np.mean(np.array(accuracy_train_nrfdw, dtype=np.float64), axis=0))
    accuracy_test_nrfdw = list(np.mean(np.array(accuracy_test_nrfdw, dtype=np.float64), axis=0))
    loss_train_nrfdw = list(np.mean(np.array(loss_train_nrfdw, dtype=np.float64), axis=0))
    loss_test_nrfdw = list(np.mean(np.array(loss_test_nrfdw, dtype=np.float64), axis=0))

    accuracy_train_nrf = list(np.mean(np.array(accuracy_train_nrf, dtype=np.float64), axis=0))
    accuracy_test_nrf = list(np.mean(np.array(accuracy_test_nrf, dtype=np.float64), axis=0))
    loss_train_nrf = list(np.mean(np.array(loss_train_nrf, dtype=np.float64), axis=0))
    loss_test_nrf = list(np.mean(np.array(loss_test_nrf, dtype=np.float64), axis=0))

    accuracy_train_nrfel = list(np.mean(np.array(accuracy_train_nrfel, dtype=np.float64), axis=0))
    accuracy_test_nrfel = list(np.mean(np.array(accuracy_test_nrfel, dtype=np.float64), axis=0))
    loss_train_nrfel = list(np.mean(np.array(loss_train_nrfel, dtype=np.float64), axis=0))
    loss_test_nrfel = list(np.mean(np.array(loss_test_nrfel, dtype=np.float64), axis=0))

    accuracy_train_nrfeldw = list(np.mean(np.array(accuracy_train_nrfeldw, dtype=np.float64), axis=0))
    accuracy_test_nrfeldw = list(np.mean(np.array(accuracy_test_nrfeldw, dtype=np.float64), axis=0))
    loss_train_nrfeldw = list(np.mean(np.array(loss_train_nrfeldw, dtype=np.float64), axis=0))
    loss_test_nrfeldw = list(np.mean(np.array(loss_test_nrfeldw, dtype=np.float64), axis=0))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs,accuracy_train_nrf,color = 'red',marker='o')
    ax.plot(epochs, accuracy_train_nrf,color = 'red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0,70])
    plt.title('NRF')
    plt.ylim([min(accuracy_train_nrf) - 0.05, max(accuracy_train_nrf) + 0.05])
    fig.savefig('NRF_epochs_acc_train.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, accuracy_test_nrf, color='red', marker='o')
    ax.plot(epochs, accuracy_test_nrf, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0, 70])
    plt.title('NRF')
    plt.ylim([min(accuracy_test_nrf) - 0.05, max(accuracy_test_nrf) + 0.05])
    fig.savefig('NRF_epochs_acc_test.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, loss_train_nrf, color='red', marker='o')
    ax.plot(epochs, loss_train_nrf, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, 70])
    plt.title('NRF')
    plt.ylim([min(loss_train_nrf) - 0.05, max(loss_train_nrf) + 0.05])
    fig.savefig('NRF_epochs_loss_train.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, loss_test_nrf, color='red', marker='o')
    ax.plot(epochs, loss_test_nrf, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, 70])
    plt.title('NRF')
    plt.ylim([min(loss_test_nrf) - 0.05, max(loss_test_nrf) + 0.05])
    fig.savefig('NRF_epochs_loss_test.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs,accuracy_train_nrfdw,color = 'red',marker='o')
    ax.plot(epochs, accuracy_train_nrfdw,color = 'red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0,70])
    plt.title('NRF_DW')
    plt.ylim([min(accuracy_train_nrfdw) - 0.05, max(accuracy_train_nrfdw) + 0.05])
    fig.savefig('NRF_DW_epochs_acc_train.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, accuracy_test_nrfdw, color='red', marker='o')
    ax.plot(epochs, accuracy_test_nrfdw, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0, 70])
    plt.title('NRF_DW')
    plt.ylim([min(accuracy_test_nrfdw) - 0.05, max(accuracy_test_nrfdw) + 0.05])
    fig.savefig('NRF_DW_epochs_acc_test.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, loss_train_nrfdw, color='red', marker='o')
    ax.plot(epochs, loss_train_nrfdw, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, 70])
    plt.title('NRF_DW')
    plt.ylim([min(loss_train_nrfdw) - 0.05, max(loss_train_nrfdw) + 0.05])
    fig.savefig('NRF_DW_epochs_loss_train.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, loss_test_nrfdw, color='red', marker='o')
    ax.plot(epochs, loss_test_nrfdw, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, 70])
    plt.title('NRF_DW')
    plt.ylim([min(loss_test_nrfdw) - 0.05, max(loss_test_nrfdw) + 0.05])
    fig.savefig('NRF_DW_epochs_loss_test.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs,accuracy_train_nrfel,color = 'red',marker='o')
    ax.plot(epochs, accuracy_train_nrfel,color = 'red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0,70])
    plt.title('NRF_EL')
    plt.ylim([min(accuracy_train_nrfel) - 0.05, max(accuracy_train_nrfel) + 0.05])
    fig.savefig('NRF_EL_epochs_acc_train.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, accuracy_test_nrfel, color='red', marker='o')
    ax.plot(epochs, accuracy_test_nrfel, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0, 70])
    plt.title('NRF_EL')
    plt.ylim([min(accuracy_test_nrfel) - 0.05, max(accuracy_test_nrfel) + 0.05])
    fig.savefig('NRF_EL_epochs_acc_test.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, loss_train_nrfel, color='red', marker='o')
    ax.plot(epochs, loss_train_nrfel, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, 70])
    plt.title('NRF_EL')
    plt.ylim([min(loss_train_nrfel) - 0.05, max(loss_train_nrfel) + 0.05])
    fig.savefig('NRF_EL_epochs_loss_train.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, loss_test_nrfel, color='red', marker='o')
    ax.plot(epochs, loss_test_nrfel, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, 70])
    plt.title('NRF_EL')
    plt.ylim([min(loss_test_nrfel) - 0.05, max(loss_test_nrfel) + 0.05])
    fig.savefig('NRF_EL_epochs_loss_test.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs,accuracy_train_nrfeldw,color = 'red',marker='o')
    ax.plot(epochs, accuracy_train_nrfeldw,color = 'red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0,70])
    plt.title('NRF_EL_DW')
    plt.ylim([min(accuracy_train_nrfeldw) - 0.05, max(accuracy_train_nrfeldw) + 0.05])
    fig.savefig('NRF_EL_DW_epochs_acc_train.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, accuracy_test_nrfeldw, color='red', marker='o')
    ax.plot(epochs, accuracy_test_nrfeldw, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0, 70])
    plt.title('NRF_EL_DW')
    plt.ylim([min(accuracy_test_nrfeldw) - 0.05, max(accuracy_test_nrfeldw) + 0.05])
    fig.savefig('NRF_EL_DW_epochs_acc_test.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, loss_train_nrfeldw, color='red', marker='o')
    ax.plot(epochs, loss_train_nrfeldw, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, 70])
    plt.title('NRF_EL_DW')
    plt.ylim([min(loss_train_nrfeldw) - 0.05, max(loss_train_nrfeldw) + 0.05])
    fig.savefig('NRF_EL_DW_epochs_loss_train.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs, loss_test_nrfeldw, color='red', marker='o')
    ax.plot(epochs, loss_test_nrfeldw, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, 70])
    plt.title('NRF_EL_DW')
    plt.ylim([min(loss_test_nrfeldw) - 0.05, max(loss_test_nrfeldw) + 0.05])
    fig.savefig('NRF_EL_DW_epochs_loss_test.png')

    '''

    # prepare neural random tree from @estimator

    from keras.models import Sequential
    from keras.layers import Dense
    from keras import optimizers
    import keras
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(df):
        X_train, X_test = df[train_index], df[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
        y_test_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
        for j, k in zip(y_train, range(len(y_train))):
            y_train_keras[int(k), int(j)] = 1
        for j, k in zip(y_test, range(len(y_test))):
            y_test_keras[int(k), int(j)] = 1
        y_train_keras = y_train_keras.astype('int64')
        y_test_keras = y_test_keras.astype('int64')
        y_train = y_train.astype('int64')
        y_test = y_test.astype('int64')


        mod = Sequential()
        mod.add(Dense(units=40, activation='relu', input_shape=(X_train.shape[1],)))
        mod.add(Dense(units=20, activation='relu'))
        mod.add(Dense(units=max(y_train) + 1,
                        activation='softmax'))  # přičítáme 1, protože předpokládáme, že první classa je 0
        sgd = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

        mod.compile(loss='categorical_crossentropy',
                      optimizer=sgd,  # 'sgd',
                      metrics=['accuracy'])
        mod.fit(X_train, y_train_keras, epochs=30)

        classes = np.argmax(mod.predict(X_test),axis=1)
        print(classification_report(y_test,classes))
'''