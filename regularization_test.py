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

# LOAD DATASET (bank_marketing, cars, vehicle_silhouette,diabetes,messidor)

if __name__ == '__main__':

    df,y = load_datasets('messidor')

    # prepare DECISION TREE CLASSIFIER
    #estimator = RandomForestClassifier(n_estimators=10,max_depth=6, max_features='auto',random_state=0) # beware of random_state, use only if necessary to repeat experiment with same trees

    # split to test and train data
    '''
    X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.2) # split to train and test dataset
    y_train_keras = np.zeros((X_train.shape[0],int(max(y_train)+1)))
    y_test_keras = np.zeros((X_train.shape[0],int(max(y_train)+1)))
    
    y_train_keras = y_train_keras.astype('int64')
    y_test_keras = y_test_keras.astype('int64')
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')
    
    
    
    for j,k in zip(y_train,range(len(y_train))):
        y_train_keras[int(k),int(j)] = 1
    for j,k in zip(y_test,range(len(y_test))):
        y_test_keras[int(k),int(j)] = 1
    
    # fit decision tree and print classification results
    estimator.fit(X_train,y_train)
    predictions_DT = estimator.predict(X_test)
    print(classification_report(y_test,predictions_DT))
    '''
    learn_rates_nrfdw = list(np.linspace(0,0.07,num=8))
    learn_rates_nrf = list(np.linspace(0,0.07,num=8))
    learn_rates_nrfel = list(np.linspace(0,0.07,num=8))
    learn_rates_nrfeldw = list(np.linspace(0,0.07,num=8))

    accuracy_nrfdw = []
    accuracy_nrf = []
    accuracy_nrfel = []
    accuracy_nrfeldw = []

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

        accuracy_temp_nrfdw = []
        accuracy_temp_nrf = []
        accuracy_temp_nrfel = []
        accuracy_temp_nrfeldw = []

        f_temp_nrfdw = []
        f_temp_nrf = []
        f_temp_nrfel = []
        f_temp_nrfeldw = []

        for eta,eta_nrfdw,eta_nrfel,eta_nrfeldw in zip(learn_rates_nrf,learn_rates_nrfdw,learn_rates_nrfel,learn_rates_nrfeldw):
            nrfdw = NeuralRandomForest(rf, 'NRF_analyticWeights_adam', X_train, y_train, output_func='softmax',
                                     cost_func='CrossEntropy',
                                     gamma_output=1.5, gamma=[2.3, 2.3])  # zde změna, gamma_output je 1
            nrfdw.get_NRF_ensemble(30, 10, 0.0035, eta)
            predictions_nrfdw = nrfdw.predict(X_test)
            results_nrfdw = classification_report(y_test, predictions_nrfdw, output_dict=True)
            accuracy_temp_nrfdw.append(results_nrfdw['accuracy'])
            f_temp_nrfdw.append(results_nrfdw['macro avg']['f1-score'])

            nrf = NeuralRandomForest(rf, 'NRF_basic_adam', X_train, y_train, output_func='softmax',
                                       cost_func='CrossEntropy',
                                       gamma_output=1.5, gamma=[1.5, 1.5])  # zde změna, gamma_output je 1
            nrf.get_NRF_ensemble(30, 10, 0.002, eta)
            predictions_nrf = nrf.predict(X_test)
            results_nrf = classification_report(y_test, predictions_nrf, output_dict=True)
            accuracy_temp_nrf.append(results_nrf['accuracy'])
            f_temp_nrf.append(results_nrf['macro avg']['f1-score'])

            nrf_el = NeuralRandomForest(rf, 'NRF_extraLayer_adam', X_train, y_train, output_func='softmax',
                                     cost_func='CrossEntropy',
                                     gamma_output=1, gamma=[1.3, 1.3])  # zde změna, gamma_output je 1
            nrf_el.get_NRF_ensemble(30, 10, 0.01, eta)
            predictions_nrf_el = nrf_el.predict(X_test)
            results_nrf_el = classification_report(y_test, predictions_nrf_el, output_dict=True)
            accuracy_temp_nrfel.append(results_nrf_el['accuracy'])
            f_temp_nrfel.append(results_nrf_el['macro avg']['f1-score'])

            nrf_eldw = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_adam', X_train, y_train, output_func='softmax',
                                        cost_func='CrossEntropy',
                                        gamma_output=1, gamma=[1.3, 1.3])  # zde změna, gamma_output je 1
            nrf_eldw.get_NRF_ensemble(30, 10, 0.005, eta)
            predictions_nrf_eldw = nrf_eldw.predict(X_test)
            results_nrf_eldw = classification_report(y_test, predictions_nrf_eldw, output_dict=True)
            accuracy_temp_nrfeldw.append(results_nrf_eldw['accuracy'])
            f_temp_nrfeldw.append(results_nrf_eldw['macro avg']['f1-score'])



        accuracy_nrfdw.append(accuracy_temp_nrfdw)
        accuracy_nrf.append(accuracy_temp_nrf)
        accuracy_nrfeldw.append(accuracy_temp_nrfeldw)
        accuracy_nrfel.append(accuracy_temp_nrfel)
        f_nrfdw.append(f_temp_nrfdw)
        f_nrf.append(f_temp_nrf)
        f_nrfeldw.append(f_temp_nrfeldw)
        f_nrfel.append(f_temp_nrfel)




    accuracy_nrfdw = np.array(accuracy_nrfdw,dtype=np.float64)
    accuracy_nrfdw = list(np.mean(accuracy_nrfdw,axis=0))
    accuracy_nrf = np.array(accuracy_nrf,dtype=np.float64)
    accuracy_nrf = list(np.mean(accuracy_nrf,axis=0))
    accuracy_nrfeldw = np.array(accuracy_nrfeldw,dtype=np.float64)
    accuracy_nrfeldw = list(np.mean(accuracy_nrfeldw,axis=0))
    accuracy_nrfel = np.array(accuracy_nrfel,dtype=np.float64)
    accuracy_nrfel = list(np.mean(accuracy_nrfel,axis=0))
    f_nrfdw = np.array(f_nrfdw,dtype=np.float64)
    f_nrfdw = list(np.mean(f_nrfdw,axis=0))
    f_nrf = np.array(f_nrf,dtype=np.float64)
    f_nrf = list(np.mean(f_nrf,axis=0))
    f_nrfeldw = np.array(f_nrfeldw,dtype=np.float64)
    f_nrfeldw = list(np.mean(f_nrfeldw,axis=0))
    f_nrfel = np.array(f_nrfel,dtype=np.float64)
    f_nrfel = list(np.mean(f_nrfel,axis=0))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(learn_rates_nrf,accuracy_nrf,width = 0.8*(learn_rates_nrf[1]-learn_rates_nrf[0]))
    plt.xlabel('Regularization constant')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrf)-(learn_rates_nrf[1]-learn_rates_nrf[0]),max(learn_rates_nrf)+(learn_rates_nrf[1]-learn_rates_nrf[0])])
    plt.title('NRF')
    plt.ylim([min(accuracy_nrf)-0.05,max(accuracy_nrf) + 0.05])
    fig.savefig('NRF_regularization_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(learn_rates_nrf,f_nrf,width = 0.8*(learn_rates_nrf[1]-learn_rates_nrf[0]))
    plt.xlabel('Regularization constant')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrf)-(learn_rates_nrf[1]-learn_rates_nrf[0]),max(learn_rates_nrf)+(learn_rates_nrf[1]-learn_rates_nrf[0])])
    plt.title('NRF')
    plt.ylim([min(f_nrf)-0.05,max(f_nrf) + 0.05])
    fig.savefig('NRF_regularization_F1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(learn_rates_nrfdw,accuracy_nrfdw,width = 0.8*(learn_rates_nrfdw[1]-learn_rates_nrfdw[0]))
    plt.xlabel('Regularization constant')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfdw)-(learn_rates_nrfdw[1]-learn_rates_nrfdw[0]),max(learn_rates_nrfdw)+(learn_rates_nrfdw[1]-learn_rates_nrfdw[0])])
    plt.title('NRF_DW')
    plt.ylim([min(accuracy_nrfdw)-0.05,max(accuracy_nrfdw) + 0.05])
    fig.savefig('NRF_DW_regularization_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(learn_rates_nrfdw,f_nrfdw,width = 0.8*(learn_rates_nrfdw[1]-learn_rates_nrfdw[0]))
    plt.xlabel('Regularization constant')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrfdw)-(learn_rates_nrfdw[1]-learn_rates_nrfdw[0]),max(learn_rates_nrfdw)+(learn_rates_nrfdw[1]-learn_rates_nrfdw[0])])
    plt.title('NRF_DW')
    plt.ylim([min(f_nrfdw)-0.05,max(f_nrfdw) + 0.05])
    fig.savefig('NRF_DW_regularization_F1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(learn_rates_nrfel,accuracy_nrfel,width = 0.8*(learn_rates_nrfel[1]-learn_rates_nrfel[0]))
    plt.xlabel('Regularization constant')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfel)-(learn_rates_nrfel[1]-learn_rates_nrfel[0]),max(learn_rates_nrfel)+(learn_rates_nrfel[1]-learn_rates_nrfel[0])])
    plt.title('NRF_EL')
    plt.ylim([min(accuracy_nrfel)-0.05,max(accuracy_nrfel) + 0.05])
    fig.savefig('NRF_EL_regularization_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(learn_rates_nrfel,f_nrfel,width = 0.8*(learn_rates_nrfel[1]-learn_rates_nrfel[0]))
    plt.xlabel('Regularization constant')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrfel)-(learn_rates_nrfel[1]-learn_rates_nrfel[0]),max(learn_rates_nrfel)+(learn_rates_nrfel[1]-learn_rates_nrfel[0])])
    plt.title('NRF_EL')
    plt.ylim([min(f_nrfel)-0.05,max(f_nrfel) + 0.05])
    fig.savefig('NRF_EL_regularization_F1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(learn_rates_nrfeldw,accuracy_nrfeldw,width = 0.8*(learn_rates_nrfeldw[1]-learn_rates_nrfeldw[0]))
    plt.xlabel('Regularization constant')
    plt.ylabel('Accuracy')
    plt.xlim([min(learn_rates_nrfeldw)-(learn_rates_nrfeldw[1]-learn_rates_nrfeldw[0]),max(learn_rates_nrfeldw)+(learn_rates_nrfeldw[1]-learn_rates_nrfeldw[0])])
    plt.title('NRF_EL_DW')
    plt.ylim([min(accuracy_nrfeldw)-0.05,max(accuracy_nrfeldw) + 0.05])
    fig.savefig('NRF_EL_DW_regularization_acc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(learn_rates_nrfeldw,f_nrfeldw,width = 0.8*(learn_rates_nrfeldw[1]-learn_rates_nrfeldw[0]))
    plt.xlabel('Regularization constant')
    plt.ylabel('F1-score')
    plt.xlim([min(learn_rates_nrfeldw)-(learn_rates_nrfeldw[1]-learn_rates_nrfeldw[0]),max(learn_rates_nrfeldw)+(learn_rates_nrfeldw[1]-learn_rates_nrfeldw[0])])
    plt.title('NRF_EL_DW')
    plt.ylim([min(f_nrfeldw)-0.05,max(f_nrfeldw) + 0.05])
    fig.savefig('NRF_EL_DW_regularization_F1.png')

