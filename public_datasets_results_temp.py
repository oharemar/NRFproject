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
from helpful_functions import *
from sklearn.model_selection import KFold
from statistics import mean
from statistics import stdev
import json
from NeuralRandomForest import NeuralRandomForest

datasets = ['USPS','USPS_no_scale'] # pak přidáme USPS


# do modelů pak přidat logistic regression, RF a neuronku z kerasu
model_names = ['NRF_analyticWeights','NRF_analyticWeights_adam','NRF_analyticWeights_nesterov',
               'NRF_extraLayer_analyticWeights','NRF_extraLayer_analyticWeights_adam','NRF_extraLayer_analyticWeights_nesterov']



learn_rates = [0.0065,0.0035,0.0035,0.15,0.005,0.01]


for dataset in datasets:
    print(dataset)
    for model,eta in zip(model_names,learn_rates):
        if dataset == 'USPS' and model in ['NRF_analyticWeights','NRF_analyticWeights_adam','NRF_analyticWeights_nesterov']:
            continue
        print(model)
        df, y = load_datasets(dataset)

        # prepare RF

        rf = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=6, max_features='auto')

        kf = KFold(n_splits=5)

        final_results_vals = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                              'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
        final_results_stds = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                              'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}

        results_all = []
        results_all_avg = []

        for train_index, test_index in kf.split(df):
            X_train, X_test = df[train_index], df[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #y_train_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
            #y_test_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
            #y_train_keras = y_train_keras.astype('int64')
            #y_test_keras = y_test_keras.astype('int64')
            y_train = y_train.astype('int64')
            y_test = y_test.astype('int64')
            rf.fit(X_train, y_train)  # zde trénink random forestu
            if model in ['NRF_extraLayer','NRF_extraLayer_adam','NRF_extraLayer_nesterov',
               'NRF_extraLayer_analyticWeights','NRF_extraLayer_analyticWeights_adam','NRF_extraLayer_analyticWeights_nesterov']:
                nrf = NeuralRandomForest(rf,model,X_train,y_train,output_func='softmax',cost_func='CrossEntropy',
                                         gamma_output=1,gamma=[1.3,1.3])
                nrf.get_NRF_ensemble(30,10,eta,0.02)
            elif model in ['NRF_analyticWeights','NRF_analyticWeights_adam','NRF_analyticWeights_nesterov']:
                nrf = NeuralRandomForest(rf, model, X_train, y_train, output_func='sigmoid', cost_func='CrossEntropy',# zde musíme sigmoid kvůli inverzi
                                         gamma_output=1.5, gamma=[2.3, 2.3])
                nrf.get_NRF_ensemble(30,10,eta,0.02)
            else:
                nrf = NeuralRandomForest(rf, model, X_train, y_train, output_func='softmax', cost_func='CrossEntropy',
                                         gamma_output=1.5, gamma=[1.5, 1.5])
                nrf.get_NRF_ensemble(30,10,eta,0.02)


            predictions = nrf.predict(X_test)
            predictions_avg = nrf.predict_averaging(X_test)

            results_temp = classification_report(y_test, predictions, output_dict=True)
            results_temp_avg = classification_report(y_test, predictions_avg, output_dict=True)
            results_all.append(results_temp)
            results_all_avg.append(results_temp_avg)

        final_results_vals['accuracy'] = mean([res['accuracy'] for res in results_all])
        final_results_stds['accuracy'] = stdev([res['accuracy'] for res in results_all])
        final_results_vals['weighted avg']['precision'] = mean([res['weighted avg']['precision'] for res in results_all])
        final_results_stds['weighted avg']['precision'] = stdev([res['weighted avg']['precision'] for res in results_all])
        final_results_vals['weighted avg']['recall'] = mean([res['weighted avg']['recall'] for res in results_all])
        final_results_stds['weighted avg']['recall'] = stdev([res['weighted avg']['recall'] for res in results_all])
        final_results_vals['weighted avg']['support'] = mean([res['weighted avg']['support'] for res in results_all])
        final_results_stds['weighted avg']['support'] = stdev([res['weighted avg']['support'] for res in results_all])
        final_results_vals['weighted avg']['f1-score'] = mean([res['weighted avg']['f1-score'] for res in results_all])
        final_results_stds['weighted avg']['f1-score'] = stdev([res['weighted avg']['f1-score'] for res in results_all])
        final_results_vals['macro avg']['precision'] = mean([res['macro avg']['precision'] for res in results_all])
        final_results_stds['macro avg']['precision'] = stdev([res['macro avg']['precision'] for res in results_all])
        final_results_vals['macro avg']['recall'] = mean([res['macro avg']['recall'] for res in results_all])
        final_results_stds['macro avg']['recall'] = stdev([res['macro avg']['recall'] for res in results_all])
        final_results_vals['macro avg']['support'] = mean([res['macro avg']['support'] for res in results_all])
        final_results_stds['macro avg']['support'] = stdev([res['macro avg']['support'] for res in results_all])
        final_results_vals['macro avg']['f1-score'] = mean([res['macro avg']['f1-score'] for res in results_all])
        final_results_stds['macro avg']['f1-score'] = stdev([res['macro avg']['f1-score'] for res in results_all])

        with open('RESULTS_PUBLIC_DATASETS/{}/{}_mean.txt'.format(model,dataset), 'w') as file:
            file.write(json.dumps(final_results_vals))
        with open('RESULTS_PUBLIC_DATASETS/{}/{}_std.txt'.format(model, dataset), 'w') as file:
            file.write(json.dumps(final_results_stds))

        final_results_vals['accuracy'] = mean([res['accuracy'] for res in results_all_avg])
        final_results_stds['accuracy'] = stdev([res['accuracy'] for res in results_all_avg])
        final_results_vals['weighted avg']['precision'] = mean([res['weighted avg']['precision'] for res in results_all_avg])
        final_results_stds['weighted avg']['precision'] = stdev([res['weighted avg']['precision'] for res in results_all_avg])
        final_results_vals['weighted avg']['recall'] = mean([res['weighted avg']['recall'] for res in results_all_avg])
        final_results_stds['weighted avg']['recall'] = stdev([res['weighted avg']['recall'] for res in results_all_avg])
        final_results_vals['weighted avg']['support'] = mean([res['weighted avg']['support'] for res in results_all_avg])
        final_results_stds['weighted avg']['support'] = stdev([res['weighted avg']['support'] for res in results_all_avg])
        final_results_vals['weighted avg']['f1-score'] = mean([res['weighted avg']['f1-score'] for res in results_all_avg])
        final_results_stds['weighted avg']['f1-score'] = stdev([res['weighted avg']['f1-score'] for res in results_all_avg])
        final_results_vals['macro avg']['precision'] = mean([res['macro avg']['precision'] for res in results_all_avg])
        final_results_stds['macro avg']['precision'] = stdev([res['macro avg']['precision'] for res in results_all_avg])
        final_results_vals['macro avg']['recall'] = mean([res['macro avg']['recall'] for res in results_all_avg])
        final_results_stds['macro avg']['recall'] = stdev([res['macro avg']['recall'] for res in results_all_avg])
        final_results_vals['macro avg']['support'] = mean([res['macro avg']['support'] for res in results_all_avg])
        final_results_stds['macro avg']['support'] = stdev([res['macro avg']['support'] for res in results_all_avg])
        final_results_vals['macro avg']['f1-score'] = mean([res['macro avg']['f1-score'] for res in results_all_avg])
        final_results_stds['macro avg']['f1-score'] = stdev([res['macro avg']['f1-score'] for res in results_all_avg])

        with open('RESULTS_PUBLIC_DATASETS_averaging/{}/{}_mean.txt'.format(model,dataset), 'w') as file:
            file.write(json.dumps(final_results_vals))
        with open('RESULTS_PUBLIC_DATASETS_averaging/{}/{}_std.txt'.format(model, dataset), 'w') as file:
            file.write(json.dumps(final_results_stds))
