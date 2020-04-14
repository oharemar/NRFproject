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


if __name__ == '__main__':

    df, y = load_datasets('vehicle_silhouette')

    rf = RandomForestClassifier(n_estimators=4,criterion='entropy',max_depth=6,max_features='auto')

    epochs = [5,10,15,20,25,30]

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)  # split to train and test dataset

    rf.fit(X_train,y_train)

    nrf = NeuralRandomForest(rf, 'NRF_analyticWeights_adam', X_train, y_train, output_func='softmax',
                                       cost_func='CrossEntropy',
                                       gamma_output=1.5, gamma=[1, 1])

    nrf.get_NRF_ensemble(40, 10, 0.0035, 0.02)

    preds = nrf.predict_averaging_loss(X_train)

    print(log_loss(y_train,preds))




