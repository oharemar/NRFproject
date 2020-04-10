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

# LOAD DATASET (bank_marketing, cars, vehicle_silhouette,diabetes,messidor)
df,y = load_datasets('vehicle_silhouette')

'''
# prepare DECISION TREE CLASSIFIER
estimator = DecisionTreeClassifier(max_depth=6, random_state=0) # beware of random_state, use only if necessary to repeat experiment with same trees

# split to test and train data
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
print('DECISION TREE')
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
