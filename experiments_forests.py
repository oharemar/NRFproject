from NeuralRandomForest import *
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

# LOAD DATASET (bank_marketing, cars, vehicle_silhouette,diabetes)
df,y = load_datasets('bank_marketing')

print(df.shape)

# RANDOM FOREST , we try few different combinations of its hyperparameters and compare with NRF and NN
rf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=6,max_features=17)


# split to test and train data
X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.33) # split to train and test dataset
y_train_keras = np.zeros((X_train.shape[0],max(y_train)+1))
y_test_keras = np.zeros((X_train.shape[0],max(y_train)+1))


for j,k in zip(y_train,range(len(y_train))):
    y_train_keras[k,j] = 1
for j,k in zip(y_test,range(len(y_test))):
    y_test_keras[k,j] = 1

# fit decision tree and print classification results
rf.fit(X_train,y_train)
predictions_DT = rf.predict(X_test)
print('RANDOM FOREST')
print(classification_report(y_test,predictions_DT))


# prepare neural random tree from @estimator

#### ordinary NEURAL NETWORK ####


from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


model = Sequential()
model.add(Dense(units=20, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=max(y_train)+1, activation = 'softmax')) # přičítáme 1, protože předpokládáme, že první classa je 0
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer= sgd,#'sgd',
              metrics=['accuracy'])
model.fit(X_train, y_train_keras, epochs=30)
classes = np.argmax(model.predict(X_test),axis=1)

print('NEURAL NETWORK')
print(classification_report(y_test,classes))

print('NRF BASIC')
nrf_basic = NeuralRandomForest(rf,'NRF_basic',X_train,y_train,'softmax',cost_func='LogLikelihood',gamma_output=1.5,gamma=[1.5,1.5])
nrf_basic.get_NRF_ensemble(30,10,0.02,0.02)
predictions_NRT_basic = nrf_basic.predict(X_test)
print(classification_report(y_test,predictions_NRT_basic))

print('NRF BASIC NESTEROV')
nrf_basic_nesterov = NeuralRandomForest(rf,'NRF_basic_nesterov',X_train,y_train,'softmax',cost_func='LogLikelihood',gamma_output=1.5,gamma=[1.5,1.5])
nrf_basic_nesterov.get_NRF_ensemble(30,10,0.02,0.02)
predictions_NRT_basic_nesterov = nrf_basic_nesterov.predict(X_test)
print(classification_report(y_test,predictions_NRT_basic_nesterov))

print('NRF BASIC ADAM')
nrf_basic_adam = NeuralRandomForest(rf,'NRF_basic_adam',X_train,y_train,'softmax',cost_func='LogLikelihood',gamma_output=1.5,gamma=[1.5,1.5])
nrf_basic_adam.get_NRF_ensemble(30,10,0.02,0.02)
predictions_NRT_basic_adam = nrf_basic_adam.predict(X_test)
print(classification_report(y_test,predictions_NRT_basic_adam))


print('NRF_analyticWeights')
nrf_analytic_weights= NeuralRandomForest(rf,'NRF_analyticWeights',X_train,y_train,'softmax',cost_func='LogLikelihood',gamma_output=1.5,gamma=[2.3,2.3])
nrf_analytic_weights.get_NRF_ensemble(30,10,0.0065,0.02)
predictions_NRT_basic_aw = nrf_analytic_weights.predict(X_test)
print(classification_report(y_test,predictions_NRT_basic_aw))

print('NRF_analyticWeights adam')
nrf_analytic_weights= NeuralRandomForest(rf,'NRF_analyticWeights_adam',X_train,y_train,'softmax',cost_func='LogLikelihood',gamma_output=1.5,gamma=[2.3,2.3])
nrf_analytic_weights.get_NRF_ensemble(15,10,0.0035,0.02)
predictions_NRT_basic_aw = nrf_analytic_weights.predict(X_test)
print(classification_report(y_test,predictions_NRT_basic_aw))

print('NRF_analyticWeights nesterov')
nrf_analytic_weights= NeuralRandomForest(rf,'NRF_analyticWeights_nesterov',X_train,y_train,'softmax',cost_func='LogLikelihood',gamma_output=1.5,gamma=[2.3,2.3])
nrf_analytic_weights.get_NRF_ensemble(15,10,0.0035,0.02)
predictions_NRT_basic_aw = nrf_analytic_weights.predict(X_test)
print(classification_report(y_test,predictions_NRT_basic_aw))

print('NRF_extraLayer')
nrf_el= NeuralRandomForest(rf,'NRF_extraLayer',X_train,y_train,'softmax',cost_func='LogLikelihood',gamma_output=1,gamma=[1.3,1.3])
nrf_el.get_NRF_ensemble(30,10,0.035,0.02)
predictions_NRT_el = nrf_el.predict(X_test)
print(classification_report(y_test,predictions_NRT_el))

print('NRF_extraLayer ANALYTIC WEIGHTS')
nrf_el= NeuralRandomForest(rf,'NRF_extraLayer_analyticWeights',X_train,y_train,'softmax',cost_func='LogLikelihood',gamma_output=1,gamma=[1.3,1.3])
nrf_el.get_NRF_ensemble(30,10,0.015,0.02,eta2=0.15)
predictions_NRT_el = nrf_el.predict(X_test)
print(classification_report(y_test,predictions_NRT_el))
