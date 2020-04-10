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
df,y = load_datasets('OBSnetwork')

# prepare DECISION TREE CLASSIFIER
estimator = DecisionTreeClassifier(max_depth=6, random_state=0) # beware of random_state, use only if necessary to repeat experiment with same trees

# split to test and train data
X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.33) # split to train and test dataset
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

# prepare neural random tree from @estimator

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers



'''
from NRF_basic_boosted_adam import *
nrf_basic_adaptive = NeuralTreeBasic_boosted_adam(estimator,X_train,y_train,output_func='sigmoid',gamma_output=1.5,gamma = [1.5,1.5])
evaluation_cost_basic, evaluation_accuracy_basic, training_cost_basic, training_accuracy_basic = nrf_basic_adaptive.train_NRF(50,100,0.05,0.02,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_basic_ad = nrf_basic_adaptive.predict(X_test)
print(classification_report(y_test,predictions_NRT_basic_ad))

'''

from NRF_analyticWeights import *
nrf_analyticsWeights = NeuralTree_analyticWeights(estimator,X_train,y_train,output_func='softmax',gamma_output=1.5,gamma = [2.3,2.3]) # čím vyšší gamma_output a gamma, tím lépe aproximuje nrf původní decision tree, zkusit ty hodnoty zvýšit a zároveň snížit learning rate
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = nrf_analyticsWeights.train_NRF(30,10,0.0065,0.03,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_analytic = np.argmax(nrf_analyticsWeights.predict_prob(X_test),axis=1)
#predictions_NRT_analytic = nrf_analyticsWeights.predict(X_test)
print(classification_report(y_test,predictions_NRT_analytic))

#predictions_NRT_EL_AN = nrf_EL_analyticsWeights.predict(X_test)
#print(classification_report(y_test,predictions_NRT_EL_AN))


#predictions_NRT_EL_AN = nrf_EL_analyticsWeights.predict(X_test)
#print(classification_report(y_test,predictions_NRT_EL_AN))
#### ordinary NEURAL NETWORK ####
'''

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


model = Sequential()
model.add(Dense(units=20, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=max(y_train)+1, activation = 'softmax')) # přičítáme 1, protože předpokládáme, že první classa je 0
sgd = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='categorical_crossentropy',
              optimizer= sgd,#'sgd',
              metrics=['accuracy'])
model.fit(X_train, y_train_keras, epochs=30)
classes = np.argmax(model.predict(X_test),axis=1)

print(classification_report(y_test,classes))


'''
'''NRT Extra Layer analytic weights adam'''
'''
from NRF_withExtraLayer_analyticWeights_adam import *
nrf_EL_analyticsWeights = NeuralTree_extraLayer_analyticWeights(estimator,X_train,y_train,output_func='sigmoid',
                                                                gamma = [1.3,1.3],gamma_sigmoid=2,gamma_output=1)
evaluation_cost_EL_AN, evaluation_accuracy_EL_AN, training_cost_EL_AN, training_accuracy_EL_AN = nrf_EL_analyticsWeights.train_NRF(30,10,0.01,0.02,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_EL_AN = nrf_EL_analyticsWeights.predict(X_test)
print(classification_report(y_test,predictions_NRT_EL_AN))
'''

'''
NRT extra layer analytic weights _classic'''
'''
from NRF_withExtraLayer_analyticWeights import *
nrf_EL_analyticsWeights = NeuralTree_extraLayer_analyticWeights(estimator,X_train,y_train,output_func='sigmoid',
                                                                gamma = [1.3,1.3],gamma_sigmoid=2,gamma_output=1)
evaluation_cost_EL_AN, evaluation_accuracy_EL_AN, training_cost_EL_AN, training_accuracy_EL_AN = nrf_EL_analyticsWeights.train_NRF(30,10,0.1,0.02,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_EL_AN = nrf_EL_analyticsWeights.predict(X_test)
print(classification_report(y_test,predictions_NRT_EL_AN))

'''
'''
NRT extra layer analytic weights nesterov
from NRF_withExtraLayer_analyticWeights_nesterov import *
nrf_EL_analyticsWeights = NeuralTree_extraLayer_analyticWeights(estimator,X_train,y_train,output_func='sigmoid',
                                                                gamma = [1.3,1.3],gamma_sigmoid=2,gamma_output=1)
evaluation_cost_EL_AN, evaluation_accuracy_EL_AN, training_cost_EL_AN, training_accuracy_EL_AN = nrf_EL_analyticsWeights.train_NRF(50,10,0.01,0.02,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_EL_AN = nrf_EL_analyticsWeights.predict(X_test)
print(classification_report(y_test,predictions_NRT_EL_AN))
'''
'''
#### NRF BASIC ####
from NRF_basic_boosted import *
nrf_basic = NeuralTreeBasic_boosted(estimator,X_train,y_train,output_func='softmax',gamma_output=1.5,gamma = [1.5,1.5])
evaluation_cost_basic, evaluation_accuracy_basic, training_cost_basic, training_accuracy_basic = nrf_basic.train_NRF(30,10,0.025,0.03,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_basic = nrf_basic.predict(X_test)
####
'''
'''
from NRF_analyticWeights_nesterov import *
nrf_analyticsWeights = NeuralTree_analyticWeights(estimator,X_train,y_train,output_func='softmax',gamma_output=1.5,gamma = [2.3,2.3]) # čím vyšší gamma_output a gamma, tím lépe aproximuje nrf původní decision tree, zkusit ty hodnoty zvýšit a zároveň snížit learning rate
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = nrf_analyticsWeights.train_NRF(15,10,0.0035,0.0,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_analytic = nrf_analyticsWeights.predict(X_test)
print(classification_report(y_test,predictions_NRT_analytic))
'''
'''
from NRF_analyticWeights_adam import *
nrf_analyticsWeights = NeuralTree_analyticWeights(estimator,X_train,y_train,output_func='softmax',gamma_output=1.5,gamma = [2.3,2.3]) # čím vyšší gamma_output a gamma, tím lépe aproximuje nrf původní decision tree, zkusit ty hodnoty zvýšit a zároveň snížit learning rate
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = nrf_analyticsWeights.train_NRF(15,10,0.0035,0.0,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_analytic = nrf_analyticsWeights.predict(X_test)
print(classification_report(y_test,predictions_NRT_analytic))
'''
'''
from NRF_basic_boosted_nesterov import *
nrf_basic_adaptive = NeuralTreeBasic_boosted_nesterov(estimator,X_train,y_train,output_func='sigmoid',gamma_output=1.5,gamma = [1.5,1.5])
evaluation_cost_basic, evaluation_accuracy_basic, training_cost_basic, training_accuracy_basic = nrf_basic_adaptive.train_NRF(30,10,0.002,0.0,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_basic_ad = nrf_basic_adaptive.predict(X_test)
print(classification_report(y_test,predictions_NRT_basic_ad))
# plot results

from NRF_basic_boosted_adam import *
nrf_basic_adaptive = NeuralTreeBasic_boosted_adam(estimator,X_train,y_train,output_func='sigmoid',gamma_output=1.5,gamma = [1.5,1.5])
evaluation_cost_basic, evaluation_accuracy_basic, training_cost_basic, training_accuracy_basic = nrf_basic_adaptive.train_NRF(30,10,0.002,0.0,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_basic_ad = nrf_basic_adaptive.predict(X_test)
print(classification_report(y_test,predictions_NRT_basic_ad))


print(classification_report(y_test,predictions_NRT_basic))

#### NRF CLASSIC ####
from NRF_boosted import *
nrf_classic = NeuralTreeBoosted(estimator,X_train,y_train,output_func='sigmoid',gamma_output=1.2,gamma = [2.3,2.3])
evaluation_cost_classic, evaluation_accuracy_classic, training_cost_classic, training_accuracy_classic = nrf_classic.train_NRF(30,10,0.012,0.03,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_classic = nrf_classic.predict(X_test)
print(classification_report(y_test,predictions_NRT_classic))
####

#### NRF ANALYTIC WEIGHTS ####
from NRF_analyticWeights import *
nrf_analyticsWeights = NeuralTree_analyticWeights(estimator,X_train,y_train,output_func='softmax',gamma_output=1.5,gamma = [2.3,2.3]) # čím vyšší gamma_output a gamma, tím lépe aproximuje nrf původní decision tree, zkusit ty hodnoty zvýšit a zároveň snížit learning rate
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = nrf_analyticsWeights.train_NRF(20,10,0.0065,0.03,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_analytic = nrf_analyticsWeights.predict(X_test)
print(classification_report(y_test,predictions_NRT_analytic))
####

#### NRF EXTRA LAYER CLASSIC #### v extra layers je předposlední vrstva sigmoid, to by se dalo změnit!!!! prozkoumat
from NRF_withExtraLayer import *
nrf_EL= NeuralTree_extraLayer(estimator,X_train,y_train,output_func='sigmoid',gamma = [1.3,1.3],gamma_sigmoid=1,gamma_output=1)
evaluation_cost_extra_layer, evaluation_accuracy_extra_layer, training_cost_extra_layer, training_accuracy_extra_layer = nrf_EL.train_NRF(30,10,0.035,0.03,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_EL = nrf_EL.predict(X_test)
print(classification_report(y_test,predictions_NRT_EL))
####


### NRF EXTRA LAYER ANALYTIC WEIGHTS ####
from NRF_withExtraLayer_analyticWeights import *
nrf_EL_analyticsWeights = NeuralTree_extraLayer_analyticWeights(estimator,X_train,y_train,output_func='sigmoid',
                                                                gamma = [2.3,2.3],gamma_sigmoid=2,gamma_output=1,
                                                                learning_rate1=0.015,learning_rate2=0.15)
evaluation_cost_EL_AN, evaluation_accuracy_EL_AN, training_cost_EL_AN, training_accuracy_EL_AN = nrf_EL_analyticsWeights.train_NRF(30,10,0.03,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT_EL_AN = nrf_EL_analyticsWeights.predict(X_test)
print(classification_report(y_test,predictions_NRT_EL_AN))
####


'''



