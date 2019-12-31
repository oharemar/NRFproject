from NRF import *
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


rf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=6,max_features=8)

'''NRF_basic test'''
'''
le = LabelEncoder() # encoder
#pd.set_option("display.max_columns", 999)
df = pd.read_csv('bank_marketing_dataset.csv')
df = df.sample(frac=1)
y = df.loc[:,'y']
df = df.loc[:,df.columns != 'y']
y = le.fit_transform(y)
#print(df.describe(include = object)) #statistics of dataset (omit @param:include to include numerical variables)
df.drop(columns = ['default'],inplace = True)
df = pd.get_dummies(df) # one hot encoding categorical variables
scaler = StandardScaler()
df = scaler.fit_transform(df)
'''
'''
df = pd.read_csv("cars_dataset.csv")
df = df.sample(frac=1)
y = df.loc[:,'car']
df = df.loc[:,df.columns != 'car']
le = LabelEncoder() # encoder
y = le.fit_transform(y)
df = pd.get_dummies(df) # one hot encoding categorical variables
scaler = StandardScaler()
df = scaler.fit_transform(df)
'''

df = pd.read_csv("vehicle_silhouette_dataset.csv")
df = df.sample(frac=1)
y = df.loc[:,'vehicle_class']
df = df.loc[:,df.columns != 'vehicle_class']
le = LabelEncoder()
y = le.fit_transform(y)
scaler = StandardScaler()
df = scaler.fit_transform(df)

'''
df = pd.read_csv("diabetes.csv")
df = df.sample(frac=1)
y = df.loc[:,'Outcome']
df = df.loc[:,df.columns != 'Outcome']
scaler = StandardScaler()
df = scaler.fit_transform(df)
'''
df = np.array(df,dtype=np.float64)
y = np.array(y)

estimator = DecisionTreeClassifier(max_leaf_nodes=6, random_state=0)

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.33) # split to train and test dataset

estimator.fit(X_train,y_train)
predictions_DT = estimator.predict(X_test)

print(classification_report(y_test,predictions_DT))

#from NRF_withExtraLayerCombined import *
#nrf2 = NeuralTree_withExtraLayerCombined(estimator,X_train,y_train)
'''NRF_basic
from NRF_basic_boosted import *
nrf_basic = NeuralTreeBasic_boosted(estimator,X_train,y_train,output_func='sigmoid',gamma_output=1.5,gamma = [3.5,3.5])
'''

'''NRF_classic
from NRF_boosted import *
nrf_classic = NeuralTreeBoosted(estimator,X_train,y_train,output_func='sigmoid',gamma_output=1.5,gamma = [3.3,3.3])
'''

'''NRF_analyticWeights'''
from NRF_analyticWeights import *
nrf_analyticsWeights = NeuralTree_analyticWeights(estimator,X_train,y_train,output_func='sigmoid',gamma_output=1.5,gamma = [2.3,2.3])
'''
print(nrf_analyticsWeights.leaves)
print(nrf_analyticsWeights.get_probs())

stats = {leaf:{'appeared':0,'correct':0} for leaf in nrf_analyticsWeights.leaves}
print(X_test.shape[0])
for j in range(X_test.shape[0]):
    lf = estimator.apply(X_test[j,:].reshape(1,-1))
    stats[lf[0]]['appeared'] += 1
    pred = estimator.predict(X_test[j,:].reshape(1,-1))
    if pred == y_test[j]:
        stats[lf[0]]['correct'] += 1


print(stats)
print(nrf_analyticsWeights.leaves)
print(nrf_analyticsWeights.get_probs())

for leaf in stats.keys():
    print(leaf)
    print(stats[leaf]['correct']/stats[leaf]['appeared'])


N = 0
arr = []
y_arr = []
for j in range(X_train.shape[0]):
    pred = estimator.predict(X_train[j,:].reshape(1,-1))
    if pred == y_train[j]:
        N += 1
    else:
        print(X_train[j,:])
        arr.append(X_train[j,:])
        y_arr.append(y_train[j])

'''
'''NRF_extraLayer_analyticWeights
from NRF_withExtraLayer_analyticWeights import *
nrf_EL_analyticsWeights = NeuralTree_extraLayer_analyticWeights(estimator,X_train,y_train,output_func='sigmoid',
                                                                gamma = [2.3,2.3],gamma_sigmoid=2,gamma_output=1,
                                                                learning_rate1=0.015,learning_rate2=0.15)
'''

'''NRF_extraLayer_classic
from NRF_withExtraLayer import *
nrf_EL= NeuralTree_extraLayer(estimator,X_train,y_train,output_func='sigmoid',gamma = [2.3,2.3],gamma_sigmoid=1,gamma_output=1.5)
'''

#print(nrf2.weights[-1])


'''
partialNRF = nrf2.initialNRF
X_trainLS = list(X_train)
for data in X_trainLS:
    d = data.reshape(-1,1)
    final = partialNRF.network.feedforward(d)
    print(final)
    print(estimator.predict_proba(data.reshape(1,-1)))

'''
nrf_analyticsWeights = NeuralTree_analyticWeights(estimator,X_train,y_train,output_func='sigmoid',gamma_output=1.5,gamma = [2.3,2.3])

#nrf = NeuralTree(estimator,X_train,y_train)
#nrf.train_NRF(100,20,0.025)
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = nrf_analyticsWeights.train_NRF(30,10,0.0055,0.03,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT = nrf_analyticsWeights.predict(X_test)
#print(predictions_NRT)
#predictions_NR = nrf.predict(X_test)

print(classification_report(y_test,predictions_NRT))
#print(classification_report(y_test,predictions_NR))


import matplotlib.pyplot as plt

x = range(0,30)
plt.figure(1)
plt.plot(x,training_cost)

plt.figure(2)
plt.plot(x,training_accuracy)

plt.show()


