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

def get_network_dataset(training_data,training_labels,estimator):
    train_labels_temp = list(training_labels)
    train_labels = []
    for label in train_labels_temp:
        lab = np.zeros((estimator.n_classes_, 1), dtype=np.float64)
        index = list(estimator.classes_).index(label)
        lab[index, 0] = 1.0
        train_labels.append(lab)

    train_data = list(zip(list(training_data), train_labels))
    train_data_aligned = []
    for j in range(len(train_data)):
        train_data_aligned.append((train_data[j][0].reshape(-1, 1), train_data[j][1]))

    return  train_data_aligned

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

df = np.array(df,dtype=np.float64)
y = np.array(y)

estimator = DecisionTreeClassifier(max_leaf_nodes=6, random_state=0)

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.33) # split to train and test dataset

estimator.fit(X_train,y_train)
predictions_DT = estimator.predict(X_test)

print(classification_report(y_test,predictions_DT))

#from NRF_withExtraLayerCombined import *
#nrf2 = NeuralTree_withExtraLayerCombined(estimator,X_train,y_train)

from NRF_analyticWeights import *

nrf2 = NeuralTree_analyticWeights(estimator,X_train,y_train,output_func='softmax',gamma_output=2,gamma = [1.3,1.3])

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
#nrf = NeuralTree(estimator,X_train,y_train)
#nrf.train_NRF(100,20,0.025)
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = nrf2.train_NRF(100,10,0.0035,0.1,monitor_training_cost=True,monitor_training_accuracy=True)
predictions_NRT = nrf2.predict(X_test)
#print(predictions_NRT)
#predictions_NR = nrf.predict(X_test)

print(classification_report(y_test,predictions_NRT))
#print(classification_report(y_test,predictions_NR))


import matplotlib.pyplot as plt

x = range(0,100)
plt.figure(1)
plt.plot(x,training_cost)

plt.figure(2)
plt.plot(x,training_accuracy)

plt.show()


