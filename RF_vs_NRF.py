
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

from NeuralRandomForest import *


rf = RandomForestClassifier(n_estimators=9,criterion='entropy',max_depth=6,max_features=4)

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
'''
df = pd.read_csv("diabetes.csv")
df = df.sample(frac=1)
y = df.loc[:,'Outcome']
df = df.loc[:,df.columns != 'Outcome']
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

df = np.array(df,dtype=np.float64)
y = np.array(y)

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.33) # split to train and test dataset

rf.fit(X_train,y_train)

predictions_DT = rf.predict(X_test)

print(classification_report(y_test,predictions_DT))

NRF = NeuralRandomForest(rf,X_train=X_train,y_train=y_train,gamma_output=1.5,gamma=[2.0,2.0],output_func='softmax')
NRF.get_NRF_ensemble(20,10,0.0055,0.025)

predictions_NRF = NRF.predict(X_test)
print(classification_report(y_test,predictions_NRF))
