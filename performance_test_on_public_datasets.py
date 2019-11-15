import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

'''DATASET1 bank_marketing_dataset'''

le = LabelEncoder()
pd.set_option("display.max_columns", 999)
df = pd.read_csv('bank_marketing_dataset.csv')
df = df.sample(frac=1)
y = df.loc[:,'y']
df = df.loc[:,df.columns != 'y']
y = le.fit_transform(y)
#print(df.describe(include = object))
df.drop(columns = ['default'],inplace = True)
'''ONE HOT ENCODING CATEGORICAL VARIABLES'''
df = pd.get_dummies(df)
#print(df)
''''''
'''DATASET2 bank_marketing_dataset'''













'''TRAINING AND TESTING DATASET'''
X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.33)
''''''

'''MODELS'''
lg = LogisticRegression(solver='newton-cg',tol=1e-2,max_iter=1000)
svm = SVC(kernel='sigmoid')
rf = RandomForestClassifier(n_estimators=30,criterion='entropy',max_depth=5,min_samples_split=3,max_features=25)
''''''

'''FIT MODELS'''
#print(cross_val_score(lg,X_train,y_train))
lg.fit(X_train,y_train)
svm.fit(X_train,y_train)
rf.fit(X_train,y_train)
''''''

'''GET PREDICTIONS'''
predictions_lg = lg.predict(X_test)
predictions_svm = svm.predict(X_test)
predicitons_rf = rf.predict(X_test)
''''''

'''PRINT CLASSIFICATION PERFORMANCE REPORT'''
print('Logistic Regression')
print(classification_report(y_test,predictions_lg))
print('-----------')
print('SVM')
print(classification_report(y_test,predictions_svm))
print('-----------')
print('RF')
print(classification_report(y_test,predicitons_rf))
''''''