import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from helpful_functions import *
from CostFunctions import *

import matplotlib.pyplot as plt

#plt.rc('text', usetex=True)

'''
N = 1000
#delta = 0.6
X = np.linspace(-4, 4, N)

plt.plot(X, leaky_relu(X,0.05), # phase field tanh profiles
        #X, np.tanh(2*X), "C2",  # composition profile
         #X, np.tanh(3 * X), "C3",  # composition profile
         #X, np.tanh(4 * X), "C4",  # composition profile
         X, (X > 1000000), 'k--')  # sharp interface

# legend
#plt.legend(('phase field', 'level set', 'sharp interface'),
 #          shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=16)

#plt.text(-4,1,r'$f(x) = tanh(\beta x)$',
         #{'color': 'black', 'fontsize': 20, 'ha': 'left', 'va': 'top',
          #'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

#plt.legend((r'$\phi(z) = \frac{1}{1 + e^{-z}}$'))#,shadow=True, loc='upper left', handlelength=1.5, fontsize=16)
plt.xlim([-4,4])
plt.title('Leaky ReLU function')



plt.xlabel('z')
plt.ylabel('y')

plt.show()

'''

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df,y = load_datasets('vehicle_silhouette')

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.2)

dt = DecisionTreeClassifier(max_depth=12,max_features=1.0)
dt.fit(X_train,y_train)
preds = dt.predict(X_test)
print(classification_report(y_test,preds))

'''
y = df.loc[:, 48]
df = df.loc[:, df.columns != 48]
scaler = StandardScaler()
df = scaler.fit_transform(df)
df = np.array(df, dtype=np.float64)
y = np.array(y)

'''



'''
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



# parametry: dataset, model

# LOAD DATASET (bank_marketing, cars, vehicle_silhouette,diabetes,messidor)
df,y = load_datasets('messidor')

# prepare DECISION TREE CLASSIFIER
estimator = DecisionTreeClassifier(max_depth=6, random_state=0) # beware of random_state, use only if necessary to repeat experiment with same trees
kf = KFold(n_splits=5)

final_results_vals = {'accuracy':0,'macro avg':{'precision':0,'recall':0,'f1-score':0,'support':0},
                     'weighted avg':{'precision':0,'recall':0,'f1-score':0,'support':0}}
final_results_stds = {'accuracy':0,'macro avg':{'precision':0,'recall':0,'f1-score':0,'support':0},
                     'weighted avg':{'precision':0,'recall':0,'f1-score':0,'support':0}}

results_all = []

for train_index, test_index in kf.split(df):
    X_train, X_test = df[train_index], df[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
    y_test_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
    y_train_keras = y_train_keras.astype('int64')
    y_test_keras = y_test_keras.astype('int64')
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')
    estimator.fit(X_train, y_train) # zde trénink modelu
    predictions_DT = estimator.predict(X_test)
    results_temp = classification_report(y_test, predictions_DT, output_dict=True)
    results_all.append(results_temp)




final_results_vals['accuracy'] =   mean([res['accuracy'] for res in results_all])
final_results_stds['accuracy'] =   stdev([res['accuracy'] for res in results_all])
final_results_vals['weighted avg']['precision'] = mean([res['weighted avg']['precision'] for res in results_all])
final_results_stds['weighted avg']['precision'] = stdev([res['weighted avg']['precision'] for res in results_all])
final_results_vals['weighted avg']['recall'] =  mean([res['weighted avg']['recall'] for res in results_all])
final_results_stds['weighted avg']['recall'] =  stdev([res['weighted avg']['recall'] for res in results_all])
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

with open('RESULTS_PUBLIC_DATASET/dataset_mean_model.txt', 'w') as file:
    file.write(json.dumps(final_results_vals))
with open('RESULTS_PUBLIC_DATASET/dataset_std_model.txt', 'w') as file:
    file.write(json.dumps(final_results_stds))


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
print(classification_report(y_test,predictions_DT,output_dict=True))
print('DECISION TREE')



'''