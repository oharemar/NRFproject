from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from helpful_functions import *
from sklearn.model_selection import KFold
from statistics import mean
import keras
from sklearn.metrics import log_loss
from statistics import stdev
import  matplotlib.pyplot as plt
import json


df, y = load_datasets('vehicle_silhouette')

epochs = [5,10,20,30,40,50,60,70,80,90,100]


accuracy_train= []
accuracy_test= []
loss_train = []
loss_test = []

# prepare RF

#rf = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=6, max_features='auto')

kf = KFold(n_splits=5)

final_results_vals = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                  'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
final_results_stds = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                  'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}

results_all = []

for _ in range(2):
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

        accuracy_train_temp = []
        accuracy_test_temp = []
        loss_train_temp = []
        loss_test_temp = []

        for epoch in epochs:
            mod = Sequential()
            mod.add(Dense(units=60, activation='relu', input_shape=(X_train.shape[1],))) # 60 units zkusit
            mod.add(Dense(units=60, activation='relu'))                                     # 60 units zkusit
            mod.add(Dense(units=max(y_train) + 1,
                        activation='softmax'))  # přičítáme 1, protože předpokládáme, že první classa je 0
            sgd = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

            mod.compile(loss='categorical_crossentropy',
                      optimizer=sgd,  # 'sgd',
                      metrics=['accuracy'])
            mod.fit(X_train, y_train_keras, epochs=epoch)
            predictions_test = np.argmax(mod.predict(X_test), axis=1)
            predictions_train = np.argmax(mod.predict(X_train), axis=1)
            preds_loss_train = mod.predict(X_train)
            preds_loss_test = mod.predict(X_test)



            results_train = classification_report(y_train, predictions_train, output_dict=True)
            results_test = classification_report(y_test, predictions_test, output_dict=True)

            accuracy_train_temp.append(results_train['accuracy'])
            accuracy_test_temp.append(results_test['accuracy'])
            loss_train_temp.append(log_loss(y_train, preds_loss_train))
            loss_test_temp.append(log_loss(y_test, preds_loss_test))

        accuracy_train.append(accuracy_train_temp)
        accuracy_test.append(accuracy_test_temp)
        loss_train.append(loss_train_temp)
        loss_test.append(loss_test_temp)

accuracy_train_mean = list(np.mean(np.array(accuracy_train, dtype=np.float64), axis=0))
accuracy_train_std = list(np.std(np.array(accuracy_train, dtype=np.float64), axis=0))

accuracy_test_mean = list(np.mean(np.array(accuracy_test, dtype=np.float64), axis=0))
accuracy_test_std = list(np.std(np.array(accuracy_test, dtype=np.float64), axis=0))

loss_train_mean = list(np.mean(np.array(loss_train, dtype=np.float64), axis=0))
loss_train_std = list(np.std(np.array(loss_train, dtype=np.float64), axis=0))

loss_test_mean = list(np.mean(np.array(loss_test, dtype=np.float64), axis=0))
loss_test_std = list(np.std(np.array(loss_test, dtype=np.float64), axis=0))




fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(epochs, accuracy_train_mean, yerr=accuracy_train_std, ecolor='darkorange', marker='o')
#ax.plot(epochs, accuracy_train_nrf,color = 'red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xlim([0,110])
plt.title('NN')
plt.ylim([min(accuracy_train_mean) - 0.05, max(accuracy_train_mean) + 0.05])
fig.savefig('NN_epochs_acc_train.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(epochs, accuracy_test_mean, yerr=accuracy_test_std, ecolor='darkorange', marker='o')
#ax.plot(epochs, accuracy_train_nrf,color = 'red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xlim([0,110])
plt.title('NN')
plt.ylim([min(accuracy_test_mean) - 0.05, max(accuracy_test_mean) + 0.05])
fig.savefig('NN_epochs_acc_test.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(epochs, loss_train_mean, yerr=loss_train_std, ecolor='darkorange', marker='o')
#ax.plot(epochs, accuracy_train_nrf,color = 'red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xlim([0,110])
plt.title('NN')
plt.ylim([min(loss_train_mean) - 0.05, max(loss_train_mean) + 0.05])
fig.savefig('NN_epochs_loss_train.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(epochs, loss_test_mean, yerr=loss_test_std, ecolor='darkorange', marker='o')
#ax.plot(epochs, accuracy_train_nrf,color = 'red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xlim([0,110])
plt.title('NN')
plt.ylim([min(loss_test_mean) - 0.05, max(loss_test_mean) + 0.05])
fig.savefig('NN_epochs_loss_test.png')