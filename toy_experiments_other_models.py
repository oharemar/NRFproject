import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import keras
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
from matplotlib.colors import ListedColormap
from sklearn import datasets

# make_classification random states: 1
# make_blobs random states: 10
# make_gaussian_quantiles random states: 1,5
# make_circles random states: 1,3
# make_moons random states: 2


if __name__ == '__main__':

    #random_state = 1
    #X,y = datasets.make_blobs(400,2,centers=[[2,3],[1,5],[3,5]],center_box=(0,7),cluster_std=0.7,random_state=10)

    #X,y = datasets.make_classification(400,2,2,0,n_classes=3,n_clusters_per_class=1,random_state=1)

    #X,y = datasets.make_gaussian_quantiles(n_samples=400,n_classes=3,random_state=5)

    #X,y = datasets.make_circles(400,noise=0.1,random_state=3)

    X,y = datasets.make_moons(400,noise=0.3,random_state=2)

    rf = RandomForestClassifier(n_estimators=10,max_depth=6,random_state=1)

    h = .05
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    y_train_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
    y_test_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
    for j, k in zip(y_train, range(len(y_train))):
        y_train_keras[int(k), int(j)] = 1
    for j, k in zip(y_test, range(len(y_test))):
        y_test_keras[int(k), int(j)] = 1
    y_train_keras = y_train_keras.astype('int64')
    y_test_keras = y_test_keras.astype('int64')

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    #cm = plt.cm.RdYlBu
    #cm = plt.cm.RdYlBu # tohle používat jen pro binární klasifikaci a vykreslíme pravděpodobnosti
    cm_bright = ListedColormap(['#FF0000','#FFFF00','#0000FF'])
    #print(y)
    #cm_bright = ListedColormap(['#0000FF','#FF0000','#FFFF00'])



    #plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
    # Plot the testing points
    #plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,edgecolors='k')

    plt.xlim([xx.min(), xx.max()])
    plt.ylim([yy.min(), yy.max()])

    rf.fit(X_train, y_train)

    mod = Sequential()
    mod.add(Dense(units=60, activation='relu', input_shape=(X_train.shape[1],)))  # 60 units zkusit
    mod.add(Dense(units=60, activation='relu'))  # 60 units zkusit
    mod.add(Dense(units=max(y_train) + 1,
                  activation='softmax'))  # přičítáme 1, protože předpokládáme, že první classa je 0
    sgd = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

    mod.compile(loss='categorical_crossentropy',
                optimizer=sgd,  # 'sgd',
                metrics=['accuracy'])
    mod.fit(X_train, y_train_keras, epochs=30)
    predictions_nn = np.argmax(mod.predict(X_test), axis=1)

    predictions = rf.predict(X_test)

    print(classification_report(y_test,predictions,output_dict=True)['accuracy'])
    print(classification_report(y_test,predictions_nn,output_dict=True)['accuracy'])


    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_nn = np.argmax(mod.predict(np.c_[xx.ravel(), yy.ravel()]),axis=1)


    Z = Z.reshape(xx.shape)
    Z_nn = Z_nn.reshape(xx.shape)

    #ax = plt.subplot()
    #plt.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.5)
    plt.contourf(xx, yy, Z_nn, cmap=cm_bright, alpha=0.5)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,edgecolors='k', alpha=0.7)

    #plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
    # Plot the testing points
    #plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,edgecolors='k', alpha=0.7)



    #plt.scatter(X_2[:,0],X_2[:,1],c=y_2)
    #plt.scatter(X[y==2,0],X[y==2,1])#,c=y)

    plt.show()
