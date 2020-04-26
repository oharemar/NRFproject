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

    X,y = datasets.make_circles(400,noise=0.1,random_state=3)

    #X,y = datasets.make_moons(400,noise=0.3,random_state=2)

    rf = RandomForestClassifier(n_estimators=10,max_depth=6,random_state=1)

    h = .05
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

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

    nrfdw = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_ultra_adam', X_train, y_train, output_func='softmax',
                                               cost_func='CrossEntropy',
                                               gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
    nrfdw.get_NRF_ensemble(20, 10, 0.0045, 0.0)
    predictions_nrfdw = nrfdw.predict(X_test)

    predictions = rf.predict(X_test)

    print(classification_report(y_test,predictions,output_dict=True)['accuracy'])
    print(classification_report(y_test,predictions_nrfdw,output_dict=True)['accuracy'])


    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_nrf = nrfdw.predict(np.c_[xx.ravel(), yy.ravel()])


    Z = Z.reshape(xx.shape)
    Z_nrf = Z_nrf.reshape(xx.shape)

    #fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    #ax = plt.subplot()
    ax1.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.5)
    ax2.contourf(xx, yy, Z_nrf, cmap=cm_bright, alpha=0.5)

    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.7)

    ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,edgecolors='k', alpha=0.7)

    ax1.set_title('RF')
    ax2.set_title('NRF')

    #plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
    # Plot the testing points
    #plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,edgecolors='k', alpha=0.7)



    #plt.scatter(X_2[:,0],X_2[:,1],c=y_2)
    #plt.scatter(X[y==2,0],X[y==2,1])#,c=y)

    plt.show()
