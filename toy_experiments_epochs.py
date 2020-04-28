from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from NeuralRandomForest_parallel import NeuralRandomForest
import statistics
from sklearn.model_selection import KFold
from helpful_functions import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets as dt

# make_classification random states: 1
# make_blobs random states: 10
# make_gaussian_quantiles random states: 1,5
# make_circles random states: 1,3
# make_moons random states: 2

"""TRY TO SIMULATE OVERFITTING  -  complex random forest overfits to the noise in the data - NRF and show that it repairs
overfitting or add regularization constant and check that it helps to increase accuracy and generalization"""

if __name__ == '__main__':

    samples = [20,50,100,200,400,800]
    for sample in samples:
        #random_state = 1
        X_1,y_1 = dt.make_blobs(sample,2,centers=[[2,3],[1,5],[3,5]],center_box=(0,7),cluster_std=0.7,random_state=10)

        X_2,y_2 = dt.make_classification(sample,2,2,0,n_classes=3,n_clusters_per_class=1,random_state=1)

        X_3,y_3 = dt.make_gaussian_quantiles(n_samples=sample,n_classes=3,random_state=5)

        X_4,y_4 = dt.make_circles(sample,noise=0.1,random_state=3)

        X_5,y_5 = dt.make_moons(sample,noise=0.3,random_state=2)

        datasets = [(X_1,y_1),(X_2,y_2),(X_3,y_3),(X_4,y_4),(X_5,y_5)]
        names = ['Input data','Random Forest', 'NRF_DW', 'NRF_EL_DW', 'NRF_EL_DW_id', 'NRF','NN']
        epochs = [5,10,20,30,50]
        h = .05
        cm_bright = ListedColormap(['#FF0000','#FFFF00','#0000FF'])
        i = 1
        dataset = 1

        for X,y in datasets:
            fig = plt.figure(figsize=(15, 10))
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
            rf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=1)
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            plt.xlim([xx.min(), xx.max()])
            plt.ylim([yy.min(), yy.max()])
            rf.fit(X_train, y_train)

            for epoch in epochs:
                for model in names:
                    ax = plt.subplot(len(epochs),len(names),i)
                    ax.set_xticks(())
                    ax.set_yticks(())
                    try:
                        if model == 'Input data':
                            ax.set_ylabel('Epochs = {}'.format(epoch))
                            if i == 1:
                                ax.set_title(model)
                            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
                            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.7)
                            i += 1
                            continue
                        elif model == 'Random Forest':
                            mod = rf
                        elif model == 'NRF_DW':
                            mod = NeuralRandomForest(rf, 'NRF_analyticWeights_adam', X_train, y_train,
                                                       output_func='softmax',
                                                       cost_func='CrossEntropy',
                                                       gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                            mod.get_NRF_ensemble(epoch, 10, 0.0035, 0.0)
                        elif model == 'NRF':
                            mod = NeuralRandomForest(rf, 'NRF_basic_adam', X_train, y_train,
                                                     output_func='softmax',
                                                     cost_func='CrossEntropy',
                                                     gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                            mod.get_NRF_ensemble(epoch, 10, 0.002, 0.0)
                        elif model == 'NRF_EL_DW':
                            mod = NeuralRandomForest(rf, 'NRF_extraLayer_adam', X_train, y_train,
                                                     output_func='softmax',
                                                     cost_func='CrossEntropy',
                                                     gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                            mod.get_NRF_ensemble(epoch, 10, 0.01, 0.0)
                        elif model == 'NRF_EL_DW_id':
                            mod = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_ultra_adam', X_train, y_train,
                                                       output_func='softmax',
                                                       cost_func='CrossEntropy',
                                                       gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                            mod.get_NRF_ensemble(epoch, 10, 0.0045, 0.0)
                        elif model == 'NN':
                            y_train_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
                            y_test_keras = np.zeros((X_train.shape[0], int(max(y_train) + 1)))
                            for j, k in zip(y_train, range(len(y_train))):
                                y_train_keras[int(k), int(j)] = 1
                            for j, k in zip(y_test, range(len(y_test))):
                                y_test_keras[int(k), int(j)] = 1
                            y_train_keras = y_train_keras.astype('int64')
                            y_test_keras = y_test_keras.astype('int64')
                            mod = Sequential()
                            mod.add(Dense(units=60, activation='relu', input_shape=(X_train.shape[1],)))  # 60 units zkusit
                            mod.add(Dense(units=60, activation='relu'))  # 60 units zkusit
                            mod.add(Dense(units=max(y_train) + 1,
                                          activation='softmax'))  # přičítáme 1, protože předpokládáme, že první classa je 0
                            sgd = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

                            mod.compile(loss='categorical_crossentropy',
                                        optimizer=sgd,  # 'sgd',
                                        metrics=['accuracy'])
                            mod.fit(X_train, y_train_keras, epochs=epoch)
                    except:
                        if i in [1, 2, 3, 4, 5, 6, 7]:
                            ax.set_title(model)
                        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
                        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.7)
                        i += 1
                        continue
                    else:
                        if model == 'NN':
                            predictions = np.argmax(mod.predict(X_test), axis=1)
                            Z = np.argmax(mod.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
                        else:
                            predictions = mod.predict(X_test)
                            Z = mod.predict(np.c_[xx.ravel(), yy.ravel()])

                        Z = Z.reshape(xx.shape)

                        if i in [1,2,3,4,5,6,7]:
                            ax.set_title(model)
                        ax.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.5)
                        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
                        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.7)
                        if dataset == 3:
                            ax.text(xx.min() + .3, yy.min() + .3,
                                    '{:.2f}'.format(classification_report(y_test, predictions, output_dict=True)['accuracy']),
                                    horizontalalignment='left', bbox=dict(facecolor='green', alpha=0.4))
                        else:
                            ax.text(xx.max() - .3, yy.min() + .3,'{:.2f}'.format(classification_report(y_test,predictions,output_dict=True)['accuracy']),
                                horizontalalignment='right',bbox=dict(facecolor='green', alpha=0.4))
                        i += 1
            i = 1
            plt.tight_layout()
            fig.savefig('epochs_{}_samples{}'.format(dataset,sample))
            dataset += 1
