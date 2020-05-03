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
        #X_1,y_1 = datasets.make_blobs(400,2,centers=[[2,3],[1,5],[3,5]],center_box=(0,7),cluster_std=0.7,random_state=10)

        #X_2,y_2 = datasets.make_classification(400,2,2,0,n_classes=3,n_clusters_per_class=1,random_state=1)

        #X_3,y_3 = datasets.make_gaussian_quantiles(n_samples=400,n_classes=3,random_state=5)

        X_4,y_4 = dt.make_circles(sample,noise=0.13,random_state=3)
        X_4 = StandardScaler().fit_transform(X_4)

        #X_5,y_5 = datasets.make_moons(400,noise=0.3,random_state=2)

        datasets = [(X_4,y_4)]
        names = ['Random Forest', 'NRF_DW', 'NRF_EL_DW_id']
        #acc = {'Random Forest':[],'NRF_DW':[],'NRF_EL_DW_id':[]}
        regularization = [0,0.2,0.4,0.6,0.8,1,2,4]
        vals = np.zeros((len(regularization),3),dtype=np.float64)
        vals_std = np.zeros((len(regularization),3),dtype=np.float64)

        kf = KFold(n_splits=5)
        for term,j in zip(regularization,range(len(regularization))):
            acc = {'Random Forest': [], 'NRF_DW': [], 'NRF_EL_DW_id': []}
            for _ in range(2):
                for train_index, test_index in kf.split(X_4):
                    rf = RandomForestClassifier(n_estimators=2, max_depth=20, random_state=1, max_features=1.0)
                    X_train, X_test = X_4[train_index], X_4[test_index]
                    y_train, y_test = y_4[train_index], y_4[test_index]
                    rf.fit(X_train,y_train)
                    for model in names:
                        if model == 'Random Forest':
                            mod = rf
                        elif model == 'NRF_DW':
                            mod = NeuralRandomForest(rf, 'NRF_analyticWeights_adam', X_train, y_train,
                                                       output_func='softmax',
                                                       cost_func='CrossEntropy',
                                                       gamma_output=1, gamma=[10, 10])  # zde změna, gamma_output je 1
                            mod.get_NRF_ensemble(30, 10, 0.0035, term)
                        elif model == 'NRF':
                            mod = NeuralRandomForest(rf, 'NRF_basic_adam', X_train, y_train,
                                                     output_func='softmax',
                                                     cost_func='CrossEntropy',
                                                     gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                            mod.get_NRF_ensemble(term, 10, 0.002, 0.0)
                        elif model == 'NRF_EL_DW':
                            mod = NeuralRandomForest(rf, 'NRF_extraLayer_adam', X_train, y_train,
                                                     output_func='softmax',
                                                     cost_func='CrossEntropy',
                                                     gamma_output=1, gamma=[1, 1])  # zde změna, gamma_output je 1
                            mod.get_NRF_ensemble(term, 10, 0.01, 0.0)
                        elif model == 'NRF_EL_DW_id':
                            mod = NeuralRandomForest(rf, 'NRF_extraLayer_analyticWeights_ultra_adam', X_train, y_train,
                                                       output_func='softmax',
                                                       cost_func='CrossEntropy',
                                                       gamma_output=1, gamma=[10, 10])  # zde změna, gamma_output je 1
                            mod.get_NRF_ensemble(30, 10, 0.0045, term)


                        predictions = mod.predict(X_test)
                        acc_temp = classification_report(y_test,predictions,output_dict=True)['accuracy']
                        acc[model].append(acc_temp)
            vals[j,0] = statistics.mean(acc['Random Forest'])
            vals_std[j,0] = statistics.stdev(acc['Random Forest'])
            vals[j,1] = statistics.mean(acc['NRF_DW'])
            vals_std[j,1] = statistics.stdev(acc['NRF_DW'])
            vals[j,2] = statistics.mean(acc['NRF_EL_DW_id'])
            vals_std[j,2] = statistics.stdev(acc['NRF_EL_DW_id'])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xticks(np.arange(len(names)), labels=names, rotation = 20,fontsize=5)
        plt.yticks(np.arange(len(regularization)), labels=regularization)
        im = ax.imshow(vals, vmin=max([np.amin(vals)-0.05,0]), vmax=min([np.amax(vals)+0.05,1]))
        plt.colorbar(im, label='Accuracy')
        # plt.clim(min_f1, max_f1)
        plt.xlabel('Models')
        plt.ylabel(r'$\lambda$')
        plt.title('Regularization')
        for k in range(len(names)):
            for j in range(len(regularization)):
                text = ax.text(k, j, u'{:.3f}\n\u00B1\n{:.3f}'.format(vals[j, k], vals_std[j, k]),
                               ha="center", va="center", color="red", fontsize=7)
        fig.savefig('toy_reg_{}samples.png'.format(sample))

