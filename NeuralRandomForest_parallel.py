from statistics import mode
import numpy as np
import random
from helpful_functions import *
import concurrent.futures


class NeuralRandomForest():

    def __init__(self,random_forest, NRF_type = 'NRF_analyticWeights',X_train = None, y_train = None,
                 output_func = 'sigmoid',cost_func = 'CrossEntropy',gamma_output = 1, gamma = [15,15]):
        self.random_forest = random_forest
        self.NRF_ensemble = []
        self.cost_func = cost_func
        self.output_func = output_func
        self.NRF_type = NRF_type
        self.training_data = X_train
        self.training_labels = y_train
        self.gamma_output = gamma_output
        self.gamma = gamma


    def get_NRT(self,estimator,epochs,mini_batch,eta,lmbda,penultimate_func = 'LeakyReLU'):

        if self.NRF_type == 'NRF_analyticWeights':
            from NRF_analyticWeights import NeuralTree_analyticWeights
        elif self.NRF_type == 'NRF_analyticWeights_adam':
            from NRF_analyticWeights_adam import NeuralTree_analyticWeights
        elif self.NRF_type == 'NRF_analyticWeights_nesterov':
            from NRF_analyticWeights_nesterov import NeuralTree_analyticWeights
        elif self.NRF_type == 'NRF_basic':
            from NRF_basic_boosted import NeuralTreeBasic_boosted
        elif self.NRF_type == 'NRF_basic_adam':
            from NRF_basic_boosted_adam import NeuralTreeBasic_boosted_adam
        elif self.NRF_type == 'NRF_basic_nesterov':
            from NRF_basic_boosted_nesterov import NeuralTreeBasic_boosted_nesterov
        elif self.NRF_type == 'NRF_extraLayer':
            from NRF_withExtraLayer import NeuralTree_extraLayer
        elif self.NRF_type == 'NRF_extraLayer_adam':
            from NRF_withExtraLayer_adam import NeuralTree_extraLayer
        elif self.NRF_type == 'NRF_extraLayer_nesterov':
            from NRF_withExtraLayer_nesterov import NeuralTree_extraLayer
        elif self.NRF_type == 'NRF_extraLayer_analyticWeights':
            from NRF_withExtraLayer_analyticWeights import NeuralTree_extraLayer_analyticWeights
        elif self.NRF_type == 'NRF_extraLayer_analyticWeights_adam':
            from NRF_withExtraLayer_analyticWeights_adam import NeuralTree_extraLayer_analyticWeights
        elif self.NRF_type == 'NRF_extraLayer_analyticWeights_nesterov':
            from NRF_withExtraLayer_analyticWeights_nesterov import NeuralTree_extraLayer_analyticWeights

        if self.NRF_type == 'NRF_analyticWeights':
            nrf = NeuralTree_analyticWeights(decision_tree=estimator, X_train=self.training_data,
                                             y_train=self.training_labels,
                                             output_func=self.output_func, gamma_output=self.gamma_output,
                                             gamma=self.gamma, cost=self.cost_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)

        elif self.NRF_type == 'NRF_analyticWeights_adam':
            nrf = NeuralTree_analyticWeights(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_analyticWeights_nesterov':
            nrf = NeuralTree_analyticWeights(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_basic':
            nrf = NeuralTreeBasic_boosted(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_basic_adam':
            nrf = NeuralTreeBasic_boosted_adam(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_basic_nesterov':
            nrf = NeuralTreeBasic_boosted_nesterov(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_extraLayer':
            nrf = NeuralTree_extraLayer(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func,penultimate_func=penultimate_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_extraLayer_adam':
            nrf = NeuralTree_extraLayer(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func,penultimate_func=penultimate_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_extraLayer_nesterov':
            nrf = NeuralTree_extraLayer(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func,penultimate_func=penultimate_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_extraLayer_analyticWeights':
            nrf = NeuralTree_extraLayer_analyticWeights(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func,penultimate_func=penultimate_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_extraLayer_analyticWeights_adam':
            nrf = NeuralTree_extraLayer_analyticWeights(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func,penultimate_func=penultimate_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)
        elif self.NRF_type == 'NRF_extraLayer_analyticWeights_nesterov':
            nrf = NeuralTree_extraLayer_analyticWeights(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
             output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma, cost=self.cost_func,penultimate_func=penultimate_func)
            nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=False,monitor_training_cost=False)

        return nrf

    def get_NRF_ensemble(self,epochs,mini_batch,eta,lmbda,penultimate_func = 'LeakyReLU'): # running in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            args = [(estimator,epochs,mini_batch,eta,lmbda,penultimate_func) for estimator in self.random_forest.estimators_]
            nrf_list = executor.map(self.get_NRT,*zip(*args))
            for nrt in nrf_list:
                self.NRF_ensemble.append(nrt)

    def predict_averaging(self, X_test):

        predictions = np.zeros((X_test.shape[0],self.NRF_ensemble[0].decision_tree.n_classes_))
        for nrf,index in zip(self.NRF_ensemble,range(len(self.NRF_ensemble))):
            predictions += nrf.predict_prob(X_test)#.reshape(-1,1)

        predictions = (1/len(self.NRF_ensemble))*predictions

        return np.argmax(predictions,axis=1)

    def predict_averaging_loss(self, X_test):

        predictions = np.zeros((X_test.shape[0], self.NRF_ensemble[0].decision_tree.n_classes_))
        for nrf, index in zip(self.NRF_ensemble, range(len(self.NRF_ensemble))):
            predictions += nrf.predict_prob(X_test)  # .reshape(-1,1)

        predictions = (1 / len(self.NRF_ensemble)) * predictions

        return predictions


    def predict(self, X_test):

        predictions = np.zeros((X_test.shape[0],len(self.NRF_ensemble)))
        for nrf,index in zip(self.NRF_ensemble,range(len(self.NRF_ensemble))):
            preds = nrf.predict(X_test)#.reshape(-1,1)
            predictions[:,index] = preds

        predictions = predictions.tolist()
        final_predictions = []
        for nrf_preds in predictions:
            x = np.array(nrf_preds)
            val = most_common_value(x)
            final_predictions.append(val)
            '''
            try:
                val = mode(nrf_preds) # we choose the most frequent element of the list
                final_predictions.append(val)
            except:
                val = random.choice(nrf_preds) # if all values are same common, we pick one random value
                final_predictions.append(val)
            '''

        return np.array(final_predictions)