from NRF_analyticWeights import *
from statistics import mode
import random

class NeuralRandomForest():

    def __init__(self,random_forest, NRF_type = 'NRF_analyticWeights',X_train = None, y_train = None,
                 output_func = 'sigmoid',cost_func = 'CrossEntropy',gamma_output = 1, gamma = [15,15]):
        self.random_forest = random_forest
        self.NRF_ensemble = []
        if cost_func == 'CrossEntropy':
            self.cost_func = CrossEntropyCost
        self.output_func = output_func
        self.NRF_type = NRF_type
        self.training_data = X_train
        self.training_labels = y_train
        self.gamma_output = gamma_output
        self.gamma = gamma

    def get_NRF_ensemble(self,epochs,mini_batch,eta,lmbda):
        for estimator in self.random_forest.estimators_:
            if self.NRF_type == 'NRF_analyticWeights':
                nrf = NeuralTree_analyticWeights(decision_tree = estimator, X_train = self.training_data, y_train = self.training_labels,
                 output_func = self.output_func,gamma_output = self.gamma_output, gamma = self.gamma)
                nrf.train_NRF(epochs,mini_batch,eta,lmbda,monitor_training_accuracy=True,monitor_training_cost=True)
                self.NRF_ensemble.append(nrf)

    def predict(self, X_test):

        predictions = np.zeros((X_test.shape[0],len(self.NRF_ensemble)))
        for nrf,index in zip(self.NRF_ensemble,range(len(self.NRF_ensemble))):
            preds = nrf.predict(X_test)#.reshape(-1,1)
            predictions[:,index] = preds

        predictions = predictions.tolist()
        final_predictions = []
        for nrf_preds in predictions:
            try:
                val = mode(nrf_preds) # we choose the most frequent element of the list
                final_predictions.append(val)
            except:
                val = random.choice(nrf_preds) # if all values are same common, we pick one random value
                final_predictions.append(val)


        return np.array(final_predictions)