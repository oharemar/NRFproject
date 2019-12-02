from NRF import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(X_train.shape)

estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
estimator.fit(X_train, y_train)




#nrf = NeuralTree(estimator,X_train,y_train)

test_data = list(zip(list(X_test), list(y_test)))
test_data_aligned = []
good = 0
total = 0
for j in range(len(test_data)):
    total += 1
    test_data_aligned.append((test_data[j][0].reshape(-1, 1), test_data[j][1]))
    pred_label = estimator.predict(test_data[j][0].reshape(1,-1))
    if pred_label == test_data[j][1]:
        good += 1

print('DECISION TREE')
print('{} / {}'.format(good,total))

'''
#nrf.train_NRF(10,10,2)

from NRF_basic import *

nrf2 = NeuralTreeBasic(estimator,X_train,y_train)

#nrf2.train_NRF(10,10,2)

from ANN import *


train_labels_temp = list(y_train)
train_labels = []
for label in train_labels_temp:
    lab = np.zeros((estimator.n_classes_,1),dtype=np.float64)
    index = list(estimator.classes_).index(label)
    lab[index,0] = 1.0
    train_labels.append(lab)

train_data = list(zip(list(X_train), train_labels))
train_data_aligned = []
for j in range(len(train_data)):
    train_data_aligned.append((train_data[j][0].reshape(-1, 1), train_data[j][1]))


net = Network(sizes = [4,6,7,3])
'''
#net.SGD(train_data_aligned,10,10,2,test_data_aligned)
'''
print(nrf.decision_tree.tree_.children_left)
print(nrf.decision_tree.tree_.children_right)
#print(nrf.decision_tree.tree_.feature)
#print(nrf.decision_tree.tree_.threshold)
print(nrf.network.weights)
print(nrf.network.biases)
print(train_data_aligned[6][0])
print(nrf.inner_nodes)
print(nrf.leaves)
print(nrf.decision_tree.apply(train_data_aligned[6][0].reshape(1,-1)))
print(nrf.label_numbers)
print(train_data_aligned[6][1])
nrf.network.feedforward(train_data_aligned[6][0])
'''

train_labels_temp = list(y_train)
train_labels = []
for label in train_labels_temp:
    lab = np.zeros((estimator.n_classes_,1),dtype=np.float64)
    index = list(estimator.classes_).index(label)
    lab[index,0] = 1.0
    train_labels.append(lab)

train_data = list(zip(list(X_train), train_labels))
train_data_aligned = []
for j in range(len(train_data)):
    train_data_aligned.append((train_data[j][0].reshape(-1, 1), train_data[j][1]))


'''EXTRA LAYER RANDOM WEIGHTS'''
from NRF_withExtraLayerCombined_lessInitialModification import *

nrf = NeuralTree_withExtraLayerCombined(estimator,X_train,y_train)


def evaluate(test_data,NRF):
    test_results = [(np.argmax(NRF.network.feedforward(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)


#print(nrf.network.weights)
#print(nrf.network.biases)
#print(evaluate(test_data_aligned,nrf))
nrf.train_NRF(100,28,0.1,test_data_aligned)
