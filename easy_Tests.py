import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from NRF_basic import NeuralTreeBasic
import sklearn

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
estimator.fit(X_train, y_train)
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left # z toho udělám first hidden layer of NRF, second hidden layer of NRF
children_right = estimator.tree_.children_right # z toho udělám first hidden layer of NRF, second hidden layer of NRF
feature = estimator.tree_.feature # z toho udělám first hidden layer of NRF
threshold = estimator.tree_.threshold # z toho udělám first hidden layer of NRF

print(n_nodes)
print(estimator.get_n_leaves())
print(threshold)
print(children_left)
print(children_right)
print(feature)

# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
node_indicator = estimator.decision_path(X_test) # zjistím, které nody jsou involved v path do listu

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = estimator.apply(X_test) # zjistím, které instance došly do toho listu a pomocí decision path zjistím, které nody jsou v té cestě involved



nrf = NeuralTreeBasic(estimator)

print(nrf.inner_nodes)
print(nrf.leaves)
print(nrf.weights)
print(nrf.biases)
L = [1,2,3,4,5]
L = [L]*2
#L = 2*L
print(L)

k = [[10,11,12]]
l = 4*k
#L[1:4] = k
#print(l)

probs = [[0.2,0.4],[0.5,0.7]]

leaves = [0,1]
n_classes = 2

weights = np.zeros((n_classes, len(leaves) * n_classes),dtype=np.float64)
for label in range(n_classes):
    leaf_part_weights = [0 for j in range(len(leaves) * n_classes)]
    leaf_part_weights[label:len(leaves) * n_classes:n_classes] = [1 for j in range(len(leaves))]
    leaf_part_weights = np.array(leaf_part_weights, dtype=np.float64)
    weights[label, :] = leaf_part_weights

biases = np.zeros((n_classes, 1), dtype=np.float64)  # initial biases are zero
print(weights)
print(biases)
