import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

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

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left  # z toho udělám first hidden layer of NRF, second hidden layer of NRF
children_right = estimator.tree_.children_right  # z toho udělám first hidden layer of NRF, second hidden layer of NRF
feature = estimator.tree_.feature  # z toho udělám first hidden layer of NRF
threshold = estimator.tree_.threshold  # z toho udělám first hidden layer of NRF
n_leaves = estimator.get_n_leaves()
n_features = estimator.n_features_

"""first hidden layer - corresponds to inner nodes
----neurons in this layer has initial weights [0,0,...,0,1,0,...,0], where only one 1 corresponds to chosen feature in DT node split
----initial biases are thresholds of node split function
"""
inner_nodes = []
leaves = []
first_hidden_layer_weights = []
first_hidden_layer_biases = []
# first_hidden_layer_weights = np.zeros(n_nodes - n_leaves,n_features) # will have #neurons = #inner nodes of DT
# first_hidden_layer_biases = np.zeros(n_nodes - n_leaves, 1)  # will have #neurons = #inner nodes of DT

for node_id in range(n_nodes):
    if (children_left[node_id] != children_right[node_id]):  # if satisfied, we are in inner node or root
        inner_nodes.append(node_id)
        first_hidden_layer_biases.append(threshold[node_id])  # this appends bias to the actual node
        actual_node_weight = [0 for j in range(n_features)]
        actual_used_feature = feature[node_id]
        actual_node_weight[actual_used_feature] = 1
        first_hidden_layer_weights.append(actual_node_weight)
    else:
        leaves.append(node_id)
first_hidden_layer_biases = np.array(first_hidden_layer_biases, dtype=np.float64)
first_hidden_layer_weights = np.array(first_hidden_layer_weights, dtype=np.float64)

path_to_leaf = []  # here we store paths to all leaves , each path is a list
for leaf in leaves:
    actual_index = None
    path_leaf = []
    path_leaf.append({leaf: []})
    while (actual_index != 0):
        node_right = list(children_right).index(leaf)
        if node_right is not None:
            path_leaf[leaf].append((node_right, 1))
            actual_index = node_right
        else:
            node_left = list(children_left).index(leaf)
            path_leaf[leaf].append((node_left, -1))
            actual_index = node_left
    path_to_leaf.append(path_leaf)

print(leaves)
print(path_to_leaf)

