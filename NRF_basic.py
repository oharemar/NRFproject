from ANN import Network
import pandas as pd
import numpy as np
import copy


class NeuralTreeBasic():

    def __init__(self, decision_tree = None, class_probs = None):

        self.decision_tree = decision_tree
        self.network = None # corresponding neural network classifier
        self.class_probs = class_probs # normalized probabilities of classes in individual leaves
        self.weights = []
        self.biases = []
        self.inner_nodes = None
        self.leaves = None

        self.initialize_first_hidden_layer()
        self.initialize_second_hidden_layer()
        self.initialize_output_layer()
        self.create_NN()


    def initialize_first_hidden_layer(self):

        n_nodes = self.decision_tree.tree_.node_count
        children_left = self.decision_tree.tree_.children_left  # z toho udělám first hidden layer of NRF, second hidden layer of NRF
        children_right = self.decision_tree.tree_.children_right  # z toho udělám first hidden layer of NRF, second hidden layer of NRF
        feature = self.decision_tree.tree_.feature  # z toho udělám first hidden layer of NRF
        threshold = self.decision_tree.tree_.threshold  # z toho udělám first hidden layer of NRF
        n_features = self.decision_tree.n_features_

        """first hidden layer - corresponds to inner nodes
        ----neurons in this layer has initial weights [0,0,...,0,1,0,...,0], where only one 1 corresponds to chosen feature in DT node split
        ----initial biases are thresholds of node split function
        """
        inner_nodes = []
        leaves = []
        first_hidden_layer_weights = []
        first_hidden_layer_biases = []
        #first_hidden_layer_weights = np.zeros(n_nodes - n_leaves,n_features) # will have #neurons = #inner nodes of DT
        #first_hidden_layer_biases = np.zeros(n_nodes - n_leaves, 1)  # will have #neurons = #inner nodes of DT

        for node_id in range(n_nodes):
            if (children_left[node_id] != children_right[node_id]): # if satisfied, we are in inner node or root
                inner_nodes.append(node_id)
                first_hidden_layer_biases.append(threshold[node_id]) # this appends bias to the actual node
                actual_node_weight = [0 for j in range(n_features)]
                actual_used_feature = feature[node_id]
                actual_node_weight[actual_used_feature] = 1
                first_hidden_layer_weights.append(actual_node_weight)
            else:
                leaves.append(node_id)
        first_hidden_layer_biases = np.array(first_hidden_layer_biases,dtype = np.float64).reshape((len(first_hidden_layer_biases),1))
        first_hidden_layer_weights = np.array(first_hidden_layer_weights,dtype = np.float64)

        self.weights.append(first_hidden_layer_weights)
        self.biases.append(first_hidden_layer_biases)
        self.inner_nodes = inner_nodes
        self.leaves = leaves

    def initialize_second_hidden_layer(self):
        """first hidden layer has same number of neurons as number of leaves in DT"""
        """for each leaf we need to find exact path from the root that could reach this leaf --- usage of decision path method"""
        children_left = list(self.decision_tree.tree_.children_left)  # z toho udělám first hidden layer of NRF, second hidden layer of NRF
        children_right = list(self.decision_tree.tree_.children_right)  # z toho udělám first hidden layer of NRF, second hidden layer of NRF
        path_to_leaf = [] # here we store paths to all leaves , each path is a list
        for leaf in self.leaves:
            actual_index = copy.deepcopy(leaf) # we search the tree upside down , starting from the leaves towards the root
            path_leaf = {leaf:[]}
            while (actual_index != 0):
                try:
                    node_right = children_right.index(actual_index) # tries to find actual node in the right children, if it is not successful, it goes straight to the except part
                except ValueError:
                    node_left = children_left.index(actual_index) # this will be conducted if actual node belongs to the left children
                    path_leaf[leaf].append((node_left,-1))
                    actual_index = node_left
                else:
                    path_leaf[leaf].append((node_right, 1)) # this will be conducted if actual node belongs to the right children
                    actual_index = node_right
            path_leaf[leaf].reverse() # reverse the order of elements in list, first element corresponds to the start in the root
            path_to_leaf.append(path_leaf)

        second_hidden_layer_weights = []
        second_hidden_layer_biases = []
        for path_leaf in path_to_leaf:
            second_hidden_layer_weights.append([]) # each row corresponds to weights for one leaf
            actual_leaf = list(path_leaf.keys())[0]
            second_hidden_layer_biases.append(-len(path_leaf[actual_leaf])+0.5)
            nodes_in_path = [y[0] for y in path_leaf[actual_leaf]]
            for node in self.inner_nodes:
                try:
                    node_index = nodes_in_path.index(node)
                except ValueError:
                    second_hidden_layer_weights[-1].append(0) # weight is 0 if inner node is not part of the path from root to leaf
                else:
                    weight = path_leaf[actual_leaf][node_index][1]
                    second_hidden_layer_weights[-1].append(weight)
        second_hidden_layer_weights = np.array(second_hidden_layer_weights,dtype=np.float64)
        second_hidden_layer_biases = np.array(second_hidden_layer_biases,dtype=np.float64).reshape((len(second_hidden_layer_biases),1))

        self.weights.append(second_hidden_layer_weights)
        self.biases.append(second_hidden_layer_biases)

    def initialize_output_layer(self): # in basic version
        weights = np.zeros((self.decision_tree.n_classes_, len(self.leaves)),dtype=np.float64)
        for label in range(self.decision_tree.n_classes_):
            label_weights = [cls[label] for cls in self.class_probs]
            label_weights = np.array(label_weights,dtype=np.float64)
            weights[label,:] = label_weights

        biases = np.zeros((self.decision_tree.n_classes_,1),dtype=np.float64) # initial biases are zero

        self.weights.append(weights)
        self.biases.append(biases)

    def create_NN(self):

        self.network = Network(sizes = [self.decision_tree.n_features_,
                                        len(self.inner_nodes),
                                        len(self.leaves),
                                        self.decision_tree.n_classes_],biases=self.biases,weights=self.weights)

    """now will come methods for training, prediction etc., but it could be easily obtained from already existing methods of Network()"""


























