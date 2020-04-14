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
from helpful_functions import *
from sklearn.model_selection import KFold
from statistics import mean
from statistics import stdev
import json
from NeuralRandomForest import NeuralRandomForest


# budeme zkoušet na messidor a vehicle silhouette pomocí 5 fold cross validation, zajímá nás train and test accuracy
# a train and test macro avg F1-score
"""LEARNING RATE SENSITIVITY"""

"""nrf analytic Weight adam"""
"""nrf basic adam"""
"""nrf extra layer adam"""
"""nrf extra layer """
