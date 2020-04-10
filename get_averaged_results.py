import numpy as np
from sklearn.model_selection import KFold
from collections import Counter


def collect_and_average_results(X,y,k=5):
    results = {}