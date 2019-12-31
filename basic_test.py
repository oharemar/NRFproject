import numpy as np
from CostFunctions import *
from statistics import mode
import random

'''
A = np.diag(np.ones(4))
print(A)
A = np.where(A==0,-1,A)
print(A)
w = np.dot(np.linalg.inv(A),np.array([[0.3],[0],[0],[0.7123456]]))
print(w)
print(np.dot(w.reshape(1,-1),np.array([-0.96,0.98,-0.99,-0.99]).reshape(-1,1)))
'''

x = np.array([[0.6],[0.2],[0.1],[0.1]]).reshape(1,-1)
x = np.array([[1,0,1],[0,0,1],[1,2,3]])#.reshape(1,-1)

nulls = np.zeros(x.shape)
print(nulls)
print(x.tolist())

for j in x.tolist():
    try:
        md = mode(j)
        print(md)
    except:
        md = random.choice(j)
        print(md)

#X = [1 if number in [0,1,2] else 2 for number in range(10)]

#print(X)
