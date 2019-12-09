import numpy as np
from CostFunctions import *
'''
A = np.diag(np.ones(4))
print(A)
A = np.where(A==0,-1,A)
print(A)
w = np.dot(np.linalg.inv(A),np.array([[0.3],[0],[0],[0.7123456]]))
print(w)
print(np.dot(w.reshape(1,-1),np.array([-0.96,0.98,-0.99,-0.99]).reshape(-1,1)))
'''

x = np.array([[0.9],[0.1],[0],[0]]).reshape(1,-1)
#z = x.tolist()[0]
print(x.shape[1])
y = softmax_inverse(x)
z = softmax(y)
print(y)
print(z)

