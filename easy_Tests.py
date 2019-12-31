import numpy as np


A = np.diag(np.ones(4))
print(A)
A = np.where(A==0,-1,A)
inverse_A = np.linalg.inv(A)
print(inverse_A)