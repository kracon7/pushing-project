'''
Compute the pressure map based on the mass distribution and the bottom contact
'''

import os
import sys
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

M = np.array([1, 2, 3, 1]).astype('double')
X = np.array([0, 1, 2, 3]).astype('double')

# construct A
A = [np.ones_like(M)]
for i in range(M.shape[0]):
	row = []
	# construct row
	for j in range(M.shape[0]):
		row.append(X[j] - X[i])
	
	A.append(np.array(row))

A = np.stack(A)
B = np.stack([A[:,0], A[:,2], A[:,3]]).T

projection = la.pinv(B.T @ B) @ (B.T @ A)

print(projection)