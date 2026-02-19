import numpy as np
import scipy.linalg as la

A = np.random.rand(5,5)
w, _ = la.eig(A)
print("Eigenvalues:", w)