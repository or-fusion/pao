import scipy.sparse as sp

help(sp)

B = sp.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
A = sp.coo_matrix([(0,0,1), (0,1,2), (1,2,3), (2,0,4), (2,2,5)])
print(A)
