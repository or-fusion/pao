
import numpy as np
from scipy.sparse import coo_matrix
from pao.tensor import *


M = LinearBilevelProblem()

U = M.add_upper(nxR=1)
U.xR.lower_bounds = np.array([0])
U.c.U.xR = np.array([1])
U.c.L.xR = np.array([-4])

L = M.add_lower(nxR=1)
L.xR.lower_bounds = np.array([0])
L.c.L.xR = np.array([1])

L.A.U.xR = coo_matrix((np.array([-1, -2, 2, -3]),
                      (np.array([0, 1, 2, 3]),
                       np.array([0, 0, 0, 0]))))
L.A.L.xR = coo_matrix((np.array([-1, 1, 1, 2]),
                      (np.array([0, 1, 2, 3]),
                       np.array([0, 0, 0, 0]))))
L.b = np.array([-3, 0, 12, -4])


opt = BilevelSolver('pao.bilevel.blp_global')
opt.solve(M)


M.print()
