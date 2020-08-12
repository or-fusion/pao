
# Example 2.7 from
#
# Near-Optimal Robust Bilevel Optimization
#   M. Besancon, M. F. Anjos and L. Brotcorne
#   arXiv:1908.04040v5 (2019)

from pao.matvec import *


M = LinearBilevelProblem()

# Variables
U = M.add_upper(nxR=1)
U.xR.lower_bounds = [0]

L = M.add_lower(nxR=1)

# Objectives
U.c.U.xR = [1]

L.c.L.xR = [1]
L.minimize = False

# Constraints
U.A.U.xR = [(0,0,-1/10)]
U.A.L.xR = [(0,0,-1)]
U.b = [-1]

L.A.U.xR = [(0,0,-1/10)]
L.A.L.xR = [(0,0,1)]
L.b = [1]



opt = BilevelSolver('pao.bilevel.blp_global')
opt.solve(M)


M.print()
