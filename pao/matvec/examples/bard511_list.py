
from pao.matvec import *


M = LinearBilevelProblem()

U = M.add_upper(nxR=1)
U.xR.lower_bounds = [0]
U.c.U.xR = [1]
U.c.L.xR = [-4]

L = M.add_lower(nxR=1)
L.xR.lower_bounds = [0]
L.c.L.xR = [1]

L.A.U.xR = [(0,0,-1), (1,0,-2), (2,0,2), (3,0,-3)]
L.A.L.xR = [(0,0,1),  (1,0,1),  (2,0,1), (3,0,2)]
L.b = [-3, 0, 12, -4]


opt = BilevelSolver('pao.bilevel.ld')
opt.solve(M)


M.print()
