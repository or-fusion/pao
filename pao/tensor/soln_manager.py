import pyomo.environ as pe


class LBP_SolutionManager(object):

    def __init__(self, multipliers_U, multipliers_L):
        self.multipliers_U = multipliers_U
        self.multipliers_L = multipliers_L

    def copy_from_to(self, M, repn):
        for j in repn.U.xR:
            repn.U.xR.values[j] = sum(pe.value(M.U.xR[v]) * c for v,c in self.multipliers_U[j])
        for j in repn.U.xZ:
            repn.U.xZ.values[j] = pe.value(M.U.xZ[j])
        for j in repn.U.xB:
            repn.U.xB.values[j] = pe.value(M.U.xB[j])
        #
        # TODO - generalize to multiple sub-problems
        #
        for i in range(len(repn.L)):
            for j in repn.L[i].xR:
                repn.L[i].xR.values[j] = sum(pe.value(M.L.xR[v]) * c for v,c in self.multipliers_L[i][j])
            for j in repn.L[i].xZ:
                repn.L[i].xZ.values[j] = pe.value(M.L.xZ[j])
            for j in repn.L[i].xB:
                repn.L[i].xB.values[j] = pe.value(M.L.xB[j])

