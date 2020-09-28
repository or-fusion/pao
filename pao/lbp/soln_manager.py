import pyomo.environ as pe


class LBP_SolutionManager(object):

    def __init__(self, real=None, integer=None):
        if real:
            self.multipliers_UxR = real[0]
            self.multipliers_LxR = real[1]
        if integer:
            self.multipliers_UxZ = integer[0]
            self.multipliers_LxZ = integer[1]

    def copy_from_to(self, M, repn):
        for j in repn.U.xR:
            repn.U.xR.values[j] = sum(pe.value(M.U.xR[v]) * c for v,c in self.multipliers_UxR[j])
        for j in repn.U.xZ:
            repn.U.xZ.values[j] = sum(pe.value(M.U.xZ[v]) * c for v,c in self.multipliers_UxZ[j])
        for j in repn.U.xB:
            repn.U.xB.values[j] = pe.value(M.U.xB[j])
        #
        # TODO - generalize to multiple sub-problems
        #
        for i in range(len(repn.L)):
            for j in repn.L[i].xR:
                repn.L[i].xR.values[j] = sum(pe.value(M.L.xR[v]) * c for v,c in self.multipliers_LxR[i][j])
            for j in repn.L[i].xZ:
                repn.L[i].xZ.values[j] = sum(pe.value(M.L.xZ[v]) * c for v,c in self.multipliers_LxZ[i][j])
            for j in repn.L[i].xB:
                repn.L[i].xB.values[j] = pe.value(M.L.xB[j])

    def load_from(self, data):
        # TODO - should we copy the data from a Pyomo model?  or a Pyomo results object?
        #           
        assert (False), "LBP_SolutionManager.load_from() is not implemented yet"
