import pyomo.environ as pe


class LBP_SolutionManager(object):

    def __init__(self, multipliers):
        self.multipliers = multipliers

    def copy_from_to(self, LxR=None, LxZ=None, LxB=None, pyomo=None, lbp=None):
        if pyomo is not None:
            #
            # TODO - generalize this logic to multi-level models and models with multiple subproblems
            #
            LxR = {}
            LxR[lbp.U.id] = pyomo.U.xR
            LxR[lbp.U.LL[0].id] = pyomo.L.xR
            LxZ = {}
            LxZ[lbp.U.id] = pyomo.U.xZ
            LxZ[lbp.U.LL[0].id] = pyomo.L.xZ
            LxB = {}
            LxB[lbp.U.id] = pyomo.U.xB
            LxB[lbp.U.LL[0].id] = pyomo.L.xB
            return self.copy_from_to(LxR=LxR, LxZ=LxZ, LxB=LxB, lbp=lbp)

        for L in lbp.levels():
            multipliers = self.multipliers[L.id]
            for j in range(L.x.nxR):
                L.x.values[j] = sum(pe.value(LxR[L.id][v]) * c for v,c in multipliers[j])
            for j in range(L.x.nxZ):
                jj = j+L.x.nxR
                L.x.values[jj] = round(sum(pe.value(LxZ[L.id][v]) * c for v,c in multipliers[jj]))
            if LxB is not None:
                if len(LxB) == 0:
                    # Binaries are at the end of the integers
                    for j in range(L.x.nxB):
                        L.x.values[j+L.x.nxR+L.x.nxZ] = round(pe.value(LxZ[L.id][j+L.x.nxZ]))
                else:
                    for j in range(L.x.nxB):
                        L.x.values[j+L.x.nxR+L.x.nxZ] = round(pe.value(LxB[L.id][j]))

    def load_from(self, data):      # pragma: no cover
        # TODO - should we copy the data from a Pyomo model?  or a Pyomo results object?
        #           
        assert (False), "LBP_SolutionManager.load_from() is not implemented yet"
