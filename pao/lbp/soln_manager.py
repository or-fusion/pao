import pyomo.environ as pe


class LBP_SolutionManager(object):

    def __init__(self, real=None, integer=None):
        if real:
            self.multipliers_UxR = real[0]
            self.multipliers_LxR = real[1]
        if integer:
            self.multipliers_UxZ = integer[0]
            self.multipliers_LxZ = integer[1]

    def copy_from_to(self, UxR=None, UxZ=None, UxB=None, LxR=None, LxZ=None, LxB=None, pyomo=None, lbp=None):
        if pyomo is not None:
            return self.copy_from_to(UxR=pyomo.U.xR, UxZ=pyomo.U.xZ, UxB=pyomo.U.xB, LxR=pyomo.L.xR, LxZ=pyomo.L.xZ, LxB=pyomo.L.xB, lbp=lbp)

        # lbp.U
        for j in lbp.U.xR:
            lbp.U.xR.values[j] = sum(pe.value(UxR[v]) * c for v,c in self.multipliers_UxR[j])
        for j in lbp.U.xZ:
            lbp.U.xZ.values[j] = round(sum(pe.value(UxZ[v]) * c for v,c in self.multipliers_UxZ[j]))
        if UxB is not None:
            if len(UxB) == 0:
                # Binaries are at the end of the integers
                nxZ = len(lbp.U.xZ)
                for j in lbp.U.xB:
                    lbp.U.xB.values[j] = round(pe.value(UxZ[nxZ+j]))
            else:
                for j in lbp.U.xB:
                    lbp.U.xB.values[j] = round(pe.value(UxB[j]))
        #
        # TODO: Handle multiple subproblems within Pyomo models
        #
        # lbp.L[i]
        for i in range(len(lbp.L)):
            for j in lbp.L[i].xR:
                lbp.L[i].xR.values[j] = sum(pe.value(LxR[v]) * c for v,c in self.multipliers_LxR[i][j])
            for j in lbp.L[i].xZ:
                lbp.L[i].xZ.values[j] = round(sum(pe.value(LxZ[v]) * c for v,c in self.multipliers_LxZ[i][j]))
            if LxB is not None:
                if len(LxB) == 0:
                    # Binaries are at the end of the integers
                    nxZ = len(lbp.L[i].xZ)
                    for j in lbp.L[i].xB:
                        lbp.L[i].xB.values[j] = round(pe.value(LxZ[nxZ+j]))
                else:
                    for j in lbp.L[i].xB:
                        lbp.L[i].xB.values[j] = round(pe.value(LxB[j]))

    def load_from(self, data):
        # TODO - should we copy the data from a Pyomo model?  or a Pyomo results object?
        #           
        assert (False), "LBP_SolutionManager.load_from() is not implemented yet"
