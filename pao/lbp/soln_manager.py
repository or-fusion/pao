import pyomo.environ as pe


class LBP_SolutionManager(object):

    def __init__(self, multipliers_Ux, multipliers_Lx):
        self.multipliers_Ux = multipliers_Ux
        self.multipliers_Lx = multipliers_Lx

    def copy_from_to(self, UxR=None, UxZ=None, UxB=None, LxR=None, LxZ=None, LxB=None, pyomo=None, lbp=None):
        if pyomo is not None:
            return self.copy_from_to(UxR=pyomo.U.xR, UxZ=pyomo.U.xZ, UxB=pyomo.U.xB, LxR=pyomo.L.xR, LxZ=pyomo.L.xZ, LxB=pyomo.L.xB, lbp=lbp)

        U = lbp.U
        L = lbp.L

        # lbp.U
        for j in range(U.x.nxR):
            U.x.values[j] = sum(pe.value(UxR[v]) * c for v,c in self.multipliers_Ux[j])
        for j in range(U.x.nxZ):
            jj = j+U.x.nxR
            U.x.values[jj] = round(sum(pe.value(UxZ[v]) * c for v,c in self.multipliers_Ux[jj]))
        if UxB is not None:
            if len(UxB) == 0:
                # Binaries are at the end of the integers
                for j in range(U.x.nxB):
                    U.x.values[j+U.x.nxR+U.x.nxZ] = round(pe.value(UxZ[j+U.x.nxZ]))
            else:
                for j in range(U.x.nxB):
                    U.x.values[j+U.x.nxR+U.x.nxZ] = round(pe.value(UxB[j]))
        #
        # TODO: Handle multiple subproblems within Pyomo models
        #
        # lbp.L[i]
        for i in range(len(L)):
            for j in range(L[i].x.nxR):
                L[i].x.values[j] = sum(pe.value(LxR[v]) * c for v,c in self.multipliers_Lx[i][j])
            for j in range(L[i].x.nxZ):
                jj = j+L[i].x.nxR
                L[i].x.values[jj] = round(sum(pe.value(LxZ[v]) * c for v,c in self.multipliers_Lx[i][jj]))
            if LxB is not None:
                if len(LxB) == 0:
                    # Binaries are at the end of the integers
                    for j in range(L[i].x.nxB):
                        L[i].x.values[j+L[i].x.nxR+L[i].x.nxZ] = round(pe.value(LxZ[j+L[i].x.nxZ]))
                else:
                    for j in range(L[i].x.nxB):
                        L[i].x.values[j+L[i].x.nxR+L[i].x.nxZ] = round(pe.value(LxB[j]))

    def load_from(self, data):
        # TODO - should we copy the data from a Pyomo model?  or a Pyomo results object?
        #           
        assert (False), "LBP_SolutionManager.load_from() is not implemented yet"
