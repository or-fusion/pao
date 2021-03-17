from munch import Munch
import pyomo.environ as pe
from .repn import LinearMultilevelProblem, QuadraticMultilevelProblem


class LMP_SolutionManager(object):

    def __init__(self, multipliers):
        self.multipliers = multipliers

    def copy(self, From=None, To=None):
        if type(From) is pe.ConcreteModel and type(To) in [LinearMultilevelProblem,QuadraticMultilevelProblem]:
            #
            # TODO - generalize this logic to multi-level models and models with multiple subproblems
            #
            LxR = {}
            LxZ = {}
            LxB = {}
            LxR[To.U.id] = From.U.xR
            LxZ[To.U.id] = From.U.xZ
            LxB[To.U.id] = From.U.xB
            for i in range(len(From.L)):
                LxR[To.U.LL[i].id] = From.L[i].xR
                LxZ[To.U.LL[i].id] = From.L[i].xZ
                LxB[To.U.LL[i].id] = From.L[i].xB
            return self.copy(From=Munch(LxR=LxR, LxZ=LxZ, LxB=LxB), To=To)

        elif type(From) is Munch and type(To) in [LinearMultilevelProblem,QuadraticMultilevelProblem]:
            for L in To.levels():
                multipliers = self.multipliers[L.id]
                for j in range(L.x.nxR):
                    L.x.values[j] = sum(pe.value(From.LxR[L.id][v]) * c for v,c in multipliers[j])
                for j in range(L.x.nxZ):
                    jj = j+L.x.nxR
                    L.x.values[jj] = round(sum(pe.value(From.LxZ[L.id][v-L.x.nxR]) * c for v,c in multipliers[jj]))
                if From.get('LxB',None) is not None:
                    if len(From.LxB) == 0:
                        # Binaries are at the end of the integers
                        for j in range(L.x.nxB):
                            L.x.values[j+L.x.nxR+L.x.nxZ] = round(pe.value(From.LxZ[L.id][j+L.x.nxZ]))
                    else:
                        for j in range(L.x.nxB):
                            L.x.values[j+L.x.nxR+L.x.nxZ] = round(pe.value(From.LxB[L.id][j]))

        else:
            raise RuntimeError("Unexpected types: From=%s To=%s" % (str(type(From)), str(type(To))))

    def load_from(self, data):      # pragma: no cover
        # TODO - should we copy the data from a Pyomo model?  or a Pyomo results object?
        #           
        assert (False), "LMP_SolutionManager.load_from() is not implemented yet"
