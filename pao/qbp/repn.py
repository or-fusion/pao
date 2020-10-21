#import math
#import pprint
#from scipy.sparse import coo_matrix
#import numpy as np
#import copy
#import collections.abc

from pao.lbp.repn import LevelValueWrapper, LinearBilevelProblem


class QuadraticLevelRepn(object):       # pragma: no cover

    def __init__(self, nxR, nxZ, nxB):
        super().__init__(nxR, nxZ, nxB)
        self.B = LevelValueWrapper("B") # Quadratic term in objective

    def print(self, *args, nL=0):
        print("Variables:")
        if self.xR.num > 0:
            self.xR.print("Real")
        if self.xZ.num > 0:
            self.xZ.print("Integer")
        if self.xB.num > 0:
            self.xB.print("Binary")

        print("\nObjective:")
        if self.minimize:
            print("  Minimize:")
        else:
            print("  Maximize:")
        self.c.print(*args, nL=nL)
        self.B.print(*args)

        if len(self.A) > 0:
            print("\nConstraints:")
            self.A.print(*args)
            if self.inequalities:
                print("  <=")
            else:
                print("  ==")
            print("   ",self.b)


class QuadraticBilevelProblem(LinearBilevelProblem):            # pragma: no cover
    """
    Let
        x   = [U.xR, U.xZ, U.xB, L.xR, L.xZ, L.xB]'                         # dense column vector
        U.x = [U.xR, U.xZ, U.xB]'                                           # dense column vector
        L.x = [L.xR, L.xZ, L.xB]'                                           # dense column vector
        U.c = [U.c.U.xR, U.c.U.xZ, U.c.U.xB, U.c.L.xR, U.c.L.xZ, U.c.L.xB]  # dense row vector
        L.c = [L.c.U.xR, L.c.U.xZ, L.c.U.xB, L.c.L.xR, L.c.L.xZ, L.c.L.xB]  # dense row vector
        U.A = [U.A.U.xR, U.A.U.xZ, U.A.U.xB, U.A.L.xR, U.A.L.xZ, U.A.L.xB]  # sparse matrix
        L.A = [L.A.U.xR, L.A.U.xZ, L.A.U.xB, L.A.L.xR, L.A.L.xZ, L.A.L.xB]  # sparse matrix
        U.B = [ [U.B.xR.xR, U.B.xR.xZ, U.B.xR.xB],
                [U.B.xZ.xR, U.B.xZ.xZ, U.B.xZ.xB],
                [U.B.xB.xR, U.B.xB.xZ, U.B.xB.xB]]                          # sparse matrix U.x rows, L.x cols
        L.B = [ [L.B.xR.xR, L.B.xR.xZ, L.B.xR.xB],
                [L.B.xZ.xR, L.B.xZ.xZ, L.B.xZ.xB],
                [L.B.xB.xR, L.B.xB.xZ, L.B.xB.xB]]                          # sparse matrix U.x rows, L.x cols

    min_{U.x}   U.c * x + U.x' * U.B * L.x
    s.t.        U.A * x <= U.b                      # Or ==

                where L.x satisifies

                    min_{L.x}   L.c * x + U.x' * L.B * L.x
                    s.t.        L.A * x <= L.b      # Or ==
    """

    def __init__(self, name=None):
        super().__init__(name)

    def add_upper(self, nxR=0, nxZ=0, nxB=0):
        self.U = QuadraticLevelRepn(nxR, nxZ, nxB)
        return self.U

    def add_lower(self, nxR=0, nxZ=0, nxB=0):
        self.L = QuadraticLevelRepn(nxR, nxZ, nxB)
        return self.L

    def print(self):
        if self.name:
            print("# QuadraticBilevelProblem: "+name)
        else:
            print("# QuadraticBilevelProblem: unknown")
        print("")
        print("## Upper Level")
        print("")
        self.U.print("U","L")
        print("")
        print("## Lower Level")
        print("")
        self.L.print("U","L")

    def check(self):
        LinearBilevelProblem.check(self)
        #
        U = self.U
        L = self.L
        #
        # Ncols/Nrows of U.B
        #
        assert ((U.B.xR.xR is None) or (U.B.xR.xR.shape[1] == len(L.xR)) or (U.c.xR.xR.shape[1] == 0)), "Incompatible specification of U.B.xR.xR and L.xR"
        assert ((U.B.xR.xR is None) or (U.B.xR.xR.shape[0] == len(U.xR)) or (U.c.xR.xR.shape[0] == 0)), "Incompatible specification of U.B.xR.xR and U.xR"

        assert ((U.B.xR.xZ is None) or (U.B.xR.xZ.shape[1] == len(L.xZ)) or (U.c.xR.xZ.shape[1] == 0)), "Incompatible specification of U.B.xR.xR and L.xZ"
        assert ((U.B.xR.xZ is None) or (U.B.xR.xZ.shape[0] == len(U.xR)) or (U.c.xR.xZ.shape[0] == 0)), "Incompatible specification of U.B.xR.xR and U.xR"

        assert ((U.B.xR.xB is None) or (U.B.xR.xB.shape[1] == len(L.xB)) or (U.c.xR.xB.shape[1] == 0)), "Incompatible specification of U.B.xR.xR and L.xB"
        assert ((U.B.xR.xB is None) or (U.B.xR.xB.shape[0] == len(U.xR)) or (U.c.xR.xB.shape[0] == 0)), "Incompatible specification of U.B.xR.xR and U.xR"


        assert ((U.B.xZ.xR is None) or (U.B.xZ.xR.shape[1] == len(L.xR)) or (U.c.xZ.xR.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xR"
        assert ((U.B.xZ.xR is None) or (U.B.xZ.xR.shape[0] == len(U.xZ)) or (U.c.xZ.xR.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xZ"

        assert ((U.B.xZ.xZ is None) or (U.B.xZ.xZ.shape[1] == len(L.xZ)) or (U.c.xZ.xZ.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xZ"
        assert ((U.B.xZ.xZ is None) or (U.B.xZ.xZ.shape[0] == len(U.xZ)) or (U.c.xZ.xZ.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xZ"

        assert ((U.B.xZ.xB is None) or (U.B.xZ.xB.shape[1] == len(L.xB)) or (U.c.xZ.xB.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xB"
        assert ((U.B.xZ.xB is None) or (U.B.xZ.xB.shape[0] == len(U.xZ)) or (U.c.xZ.xB.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xZ"


        assert ((U.B.xB.xR is None) or (U.B.xB.xR.shape[1] == len(L.xR)) or (U.c.xB.xR.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xR"
        assert ((U.B.xB.xR is None) or (U.B.xB.xR.shape[0] == len(U.xB)) or (U.c.xB.xR.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xB"

        assert ((U.B.xB.xZ is None) or (U.B.xB.xZ.shape[1] == len(L.xZ)) or (U.c.xB.xZ.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xZ"
        assert ((U.B.xB.xZ is None) or (U.B.xB.xZ.shape[0] == len(U.xB)) or (U.c.xB.xZ.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xB"

        assert ((U.B.xB.xB is None) or (U.B.xB.xB.shape[1] == len(L.xB)) or (U.c.xB.xB.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xB"
        assert ((U.B.xB.xB is None) or (U.B.xB.xB.shape[0] == len(U.xB)) or (U.c.xB.xB.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xB"

    def check_opposite_objectives(self, U, L):
        if id(U.c) == id(L.c) and id(U.B) == id(L.B) and L.minimize ^ U.minimize:
            return True
        U_coef = 1 if U.minimize else -1
        L_coef = 1 if L.minimize else -1

        if not self._equal_nparray(U.c.U.xR, U_coef, L.c.U.xR, L_coef):
            return False
        if not self._equal_nparray(U.c.U.xZ, U_coef, L.c.U.xZ, L_coef):
            return False
        if not self._equal_nparray(U.c.U.xB, U_coef, L.c.U.xB, L_coef):
            return False
        if not self._equal_nparray(U.c.L.xR, U_coef, L.c.L.xR, L_coef):
            return False
        if not self._equal_nparray(U.c.L.xZ, U_coef, L.c.L.xZ, L_coef):
            return False
        if not self._equal_nparray(U.c.L.xB, U_coef, L.c.L.xB, L_coef):
            return False

        if not self._equal_mat(U.c.xR.xR, U_coef, L.c.xR.xR, L_coef):
            return False
        if not self._equal_mat(U.c.xR.xZ, U_coef, L.c.xR.xZ, L_coef):
            return False
        if not self._equal_mat(U.c.xR.xB, U_coef, L.c.xR.xB, L_coef):
            return False
        if not self._equal_mat(U.c.xZ.xR, U_coef, L.c.xZ.xR, L_coef):
            return False
        if not self._equal_mat(U.c.xZ.xZ, U_coef, L.c.xZ.xZ, L_coef):
            return False
        if not self._equal_mat(U.c.xZ.xB, U_coef, L.c.xZ.xB, L_coef):
            return False
        if not self._equal_mat(U.c.xB.xR, U_coef, L.c.xB.xR, L_coef):
            return False
        if not self._equal_mat(U.c.xB.xZ, U_coef, L.c.xB.xZ, L_coef):
            return False
        if not self._equal_mat(U.c.xB.xB, U_coef, L.c.xB.xB, L_coef):
            return False
        return True

    def _equal_mat(self, U, U_coef, L, L_coef):
        if U is None and L is None:
            return True
        if U is None or L is None:
            return False

        Ucoo = U.tocoo()
        Umap = {}
        for i,j,v in zip(Acoo.row, Acoo.col, Acoo.data):
            Umap[i,j] = v
        Lcoo = L.tocoo()
        Lmap = {}
        for i,j,v in zip(Acoo.row, Acoo.col, Acoo.data):
            Lmap[i,j] = v

        for i,j in Umap:
            if math.fabs(Umap[i,j]*U_coef + Lmap[i,j]*L_ceof) > 1e-16:
                return False
        return True


if __name__ == "__main__":              # pragma: no cover
    prob = QuadraticBilevelProblem()
    U = prob.add_upper(3,2,1)
    U.xR.upper_bounds = np.array([1.5, 2.4, 3.1])
    L = prob.add_lower(1,2,3)
    L.xZ.lower_bounds = np.array([1, -2])
    prob.print()

