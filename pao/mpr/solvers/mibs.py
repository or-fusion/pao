#
# A solver interface to MibS
#
# A branch-and-cut algorithm for mixed integer bilevel linear optimization problems and its implementation
# S Tahernejad, TK Ralphs, ST DeNegre
# Mathematical Programming Computation 12 (4), 529-568
#
# TODO - Citations
#
import os
import sys
import time
import numpy as np
import pyutilib
import pyomo.environ as pe
import pyomo.opt
from pyomo.common.config import ConfigBlock, ConfigValue
#from pyomo.mpec import ComplementarityList, complements

import pao.common
from ..solver import Solver, LinearMultilevelSolverBase, LinearMultilevelResults
from ..repn import LinearMultilevelProblem
from ..convert_repn import convert_to_standard_form
from . import pyomo_util
#from .reg import create_model_replacing_LL_with_kkt


@Solver.register(
        name='pao.mpr.MIBS',
        doc='PAO solver for Multilevel Problem Representations using the COIN-OR MibS solver by Tahernejad, Ralphs, and DeNegre (2020).')

class LinearMultilevelSolver_MIBS(LinearMultilevelSolverBase):
    """
    PAO MibS solver for linear MPRs: pao.mpr.MIBS
    """
    config = LinearMultilevelSolverBase.config()
    config.declare('executable', ConfigValue(
        default='mibs',
        description="The executable used for MibS.  (default is mibs)"
        ))
    config.declare('param_file', ConfigValue(
        default=None,
        description="The parameter file used to configure MibS.  (default is None)"
        ))

    def __init__(self, **kwds):
        super().__init__(name='pao.mpr.MIBS')

    def check_model(self, model):
        #
        # Confirm that the LinearMultilevelProblem is well-formed
        #
        assert (type(model) is LinearMultilevelProblem), "Solver '%s' can only solve a linear multilevel problem" % self.name
        model.check()
        #
        # Confirm that this is a bilevel problem with just one lower-level
        #
        for L in model.U.LL:
            assert (len(L.LL) == 0), "Can only solve bilevel problems"
        assert (len(model.U.LL) == 1), "Can only solve a bilevel problem with a single lower-level"

    def solve(self, model, **options):
        #
        # Error checks
        #
        self.check_model(model)
        #
        # Process keyword options
        #
        self._update_config(options)
        #
        # Start clock
        #
        start_time = time.time()

        self.standard_form, soln_manager = convert_to_standard_form(model, inequalities=True)

        #
        # Write the MPS file and MIBS auxilliary file
        #
        # TODO - Make these temporary files
        #
        M = self.create_mibs_model(model, "mibs.mps", "mibs.aux")

        cmd = [ self.config['executable'], '-Alps_instance', 'mibs.mps', '-MibS_auxiliaryInfoFile', 'mibs.aux']
        if self.config['param_file'] is not None:
            cmd.append('-param')
            cmd.append('mibs.par')

        ans = pao.common.run_shellcmd(cmd, tee=self.config['tee'])
        os.remove("mibs.mps")
        os.remove("mibs.aux")
        #print("RC", ans.rc)
        #print("LOG", ans.log)

        results = self._initialize_results(ans, model)
        results.check_optimal_termination()

        results.solver.wallclock_time = time.time() - start_time
        return results

    def _initialize_results(self, ans, M):
        #
        # Default value is zero
        #
        M.U.x.values = [0]*len(M.U.x)
        M.U.LL.x.values = [0]*len(M.U.LL.x)
        #
        # Results
        #
        results = pao.common.Results()
        solv = results.solver
        solv.termination_condition = pao.common.TerminationCondition.unknown
        solv.name = self.config['executable']
        solv.rc = ans.rc
        #
        # Parse Log
        #
        # TODO - Handle errors
        #
        if ans.rc == 0:
            state = 0
            for line in ans.log.split("\n"):
                #print(state, line)
                line = line.strip()
                if state == 0:
                    if line.startswith("Optimal solution:"):
                        solv.termination_condition = pao.common.TerminationCondition.optimal
                        state = 1
                elif state == 1:
                    solv.best_feasible_objective = float(line.split("=")[1])
                    state = 2
                elif state == 2:
                    if line.startswith("Number"):
                        state = 3
                        continue
                    name, _, val = line.split(" ")
                    vname, index = name[:-1].split("[")
                    #print(vname, index, val)
                    if vname == "x":
                        M.U.x.values[int(index)] = float(val)
                    else:
                        M.U.LL.x.values[int(index)] = float(val)
        return results

    def _debug(self, M):    # pragma: no cover
        for j in M.U.xR:
            print("U",j,pe.value(M.U.xR[j]))
        for j in M.L.xR:
            print("L",j,pe.value(M.L.xR[j]))
        for j in M.kkt.lam:
            print("lam",j,pe.value(M.kkt.lam[j]))
        for j in M.kkt.nu:
            print("nu",j,pe.value(M.kkt.nu[j]))





    def create_mibs_model(self, repn, mps_filename, aux_filename):
        """
        TODO - Document this transformation
        """
        U = repn.U
        L = repn.U.LL[0]

        #---------------------------------------------------
        # Create Pyomo model
        #---------------------------------------------------

        M = pe.ConcreteModel()
        M.U = pe.Block()
        M.L = pe.Block()

        # upper-level variables
        pyomo_util.add_variables(M.U, U)
        # lower-level variables
        pyomo_util.add_variables(M.L, L)

        # objective
        e = pyomo_util.dot(U.c[U], U.x, num=1) + U.d
        e += pyomo_util.dot(U.c[L], L.x, num=1)
        M.o = pe.Objective(expr=e)

        # upper-level constraints
        pyomo_util.add_linear_constraints(M.U, U.A, U, L, U.b, U.inequalities)
        # lower-level constraints
        pyomo_util.add_linear_constraints(M.L, L.A, U, L, L.b, L.inequalities)

        #---------------------------------------------------
        # Write files
        #---------------------------------------------------
        # TODO - get variable mapping information
        #
        M.write("tmp.mps")
        #
        # Add a space after "BOUND"
        #
        with open("tmp.mps", "r") as INPUT:
            with open(mps_filename, "w") as OUTPUT:
                for line in INPUT:
                    OUTPUT.write(line.replace("BOUND ", "BOUND  "))
        os.remove("tmp.mps")

        with open(aux_filename, "w") as OUTPUT:
            # Num lower-level variables
            OUTPUT.write("N {}\n".format(len(L.x)))
            # Num lower-level constraints
            OUTPUT.write("M {}\n".format(L.b.size))
            # Indices of lower-level variables
            nx_upper = len(U.x)
            for i in range(len(L.x)):
                OUTPUT.write("LC {}\n".format(i+nx_upper))
            # Indices of lower-level constraints
            nc_upper = U.b.size
            for i in range(L.b.size):
                OUTPUT.write("LR {}\n".format(i+nc_upper))
            # Coefficients for lower-level objective
            for i in range(len(L.x)):
                OUTPUT.write("LO {}\n".format(L.c[L][i]))
            # Lower-level objective sense
            if L.minimize:
                OUTPUT.write("OS 1\n")
            else:
                OUTPUT.write("OS -1\n")
        
        return M

pao.common.SolverAPI._generate_solve_docstring(LinearMultilevelSolver_MIBS)
