#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.opt

safe_termination_conditions = [
    pyomo.opt.TerminationCondition.maxTimeLimit,
    pyomo.opt.TerminationCondition.maxIterations,
    pyomo.opt.TerminationCondition.minFunctionValue,
    pyomo.opt.TerminationCondition.minStepLength,
    pyomo.opt.TerminationCondition.globallyOptimal,
    pyomo.opt.TerminationCondition.locallyOptimal,
    pyomo.opt.TerminationCondition.feasible,
    pyomo.opt.TerminationCondition.optimal,
    pyomo.opt.TerminationCondition.maxEvaluations,
    pyomo.opt.TerminationCondition.other
]

optimal_termination_conditions = [
    pyomo.opt.TerminationCondition.globallyOptimal,
    pyomo.opt.TerminationCondition.locallyOptimal,
    pyomo.opt.TerminationCondition.optimal
]

def _check_termination_condition(results):
    # do we want to be more restrictive of termination conditions?
    # do we want to have different behavior for sub-optimal termination?
    if results.solver.termination_condition not in optimal_termination_conditions:
        return False
    else:
        return True
