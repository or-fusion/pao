

from pyomo.opt import *

safe_termination_conditions = [
                               TerminationCondition.maxTimeLimit,
                               TerminationCondition.maxIterations,
                               TerminationCondition.minFunctionValue,
                               TerminationCondition.minStepLength,
                               TerminationCondition.globallyOptimal,
                               TerminationCondition.locallyOptimal,
                               TerminationCondition.feasible,
                               TerminationCondition.optimal,
                               TerminationCondition.maxEvaluations,
                               TerminationCondition.other,
                              ]