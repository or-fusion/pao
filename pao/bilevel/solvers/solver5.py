#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
pao.bilevel.plugins.solver5

Declare the pao.bilevel.norbip solver.

"Near-Optimal Robust Bilevel Optimization"
M. Besancon, M.F. Anjos and L. Brotcorne
arXiv:1908.04040v5
Nov, 2019.

TODO: Currently handling linear subproblem; need to extend to
convex subproblem duality
TODO: This is incomplete, currently worked on by She'ifa

"""



import cdd
import time
import itertools
import pyutilib.misc
import pyomo.opt
from math import inf
import pyomo.common
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.mpec import complements, ComplementarityList, Complementarity
from pyomo.gdp import Disjunct, Disjunction
from pao.bilevel.solvers.solver_helpers import _check_termination_condition
from pao.bilevel.plugins.collect import BilevelMatrixRepn
from pao.bilevel.components import SubModel, varref, dataref
from pao.bilevel.solvers.solver2 import BilevelSolver2
from pyomo.core import TransformationFactory, minimize, maximize, Block, Constraint, Objective, Var, Reals, Binary, Integers, Any, Param, NonNegativeIntegers, NonNegativeReals
from pyomo.core.expr.numvalue import value
import numpy as np
from numpy import array, dot
from pyomo.common.modeling import unique_component_name


@pyomo.opt.SolverFactory.register('pao.bilevel.norvep',
                                  doc='Solver for near-optimal vertex enumeration procedure')
class BilevelSolver5(pyomo.opt.OptSolver):
    """
    A solver that performs near-optimal robustness for bilevel programs
    """

    def __init__(self, **kwds):
        kwds['type'] = 'pao.bilevel.norvep'
        pyomo.opt.OptSolver.__init__(self, **kwds)
        self._metasolver = True

    def _presolve(self, *args, **kwds):
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)
        self.results = []

    def _apply_solver(self):
        start_time = time.time()

        # matrix representation for bilevel problem
        matrix_repn = BilevelMatrixRepn(self._instance)

        # each lower-level problem
        submodel = [block for block in self._instance.component_objects(SubModel)][0]
        if len(submodel) != 1:
            raise Exception('Problem encountered, this is not a valid bilevel model for the solver.')
        self._instance.reclassify_component_type(submodel, Block)
        #varref(submodel)
        #dataref(submodel)

        all_vars = {key: var for (key, var) in matrix_repn._all_vars.items()}

        # get the variables that are fixed for the submodel (lower-level block)
        fixed_vars = {key: var for (key, var) in matrix_repn._all_vars.items() if key in matrix_repn._fixed_var_ids[submodel.name]}
        
        #Is there a way to get integer, continuous, etc for the upper level rather than lumping them all into fixed?

        # continuous variables in SubModel
        c_vars = {key: var for (key, var) in matrix_repn._all_vars.items() if key in matrix_repn._c_var_ids - fixed_vars.keys()}

        # binary variables in SubModel SHOULD BE EMPTY FOR THIS SOLVER
        b_vars = {key: var for (key, var) in matrix_repn._all_vars.items() if key in matrix_repn._b_var_ids - fixed_vars.keys()}
        if len(b_vars)!= 0:
            raise Exception('Problem encountered, this is not a valid bilevel model for the solver. Binary variables present!')
            
        # integer variables in SubModel SHOULD BE EMPTY FOR THIS SOLVER
        i_vars = {key: var for (key, var) in matrix_repn._all_vars.items() if key in matrix_repn._i_var_ids - fixed_vars.keys()}
        if len(i_vars) != 0:
            raise Exception('Problem encountered, this is not a valid bilevel model for the solver. Integer variables present!')
            
        # get constraint information related to constraint id, sign, and rhs value
        sub_cons = matrix_repn._cons_sense_rhs[submodel.name]
        
        cons= matrix_repn._cons_sense_rhs[self._instance.name]
        
        # construct the high-point problem (LL feasible, no LL objective)
        # s0 <- solve the high-point
        # if s0 infeasible then return high_point_infeasible
        xfrm = TransformationFactory('pao.bilevel.highpoint')
        xfrm.apply_to(self._instance)
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:
            solver = 'gurobi'

        for c in self._instance.component_objects(Block, descend_into=False): 
            if 'hp' in c.name:
            #if '_hp' in c.name:
                c.activate()
                with pyomo.opt.SolverFactory(solver) as opt:
                    self.results.append(opt.solve(c,
                                              tee=self._tee,
                                              timelimit=self._timelimit))
                _check_termination_condition(self.results[-1])
                c.deactivate()
        print('Solution to the Highpoint Relaxation')
        for _, var in all_vars.items():
             var.pprint()
        
        # s1 <- solve the optimistic bilevel (linear/linear) problem (call solver3)
        # if s1 infeasible then return optimistic_infeasible'
        with pyomo.opt.SolverFactory('pao.bilevel.blp_global') as opt:
            opt.options.solver = solver
            self.results.append(opt.solve(self._instance,tee=self._tee,timelimit=self._timelimit))
        _check_termination_condition(self.results[-1])
        
        print('Solution to the Optimistic Bilevel')
        for _, var in all_vars.items():
             var.pprint()
        #debugged to here, solving toy problem correctly 
        # sk <- solve the dual adversarial  problem
        # if infeasible then return dual_adversarial_infeasible

        # Collect the vertices solutions for the dual adversarial problem
        
        #Collect up the matrix B and the vector d for use in all adversarial feasibility problems 
        B=np.array([])
    
        for _, var in c_vars.items():
            (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
            B=np.hstack((B,np.transpose(np.array(A))))
        d=...
        self._instance.Vertices=Param(cons.keys()*NonNegativeIntegers*cons.keys()) #(k-th subproblem,l-th vertex, dimension of alpha)
        self._instance.VerticesB=Param(cons.keys()*NonNegativeIntegers) #(k-th subproblem, l-th vertex)
        self._instance.adversarial=Block(Any)
        #Add Adversarial blocks
        for _cid in cons.items(): # <for each constraint in the upper-level problem>
            ad=self._instance.adversarial[_cid] #shorthand
            ad.alpha=Var(cons.keys(),within=NonNegativeReals) #cons.keys() because it's a dual variable on the upper level
            ad.beta=Var(within=NonNegativeReals)
            Hk=np.array([])
            
            for _,var in c_vars.items():    
                (A,A_q,sign,b) = matrix_repn.coef_matrices(self._instance,var) #ERROR
                Hk=np.vstack((Hk,np.array(A[_cid])))
            '''    
            ad.cons=Constraint(c_vars.keys()) #B^Talpha+beta*d>= H_k, v-dimension constraints so index by c_vars
            lhs_expr = {key: 0. for key in c_vars.keys()}
            rhs_expr = {key: 0. for key in c_vars.keys()}
            for _vid, var in c_vars.items():
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A #+ dot(A_q.toarray(), _fixed)
                
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                d=np.array(C)
                lhs_expr[_vid] =float(C[_vid])*ad.beta #LHS=beta*d
                
                (A,A_q,sign,b)=matrix_repon.coef_matrices(self._instance,var)
                Hk=np.vstack((Hk,np.array(A[_cid])))
                
                for _cid2 in cons.keys():
                    idx = list(sub_cons.keys()).index(_cid2)
                    lhs_expr[_vid] += float(coef[idx])*ad.alpha[_cid2]  #Is this giving me B^Talpha or B*alpha??
                
                rhs_expr[_vid] = float(Hk[_vid])
                expr = lhs_expr[_vid] >= rhs_expr[_vid]
                if not type(expr) is bool:
                    ad.cons[_vid] = expr
                else:
                    ad.cons[_vid] = Constraint.Skip
                    
            ad.Obj=Objective(expr=0) #THIS IS A FEASIBILITY PROBLEM
            with pyomo.opt.SolverFactory(solver) as opt:
                    self.results.append(opt.solve(ad,
                                              tee=self._tee,
                                              timelimit=self._timelimit))
            _check_termination_condition(self.results[-1]) 
            '''
            Bd=np.hstack((np.transpose(B),d))
            mat=cdd.Matrix(np.hstack((-Hk,Bd)),number_type='float')
            
            mat.rep_type=cdd.RepType.INEQUALITY
            poly=cdd.Polyhedron(mat)
            ext=poly.get_generators()
            extreme=np.array(ext)
            print(ext)
            (s,t)=extreme.shape
            l=1
            for i in range(0,s): 
                if extreme[0,i]==1:
                    for j in range(1,t-1): #Need to loop over extreme 1 to t-1 and link those to the cons.keys for alpha? 
                        self._instance.Vertices[(_cid,l,j)]=extreme[i,j] #Vertex l of the k-th polytope
                    self._instance.VerticesB[(_cid,l)]=extreme[i,t-1] 
                    l=+1
            
            
            '''
            for _vid, var in c_vars.items():
                    (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                    coef = A #+ dot(A_q.toarray(), _fixed)
                    for _cid in sub_cons.keys():
                        idx = list(sub_cons.keys()).index(_cid)
                        lhs_expr_a[_cid] += float(coef[idx])*m._iter_c_tilde[_vid]

                for var in {**b_vars, **i_vars, **fixed_vars}.values():
                    (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                    coef = A #+ dot(A_q.toarray(), _fixed)
                    for _cid in sub_cons.keys():
                        idx = list(sub_cons.keys()).index(_cid)
                        ref = m._map[var]
                        rhs_expr_a[_cid] += -float(coef[idx])*ref

                for _cid, b in sub_cons.items():
                    (_,sign) = _cid
                    rhs_expr_a[_cid] += b
                    if sign=='l' or sign=='g->l':
                        expr = lhs_expr_a[_cid] <= rhs_expr_a[_cid]
                    if sign=='e' or sign=='g':
                        raise Exception('Problem encountered, this problem is not in standard form.')
                    if not type(expr) is bool:
                        lower_bounding_master.KKT_tight2a[_cid] = expr
                    else:
                        lower_bounding_master.KKT_tight2a[_cid] = Constraint.Skip
        '''
        
        # Solving the full problem sn0
        # Return the sn0 solution
    '''
    
    m.alpha=Var(m.mlset,within=NonNegativeReals)
    m.beta=Var(within=NonNegativeReals)
    
    m.Verticesalpha=Param(Any) #(k,l) is vertex l of the k-th subproblem polyhedron
    m.Verticesbeta=Param(Any) 
    m.Adversarial=Block(m.muset)
    
    
    
    def c3(m,i,k):
        value=sum(m.B[j,i]*m.alpha[j] for j in m.mlset) + m.b[i]*m.beta
        value<=m.H[i,k]
        return value
    
    #Get Bd=[B^T|d] matrix for all adversarial problems using numpy
    
    Bd=np.hstack((np.transpose(B_array),d_array))
    
    for k in m.muset:
        
        #Check Adversarial feasibility
        m.Adversarial[k].alpha=Variable(m.mlset,within=NonNegativeReals)
        m.Adversarial[k].beta=Variable(within=NonNegativeReals) #Scalar
        m.Adversarial[k].c3=Constraint(m.muset,rule=c3)
        results=opt.solve(m.Adversarial[k])
        
        
        #Get Extreme Points using CDDLIB
        mat=cdd.Matrix(np.hstack((-np.array(H_array[:,k-1]).reshape(nl,1),Bd)),number_type='float')
        mat.rep_type=cdd.RepType.INEQUALITY
        poly=cdd.Polyhedron(mat)
        ext=poly.get_generators()
        extreme=np.array(ext)
        print(ext)
        (s,t)=extreme.shape
        l=1
        for i in range(0,s):
            if extreme[0,i]==1:
                for j in m.mlset:
                    m.Verticesalpha[k,l,j]=extreme[i,j] #Vertex l of the k-th polytope
                    m.Verticesbeta[k,l]=extreme[i,t-1] 
                    l=l+1
    
    #Extended Aggragated Near-Optimal Problem
    
    m.LL.deactivate()
    m.lambda=Var(m.mlset,within=Reals)
    m.sigma=Var(m.nlset,within=Reals)
    
    m.56c=Constraint(m.mlset,rule=c2)
    def d(m,j):
        value=m.d[j]
        value=value+sum(m.lambda*m.B[i,j] for i in m.mlset)
        value=value-m.sigma[j]
        return value==0
    m.56d=Constraint(m.nlset,rule=d)  #Constraint 5.6d
    
    m.CompBock=Block()
    m.CompBlock.56e=ComplementarityList(rule=(complements(m.lambda[i] >= 0,
                                                         (sum(m.A[i,j]*m.x[j] for j in m.nuset)+sum(m.B[i,j]*m.v[j] for j in m.nlset)-m.b[i]) >=0) for i in m.mlset)
    
    m.CompBlock.56f=ComplementarityList(rule=(complements(m.sigma[j]>=0, m.v[j]>=0) for j in m.nlset)) #Check notation in paper 
    
    TransformationFactory('mpec.simple_disjunction').apply_to(m.CompBlock)
    
    #BIG DISJUNCT HERE
    
    TO DO 
    '''




    def _postsolve(self):
        #
        # Create a results object
        #
        results = pyomo.opt.SolverResults()
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self.options.subsolver
        solv.wallclock_time = self.wall_time
        cpu_ = []
        for res in self.results:
            if not getattr(res.solver, 'cpu_time', None) is None:
                cpu_.append(res.solver.cpu_time)
        if cpu_:
            solv.cpu_time = sum(cpu_)
        #
        # TODO: detect infeasibilities, etc
        #
        solv.termination_condition = pyomo.opt.TerminationCondition.optimal
        #
        # PROBLEM
        #
        prob = results.problem
        prob.name = self._instance.name
        prob.number_of_constraints = self._instance.statistics.number_of_constraints
        prob.number_of_variables = self._instance.statistics.number_of_variables
        prob.number_of_binary_variables = self._instance.statistics.number_of_binary_variables
        prob.number_of_integer_variables =\
            self._instance.statistics.number_of_integer_variables
        prob.number_of_continuous_variables =\
            self._instance.statistics.number_of_continuous_variables
        prob.number_of_objectives = self._instance.statistics.number_of_objectives
        #
        ##from pyomo.core import maximize
        ##if self._instance.sense == maximize:
            ##prob.sense = pyomo.opt.ProblemSense.maximize
        ##else:
            ##prob.sense = pyomo.opt.ProblemSense.minimize
        #
        # SOLUTION(S)
        #
        self._instance.solutions.store_to(results)
        #
        # Uncache the instance
        #
        self._instance = None
        return results
