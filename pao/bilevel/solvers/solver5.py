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
from pyomo.core import TransformationFactory, minimize, maximize, Block, Constraint, Objective, Var, Reals, Binary, Integers, Any, Param, NonNegativeIntegers, RangeSet, NonNegativeReals
from pyomo.core.expr.numvalue import value
import numpy as np
from numpy import array, dot
from pyomo.common.modeling import unique_component_name
#from pyomo.gdp.util import check_model_algebraic


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
        self._upper_level_sense = minimize
        self._lower_level_sense = minimize

        # put problem into standard form
        for odata in self._instance.component_objects(Objective):
            if odata.parent_block() == self._instance:
                if odata.sense == maximize:
                    self._upper_level_sense = maximize
                    odata.set_value(-odata.expr)
                    odata.set_sense(minimize)
            if type(odata.parent_block()) == SubModel:
                if odata.sense == minimize:
                    self._lower_level_sense = minimize
                    odata.set_value(-odata.expr)
                    odata.set_sense(maximize)

        pyomo.opt.OptSolver._presolve(self, *args, **kwds)
        self.results = []

    def _apply_solver(self):
        start_time = time.time()
        M=self.options.dual_bound
        if not self.options.dual_bound:
            M=1e6
            print(f'Dual bound not specified, set to default {M}')
        delta = self.options.delta
        if not self.options.delta:
            delta = 0.05 #What should default robustness delta be if not specified? Or should I raise an error?
            print(f'Robustness parameter not specified, set to default {delta}')
        # matrix representation for bilevel problem
        matrix_repn = BilevelMatrixRepn(self._instance,standard_form=False)

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
        if self.options.do_print==True:
            print('Solution to the Highpoint Relaxation')
            for _, var in all_vars.items():
                var.pprint()
        
        # s1 <- solve the optimistic bilevel (linear/linear) problem (call solver3)
        # if s1 infeasible then return optimistic_infeasible'
        with pyomo.opt.SolverFactory('pao.bilevel.blp_global') as opt:
            opt.options.solver = solver
            self.results.append(opt.solve(self._instance,tee=self._tee,timelimit=self._timelimit))
        _check_termination_condition(self.results[-1])
        if self.options.do_print==True:
            print('Solution to the Optimistic Bilevel')
            for _, var in all_vars.items():
                var.pprint()
        #self._instance.pprint() #checking for active blocks left over from previous solves
        
        # sk <- solve the dual adversarial  problem
        # if infeasible then return dual_adversarial_infeasible

        # Collect the vertices solutions for the dual adversarial problem
        
        #Collect up the matrix B and the vector d for use in all adversarial feasibility problems 
        n=len(c_vars.items())
        m=len(sub_cons.items())
        K=len(cons.items())
        B=np.empty([m,n])
        L=np.empty([K,1])
        i=0
        p=0
        for _, var in c_vars.items():
            (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
            B[:,i]=np.transpose(np.array(A))
            i+=1
        
        _ad_block_name='_adversarial'
        self._instance.add_component(_ad_block_name, Block(Any))
        _Vertices_name='_Vertices'
        _Vertices_B_name='_VerticesB'
        self._instance.add_component(_Vertices_name,Param(cons.keys()*NonNegativeIntegers*sub_cons.keys(),mutable=True))
        Vertices=getattr(self._instance,_Vertices_name)
        self._instance.add_component(_Vertices_B_name,Param(cons.keys()*NonNegativeIntegers,mutable=True))
        VerticesB=getattr(self._instance,_Vertices_B_name)
        adversarial=getattr(self._instance,_ad_block_name)
        #Add Adversarial blocks
        for _cidS, _ in cons.items(): # <for each constraint in the upper-level problem>
            (_cid,_)=_cidS
            ad=adversarial[_cid] #shorthand
            ad.alpha=Var(sub_cons.keys(),within=NonNegativeReals) #sub_cons.keys() because it's a dual variable on the lower level constraints
            ad.beta=Var(within=NonNegativeReals)
            Hk=np.empty([n,1])
            i=0
            d=np.empty([n,1])
             
            ad.cons=Constraint(c_vars.keys()) #B^Talpha+beta*d>= H_k, v-dimension constraints so index by c_vars
            lhs_expr = {key: 0. for key in c_vars.keys()}
            rhs_expr = {key: 0. for key in c_vars.keys()}
            for _vid, var in c_vars.items():
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A #+ dot(A_q.toarray(), _fixed)
                
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                d[i,0]=float(C)
                lhs_expr[_vid]=float(C)*ad.beta
                
                (A,A_q,sign,b)=matrix_repn.coef_matrices(self._instance,var)
                idx = list(cons.keys()).index(_cidS)
                Hk[i,0]=A[idx]
                i+=1
                
                for _cid2 in sub_cons.keys():
                    idx = list(sub_cons.keys()).index(_cid2)
                    lhs_expr[_vid] += float(coef[idx])*ad.alpha[_cid2]
                
                rhs_expr[_vid] = float(A[idx])
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
            ad.deactivate()
        
            Bd=np.hstack((np.transpose(B),d))
            Eye=np.identity(m+1)
            Bd=np.vstack((Bd,Eye))
            Hk=np.vstack((Hk,np.zeros((m+1,1))))
            
            
            mat=np.hstack((-Hk,Bd))
            mat=cdd.Matrix(mat,number_type='float') 
            
            mat.rep_type=cdd.RepType.INEQUALITY
            poly=cdd.Polyhedron(mat)
            ext=poly.get_generators()
            extreme=np.array(ext)
            if self.options.do_print==True:
                print(ext)
            
            (s,t)=extreme.shape
            l=1
            for i in range(0,s):
                j=1
                if extreme[0,i]==1:
                    for _scid in sub_cons.keys():  
                    #for j in range(1,t-1): #Need to loop over extreme 1 to t-1 and link those to the cons.keys for alpha? 
                        Vertices[(_cidS,l,_scid)]=extreme[i,j] #Vertex l of the k-th polytope
                        j+=1
                    VerticesB[(_cidS,l)]=extreme[i,t-1]                    
                    l+=1
            L[p,0]=l-1  
            p+=1
        #vertex enumeration goes from 1 to L
        
        
        # Solving the full problem sn0
        _model_name = '_extended'
        _model_name = unique_component_name(self._instance, _model_name)
        
        xfrm = TransformationFactory('pao.bilevel.highpoint') #5.6a-c
        kwds = {'submodel_name': _model_name}
        xfrm.apply_to(self._instance, **kwds)    
        extended=getattr(self._instance,_model_name)
        extended.sigma=Var(c_vars.keys(),within=NonNegativeReals,bounds=(0,M))
        extended.lam=Var(sub_cons.keys(),within=NonNegativeReals,bounds=(0,M))
        
        #5.d   
        extended.d = Constraint(c_vars.keys()) #indexed by lower level variables
        d_expr= {key: 0. for key in c_vars.keys()}
        for _vid, var in c_vars.items():
            (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var) #gets d_i
            d_expr[_vid]+=float(C)
            d_expr[_vid]=d_expr[_vid]-extended.sigma[_vid]
            (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
            for _cid, _ in sub_cons.items():
                idx = list(sub_cons.keys()).index(_cid)
                d_expr[_vid]+=extended.lam[_cid]*float(A[idx])
        expr = d_expr[_vid] == 0
        if not type(expr) is bool:
            extended.d[_vid] = expr
        else:
            extended.d[_vid] = Constraint.Skip   
        #5.e (Complementarity)
        extended.e = ComplementarityList()
        for _cid, _ in sub_cons.items():
            idx=list(sub_cons.keys()).index(_cid)
            expr=0
            for _vid, var in fixed_vars.items(): #A_i*x
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                expr+=float(A[idx])*fixed_vars[_vid]  
            for _vid, var in c_vars.items(): #B_i*v
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                expr+=float(A[idx])*c_vars[_vid]
            expr=expr-float(b[idx])
            extended.e.add(complements(extended.lam[_cid] >= 0, expr <= 0))
            
        
        #5.f (Complementarity)
        extended.f = ComplementarityList()
        for _vid,var in c_vars.items():
            extended.f.add(complements(extended.sigma[_vid]>=0,var>=0))
        
        #Replace 5.h-5.j with 5.7 Disjunction
        extended.disjunction=Block(cons.keys()) #One disjunction per adversarial problem, one adversarial problem per upper level constraint
        k=0
        for _cidS,_ in cons.items():
            idxS=list(cons.keys()).index(_cidS)
            [_cid,sign]=_cidS
            disjunction=extended.disjunction[_cidS] #shorthand
            disjunction.Lset=RangeSet(1,L[k,0])
            disjunction.disjuncts=Disjunct(disjunction.Lset)
            for i in disjunction.Lset: #defining the L disjuncts
                l_expr=0
                for _vid, var in c_vars.items():
                    (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                    l_expr+=float(C)*var #d^Tv 
                l_expr+=delta
                l_expr=VerticesB[(_cidS,i)]*l_expr #beta(d^Tv+delta)
            
                for _cid, Scons in sub_cons.items(): #SUM over i to ml
                    Ax=0
                    idx=list(sub_cons.keys()).index(_cid)
                    for _vid, var in fixed_vars.items():
                        (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                        Ax += float(A[idx])*var
                    l_expr+=Vertices[(_cidS,i,_cid)]*(float(b[idx])-Ax)
                         
                r_expr=0
                for _vid,var in fixed_vars.items():
                    (A, A_q, sign, b) = matrix_repn.coef_matrices(self._instance, var) #get q and G
                    r_expr=r_expr-float(A[idxS])*var
                r_expr+=float(b[idxS])
                        
                disjunction.disjuncts[i].cons=Constraint(expr= l_expr<=r_expr)
    
            disjunction.seven=Disjunction(expr=[disjunction.disjuncts[i] for i in disjunction.Lset],xor=False)    
            k+=1
        #extended.pprint()
        TransformationFactory('mpec.simple_disjunction').apply_to(extended)
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(extended)
        with pyomo.opt.SolverFactory(solver) as opt:
            self.results.append(opt.solve(extended,
                                             tee=self._tee,
                                             timelimit=self._timelimit))
            _check_termination_condition(self.results[-1]) 
        # Return the sn0 solution
        if self.options.do_print==True:
            print('Robust Solution')
            for _vid, _ in fixed_vars.items():
                fixed_vars[_vid].pprint()
            for _vid, _ in c_vars.items():
                c_vars[_vid].pprint()
                extended.lam.pprint()
                extended.sigma.pprint()
        stop_time = time.time()
        self.wall_time = stop_time - start_time
        return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                           log=getattr(opt, '_log', None))
        

    def _postsolve(self):
        results = self.results[-1]
        
        results.wallclock_time = self.wall_time
        
        return results