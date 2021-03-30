'''
Implementation of the algorithm presented in "A projection-based reformulation
and decomposition algorithm for global optimization of a class of mixed integer 
bilevel linear programs" by Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You

Implemented May-August 2019 by She'ifa Punla-Green at Sandia National Labs

This algorithm seeks to solve the following bilevel MILP:
    min cR*xu + cZ*yu + dR*xl0 + dZ*yl0 
    s.t. AR*xu + AZ*yu + BR*xl0 + BZ* yl0 <= r
     (xl0,yl0) in argmax {wR*xl+wZ*yl: PR*xl+PZ*yl<=s-QR*xu-QZ*yu}
'''
import time

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.mpec import *

from . import pyomo_util


infinity = float('inf')

offset=0

def mat2dict(m, start, stop):
    if m is None:
        return {}
    cx = m.tocoo()    
    return {(i+offset,j+offset-start):float(v) for i,j,v in zip(cx.row, cx.col, cx.data) if j>=start and j<stop}

def array2dict(m, start=0, stop=None):
    if m is None:
        return {}
    if stop is None:
        stop = len(m)
    return {i+offset-start:float(m[i]) for i in range(len(m)) if i>=start and i<stop}


def create_pyomo_model(mpr, M):
    '''
    Parameter Import

    Notation:
    m is for upper level
    n is for lower level
    x is continuous (R)
    y is discrete (Z)

    coefficient vectors and matrices should be dictionaries (with tuples for matrices)
    variable size should be floats
    '''
    U = mpr.U
    L = mpr.U.LL[0]

    mU = len(U.b)
    mR = U.x.nxR
    mZ = U.x.nxZ
    nL = len(L.b)
    nR = L.x.nxR
    nZ = L.x.nxZ

    AR = mat2dict(U.A[U], 0,       U.x.nxR)
    AZ = mat2dict(U.A[U], U.x.nxR, U.x.nxR+U.x.nxZ)
    BR = mat2dict(U.A[L], 0,       L.x.nxR)
    BZ = mat2dict(U.A[L], L.x.nxR, L.x.nxR+U.x.nxZ)
    r  = array2dict(U.b)
    cR = array2dict(U.c[U], 0,       U.x.nxR)
    cZ = array2dict(U.c[U], U.x.nxR, U.x.nxR+U.x.nxZ)
    dR = array2dict(U.c[L], 0,       L.x.nxR)
    dZ = array2dict(U.c[L], L.x.nxR, L.x.nxR+U.x.nxZ)

    PR = mat2dict(L.A[L], 0,       L.x.nxR)
    PZ = mat2dict(L.A[L], L.x.nxR, L.x.nxR+U.x.nxZ)
    QR = mat2dict(L.A[U], 0,       U.x.nxR)
    QZ = mat2dict(L.A[U], U.x.nxR, U.x.nxR+U.x.nxZ)
    s  = array2dict(L.b)
    wR = array2dict(L.c[L], 0,       L.x.nxR)
    wZ = array2dict(L.c[L], L.x.nxR, L.x.nxR+U.x.nxZ)
    '''
    mU number of upper level constraints
    mR number of upper level continuous variables
    mZ number of upper level integer variables

    nL number of lower level constraints
    nR number of lower level continuous variables
    nZ number of lower level integer variables

    AR constraint matrix for upper level problem, upper level continuous variables
    AZ constraint matrix for upper level problem, upper level integer variables
    BR constraint matrix for upper level problem, lower level continuous variables
    BZ constraint matrix for upper level problem, lower level integer variables
    r  RHS vector for upper level constraint
    cR coefficient vector for upper level objective, upper level continuous variables
    cZ coefficient vector for upper level objective, upper level integer variables
    dR coefficient vector for upper level objective, lower level continuous variables
    dZ coefficient vector for upper level objective, lower level integer variables

    PR constraint matrix for lower level problem, lower level continuous variables
    PZ constraint matrix for lower level problem, lower level integer variables
    QR constraint matrix for lower level problem, upper level continuous variables
    QZ constraint matrix for lower level problem, upper level integer variables
    s  RHS vector for lower level constraint
    wR coefficient vector for the lower level objective, lower level continuous variables
    wZ coefficient vector for the lower level objective, lower level integer variables
    '''

    #Master problem, subproblem 1, and subproblem 2 are all blocks on a Parent concrete model
    #so that parameters that are present in all three problems can be shared
    #They are mutable parameters on a parent model rather than just python dictionaries so that 
    #Pyomo constraints that would be trivially true such as 0<=0 do not cause an error

    Parent=ConcreteModel()

    Parent.mUset=RangeSet(offset, (mU-1+offset))
    Parent.nLset=RangeSet(offset, (nL-1+offset))
    Parent.nRset=RangeSet(offset, (nR-1+offset))
    Parent.nZset=RangeSet(offset, (nZ-1+offset))
    Parent.mRset=RangeSet(offset, (mR-1+offset))
    Parent.mZset=RangeSet(offset, (mZ-1+offset))

    Parent.s=Param(Parent.nLset,initialize=s,default=0,mutable=True)
    Parent.r=Param(Parent.mUset,initialize=r,default=0,mutable=True)
    Parent.cR=Param(Parent.mRset,initialize=cR,default=0,mutable=True)
    Parent.cZ=Param(Parent.mZset,initialize=cZ,default=0,mutable=True)
    Parent.dR=Param(Parent.nRset,initialize=dR,default=0,mutable=True)
    Parent.dZ=Param(Parent.nZset,initialize=dZ,default=0,mutable=True)
    Parent.wR=Param(Parent.nRset,initialize=wR,default=0,mutable=True)
    Parent.wZ=Param(Parent.nZset,initialize=wZ,default=0,mutable=True)

    Parent.AR=Param(Parent.mUset,Parent.mRset,initialize=AR, default=0,mutable=True)
    Parent.AZ=Param(Parent.mUset,Parent.mZset,initialize=AZ, default=0,mutable=True)
    Parent.BR=Param(Parent.mUset,Parent.nRset,initialize=BR, default=0,mutable=True)
    Parent.BZ=Param(Parent.mUset,Parent.nZset,initialize=BZ, default=0,mutable=True)
    Parent.PR=Param(Parent.nLset,Parent.nRset,initialize=PR, default=0,mutable=True)
    Parent.PZ=Param(Parent.nLset,Parent.nZset,initialize=PZ, default=0,mutable=True)
    Parent.QR=Param(Parent.nLset,Parent.mRset,initialize=QR, default=0,mutable=True)
    Parent.QZ=Param(Parent.nLset,Parent.mZset,initialize=QZ, default=0,mutable=True)

    Parent.zero=Param(initialize=0, mutable=True) 

    '''
    Master Problem: (P9)
        min cR*xu + cZ*yu + dR*xl0 + dZ*yl0 
        s.t. AR*xu + AZ*yu + BR*xl0 + BZ* yl0 <= r (12)
             PR*xl+PZ*yl<=s-QR*xu-QZ*yu (13)
             wR*xl0>= wR*xltilde (74)
             PR*xltilde <= s- QR*xu - QZ*yu-PZ*yl0 (75a)
             PR*pitilde>= wR (75b)
             xltilde \perp PR*pitilde - wR (76a)
             pitilde \perp s- QR*xu- QZ*yu - PR*xltilde (76b)
             PR*xlj-tj+PZ*yl<=s-QR*xu-QZ*yu for all 1<=j<=k (79)
             PR*lambdaj>=0 for all 1<=j<=k (83a)
             xlj \perp PR*lambdaj for all 1<=j<=k (83b)
             e-lambdaj >=0 for all 1<=j<=k (84a)
             tj \perp e - lambdaj for all 1<=j<=k (84b)
             lambdaj \perp s-QR*xu-QZ*yu-PR*xlj-PZ*ylj +tj for all 1<=j<=k (85)
             [e*tj=0] ==> [Constraint Block] for all 1<=j<=k (82)
             Constraint Block:
                 wR*xl0 + wZ*yl0 >= wR*xlj + wZ*ylj
                 PR*xlj <= s - QR*xu - QZ*yu - PZ*ylj
                 PR*pij>= wR
                 xlj \perp PR*pij-wR
                 pij \perp s - QR*xu - QZ*yu - PZ*ylj - PR*xlj
    '''

    Parent.Master=Block()

    Parent.Master.Y=Param(Any,within=NonNegativeIntegers,mutable=True) #Growing Parameter
    Parent.Master.xu=Var(Parent.mRset,within=NonNegativeReals,bounds=(0,M))
    Parent.Master.yu=Var(Parent.mZset,within=NonNegativeIntegers,bounds=(0,M))
    Parent.Master.xl0=Var(Parent.nRset,within=NonNegativeReals,bounds=(0,M))
    Parent.Master.yl0=Var(Parent.nZset,within=NonNegativeIntegers,bounds=(0,M))

    Parent.Master.x=Var(Any,within=NonNegativeReals,dense=False,bounds=(0,M)) #Growing variable
    Parent.Master.pi=Var(Any, within=NonNegativeReals,dense=False,bounds=(0,M)) #Growing variable

    Parent.Master.xltilde=Var(Parent.nRset,within=NonNegativeReals,bounds=(0,M))
    Parent.Master.pitilde=Var(Parent.nLset,within=NonNegativeReals,bounds=(0,M))

    Parent.Master.t=Var(Any,within=NonNegativeReals,dense=False,bounds=(0,M)) #Growing variable
    Parent.Master.lam=Var(Any,within=NonNegativeReals,dense=False,bounds=(0,M)) #Growing variable


    def Master_obj(Master):
        value=(sum(Parent.cR[j]*Parent.Master.xu[j] for j in Parent.mRset)+
               sum(Parent.cZ[j]*Parent.Master.yu[j] for j in Parent.mZset)+
               sum(Parent.dR[j]*Parent.Master.xl0[j] for j in Parent.nRset)+
               sum(Parent.dZ[j]*Parent.Master.yl0[j] for j in Parent.nZset))
        return value

    Parent.Master.Theta_star=Objective(rule=Master_obj,sense=minimize)
        
    def Master_c1(Master,i):
        value=(sum(Parent.AR[(i,j)]*Parent.Master.xu[j] for j in Parent.mRset)+
               sum(Parent.AZ[(i,j)]*Parent.Master.yu[j] for j in Parent.mZset)+
               sum(Parent.BR[(i,j)]*Parent.Master.xl0[j] for j in Parent.nRset)+
               sum(Parent.BZ[(i,j)]*Parent.Master.yl0[j] for j in Parent.nZset))
        return value - Parent.r[i] <= Parent.zero
    Parent.Master.c1=Constraint(Parent.mUset,rule=Master_c1) #(12)

    def Master_c2(Master,i):
        value=(sum(Parent.QR[(i,j)]*Parent.Master.xu[j] for j in Parent.mRset)+
               sum(Parent.QZ[(i,j)]*Parent.Master.yu[j] for j in Parent.mZset)+
               sum(Parent.PR[(i,j)]*Parent.Master.xl0[j] for j in Parent.nRset)+
               sum(Parent.PZ[(i,j)]*Parent.Master.yl0[j] for j in Parent.nZset))
        return value - Parent.s[i] <= Parent.zero
    Parent.Master.c2=Constraint(Parent.nLset,rule=Master_c2) #(13)


    def Master_c3(Master):
        l_value=sum(Parent.wR[i]*Parent.Master.xl0[i] for i in Parent.nRset)
        r_value=sum(Parent.wR[i]*Parent.Master.xltilde[i] for i in Parent.nRset)
        return l_value - r_value >= Parent.zero
    Parent.Master.c3=Constraint(rule=Master_c3) #(74)


    def Master_c4(Master,i):
        value=(sum(Parent.PR[(i,j)]*Parent.Master.xltilde[j] for j in Parent.nRset)+
               sum(Parent.PZ[(i,j)]*Parent.Master.yl0[j] for j in Parent.nZset)+
               sum(Parent.QZ[(i,j)]*Parent.Master.yu[j] for j in Parent.mZset)+
               sum(Parent.QR[(i,j)]*Parent.Master.xu[j] for j in Parent.mRset))
        return value - Parent.s[i] <= Parent.zero
    Parent.Master.c4=Constraint(Parent.nLset,rule=Master_c4) #(75a)

    def Master_c5(Master,j):
        PRpi=sum(Parent.PR[(i,j)]*Parent.Master.pitilde[i] for i in Parent.nLset)
        return PRpi - Parent.wR[j] >= Parent.zero
    Parent.Master.c5=Constraint(Parent.nRset, rule=Master_c5) #(75b)


    Parent.Master.CompBlock=Block()
    Parent.Master.CompBlock.c6=ComplementarityList(rule=(complements(Parent.Master.xltilde[j] >= 0,
                                                           sum(Parent.PR[(i,j)]*Parent.Master.pitilde[i] for i in Parent.nLset)-
                                                           Parent.wR[j] >=0) for j in Parent.nRset))
    #(76a)

    #(76b)
    Parent.Master.CompBlock.c7=ComplementarityList(rule=(complements(Parent.Master.pitilde[j] >= 0,
                                                           (Parent.s[j]-sum(Parent.QR[(j,i)]*Parent.Master.xu[i] for i in Parent.mRset) -
                                                           sum(Parent.QZ[(j,i)]*Parent.Master.yu[i] for i in Parent.mZset)-
                                                           sum(Parent.PR[(j,i)]*Parent.Master.xltilde[i] for i in Parent.nRset)-
                                                           sum(Parent.PZ[(j,i)]*Parent.Master.yl0[i] for i in Parent.nZset))>=0) for j in Parent.nLset))

    #TransformationFactory('mpec.simple_disjunction').apply_to(Parent.Master.CompBlock) #To get the complementarity not in disjunction


    Parent.Master.c_col = ConstraintList()
    Parent.Master.CompBlock2=Block(Any)

    Parent.Master.DisjunctionBlock=Block(Any)


    #Create parameters that will be updated on each iteration, initialized with coefficient vector of same size 
    Parent.xu_star=Param(Parent.mRset,initialize=cR,default=0,mutable=True)
    Parent.xl0_star=Param(Parent.nRset,initialize=wR,default=0,mutable=True)
    Parent.yu_star=Param(Parent.mZset,initialize=cZ,default=0,mutable=True)
    Parent.yl0_star=Param(Parent.nZset,initialize=wZ,default=0,mutable=True)
    theta=0
    Parent.theta=Param(initialize=theta,mutable=True)

    Parent.xl_hat=Param(Parent.nRset,initialize=0,within=NonNegativeReals,default=0,mutable=True)
    Parent.yl_hat=Param(Parent.nZset,initialize=0,within=NonNegativeIntegers,default=0,mutable=True)

    Parent.yl_star=Param(Parent.nZset, initialize=0,within=NonNegativeIntegers,default=0,mutable=True) 
    Parent.xl_star=Param(Parent.nRset, initialize=0,within=NonNegativeReals,default=0,mutable=True)
    Parent.Theta_0=Param(initialize=0,mutable=True)
    Parent.yl_arc=Param(Parent.nZset, initialize=0,within=NonNegativeIntegers,default=0,mutable=True)


    ''' Subproblem 1
        theta(xu*,yu*)=max wR*xl + wZ*yl
        s.t. PR*xl+PZ*yl <= s- QR*xu* - QZ*yu* (56)
    '''
    def sub1_obj(sub1):
        value=(sum(Parent.wR[j]*Parent.sub1.xl[j] for j in Parent.nRset)+
               sum(Parent.wZ[j]*Parent.sub1.yl[j] for j in Parent.nZset))
        return value

    def sub1_c1(sub1,i):
        value=(sum(Parent.PR[(i,j)]*Parent.sub1.xl[j] for j in Parent.nRset)+
               sum(Parent.PZ[(i,j)]*Parent.sub1.yl[j] for j in Parent.nZset)+
               sum(Parent.QR[(i,j)]*Parent.xu_star[j] for j in Parent.mRset)+
               sum(Parent.QZ[(i,j)]*Parent.yu_star[j] for j in Parent.mZset))
        return Parent.zero <= Parent.s[i] -value


    Parent.sub1=Block()   
    Parent.sub1.xl=Var(Parent.nRset, within=NonNegativeReals)
    Parent.sub1.yl=Var(Parent.nZset, within=NonNegativeIntegers)
    Parent.sub1.theta=Objective(rule=sub1_obj,sense=maximize)
    Parent.sub1.c1=Constraint(Parent.nLset,rule=sub1_c1)



    ''' Subproblem 2
        Theta_0(xu*,yu*)=min dR*xl + dZ*yl
        s.t. PR*xl+PZ*yl <= s- QR*xu* - QZ*yu* (56)
             BR*xl + BZ*yl <= r- AR*xu* - AZ*yu* (59)
             wR*xl + wZ*yl >= theta(xu*,yu*) (60)
    '''
    def sub2_obj(sub2):
        value=(sum(Parent.dR[i]*Parent.sub2.xl[i] for i in Parent.nRset)+
               sum(Parent.dZ[i]*Parent.sub2.yl[i] for i in Parent.nZset)) 
        return value

    def sub2_c1(sub2,i):
        value=(sum(Parent.PR[(i,j)]*Parent.sub2.xl[j] for j in Parent.nRset)+
               sum(Parent.PZ[(i,j)]*Parent.sub2.yl[j] for j in Parent.nZset)+
               sum(Parent.QR[(i,j)]*Parent.xu_star[j] for j in Parent.mRset)+
               sum(Parent.QZ[(i,j)]*Parent.yu_star[j] for j in Parent.mZset))
        return Parent.zero <= Parent.s[i] -value

    def sub2_c2(sub2,i):
        value=(sum(Parent.BR[(i,j)]*Parent.sub2.xl[j] for j in Parent.nRset)+
               sum(Parent.BZ[(i,j)]*Parent.sub2.yl[j] for j in Parent.nZset)+
               sum(Parent.AR[(i,j)]*Parent.xu_star[j] for j in Parent.mRset)+
               sum(Parent.AZ[(i,j)]*Parent.yu_star[j] for j in Parent.mZset))                                                                        
        return Parent.zero <= Parent.r[i] - value

    def sub2_c3(sub2):
        value=(sum(Parent.wR[j]*Parent.sub2.xl[j] for j in Parent.nRset)+
               sum(Parent.wZ[j]*Parent.sub2.yl[j] for j in Parent.nZset))
        return value - Parent.theta >= Parent.zero


    Parent.sub2=Block()
        
    Parent.sub2.xl=Var(Parent.nRset, within=NonNegativeReals)
    Parent.sub2.yl=Var(Parent.nZset, within=NonNegativeIntegers)
        
    Parent.sub2.Theta_0=Objective(rule=sub2_obj,sense=minimize)

    Parent.sub2.c1=Constraint(Parent.nLset,rule=sub2_c1) #(56)
    Parent.sub2.c2=Constraint(Parent.mUset,rule=sub2_c2) #(59)
    Parent.sub2.c3=Constraint(rule=sub2_c3)

    return Parent


def Master_add(Parent, k, epsilon): #function for adding constraints on each iteration
    #(79)
    
    Parent.Master.CompBlock2[k].c_comp=ComplementarityList()
    for i in Parent.nLset:
        r_value= (sum(Parent.PR[(i,j)]*Parent.Master.x[(j,k)] for j in Parent.nRset)-
                  Parent.Master.t[(i,k)])
        l_value= (Parent.s[i]-
                  sum(Parent.QR[(i,j)]*Parent.Master.xu[j] for j in Parent.mRset)-
                  sum(Parent.QZ[(i,j)]*Parent.Master.yu[j] for j in Parent.mZset)-
                  sum(Parent.PZ[(i,j)]*Parent.Master.Y[(j,k)] for j in Parent.nZset)) 
    
        Parent.Master.c_col.add(l_value-r_value >= Parent.zero)
    
    for i in Parent.nRset:  
        Parent.Master.c_col.add(sum(Parent.PR[(j,i)]*Parent.Master.lam[(j,k)] for j in Parent.nLset)>=Parent.zero)
        Parent.Master.CompBlock2[k].c_comp.add(complements(Parent.Master.x[(i,k)]>=0, 
                                                             sum(Parent.PR[(j,i)]*Parent.Master.lam[(j,k)] for j in Parent.nLset)>=0)) #(83)  
    
    for i in Parent.nLset:
        Parent.Master.c_col.add(1-Parent.Master.lam[(i,k)]>=Parent.zero)
        Parent.Master.CompBlock2[k].c_comp.add(complements(1-Parent.Master.lam[(i,k)]>=0,Parent.Master.t[(i,k)]>=0)) #(84)
    
    for i in Parent.nLset:
        Parent.Master.CompBlock2[k].c_comp.add(complements(Parent.Master.lam[(i,k)]>=0,(Parent.s[i]-
                                                     sum(Parent.QR[(i,j)]*Parent.Master.xu[j] for j in Parent.mRset)-
                                                     sum(Parent.QZ[(i,j)]*Parent.Master.yu[j] for j in Parent.mZset)-
                                                     sum(Parent.PZ[(i,j)]*Parent.Master.Y[(j,k)] for j in Parent.nZset)-
                                                     sum(Parent.PR[(i,j)]*Parent.Master.x[(j,k)] for j in Parent.nRset)+
                                                     Parent.Master.t[(i,k)]>=0))) #(85)
    
    
    
    #TransformationFactory('mpec.simple_disjunction').apply_to(Parent.Master.CompBlock2[k]) #To get the complementarity not in disjunction
    
    #(82) Disjunction
    Parent.Master.DisjunctionBlock[k].LH=Disjunct()
    Parent.Master.DisjunctionBlock[k].BLOCK=Disjunct()
    
    
    
    Parent.Master.DisjunctionBlock[k].LH.cons=Constraint(expr= sum(Parent.Master.t[(j,k)] for j in Parent.nLset) >= epsilon) 
    
    Parent.Master.DisjunctionBlock[k].BLOCK.cons=ConstraintList()
    
    l_value = (sum(Parent.wR[j]*Parent.Master.xl0[j] for j in Parent.nRset)+
               sum(Parent.wZ[j]*Parent.Master.yl0[j] for j in Parent.nZset))
    r_value = (sum(Parent.wR[j]*Parent.Master.x[(j,k)] for j in Parent.nRset)+
               sum(Parent.wZ[j]*Parent.Master.Y[(j,k)] for j in Parent.nZset))
    Parent.Master.DisjunctionBlock[k].BLOCK.cons.add(l_value - r_value >= 0)  #(82a) 
    
    for i in Parent.nLset:
        r_value = (Parent.s[i] - 
               sum(Parent.QR[(i,j)]*Parent.Master.xu[j] for j in Parent.mRset)-
               sum(Parent.QZ[(i,j)]*Parent.Master.yu[j] for j in Parent.mZset)-
               sum(Parent.PZ[(i,j)]*Parent.Master.Y[(j,k)] for j in Parent.nZset))
        l_value = sum(Parent.PR[(i,j)]*Parent.Master.x[(j,k)] for j in Parent.nRset)#(82b)
        
        Parent.Master.DisjunctionBlock[k].BLOCK.cons.add(r_value-l_value >= 0)
        
    for j in Parent.nRset:
        value= sum(Parent.PR[(i,j)]*Parent.Master.pi[(i,k)] for i in Parent.nLset)
        Parent.Master.DisjunctionBlock[k].BLOCK.cons.add(value >= Parent.wR[j])#(82c1)
    
    Parent.Master.DisjunctionBlock[k].BLOCK.comp1=ComplementarityList(rule=(complements(Parent.Master.pi[(j,k)]>=0, 
                                  (Parent.s[j]-
                                   sum(Parent.QR[(j,i)]*Parent.Master.xu[i] for i in Parent.mRset)-
                                   sum(Parent.QZ[(j,i)]*Parent.Master.yu[i] for i in Parent.mZset)-
                                   sum(Parent.PR[(j,i)]*Parent.Master.x[(i,k)] for i in Parent.nRset)-
                                   sum(Parent.PZ[(j,i)]*Parent.Master.Y[(i,k)] for i in Parent.nZset))>=0) for j in Parent.nLset)) #(82d)
    
    Parent.Master.DisjunctionBlock[k].BLOCK.comp2=ComplementarityList(rule=(complements(
            Parent.Master.x[(j,k)]>=0,
            sum(Parent.PR[(i,j)]*Parent.Master.pi[(i,k)] for i in Parent.nLset)-Parent.wR[j]>=0) for j in Parent.nRset)) #(82c2)
    
    
    Parent.Master.DisjunctionBlock[k].c_disj=Disjunction(expr=[Parent.Master.DisjunctionBlock[k].LH, Parent.Master.DisjunctionBlock[k].BLOCK])
    
    #TransformationFactory('mpec.simple_disjunction').apply_to(
    #    Parent.Master.DisjunctionBlock[k].BLOCK) #to get the complementarity in the disjunction
    return Parent


def UBnew(Parent):
    return sum(Parent.cR[j]*Parent.xu_star[j] for j in Parent.cR) +\
           sum(Parent.cZ[j]*Parent.yu_star[j] for j in Parent.cZ) +\
           Parent.Theta_0


def get_value(config, name, default):
    if name not in config:
        return default
    value = config[name]
    if value is None:
        return default
    return value

def check_termination(LB, UB, atol, rtol, quiet):
    if LB-UB > atol or (LB-UB)/(1+min(abs(LB),abs(UB))) > rtol: #Output
        if not quiet:
            raise RuntimeError(f'Error: Upper bound greater than lower bound after {k} iterations and {elapsed} seconds: Obj={LB} UB={UB}')

    if abs(UB-LB) <= atol or abs(UB-LB)/(1+min(abs(LB),abs(UB))) <= rtol: #Output
        return 1

    return 0


def execute_PCCG_solver(mpr, config, results):
    t = time.time()

    #These parameters can be changed for your specific problem
    epsilon = get_value(config, 'epsilon', 1e-4) #For use in disjunction approximation
    atol    = get_value(config, 'atol', 1e-8)   #absolute tolerance for UB-LB to claim convergence
    rtol    = get_value(config, 'rtol', 1e-8)   #relative tolerance for UB-LB to claim convergence
    maxit   = get_value(config, 'maxit', 5)     #Maximum number of iterations
    M       = get_value(config, 'bigm', 1e6)    #upper bound on variables
    solver  = config.mip_solver                 # MIP solver to use here
    quiet   = config.quiet                      # If True, then suppress output

    LB=-infinity
    UB=infinity
    k=0
     
    flag=0

    Parent = create_pyomo_model(mpr, M)

    bigm_xfrm = TransformationFactory('gdp.bigm')

    #Step 1: Initialization (done)
    if isinstance(solver, str):
        opt = SolverFactory(solver)
    else:
        opt = solver

    #Iteration
    while k < maxit:
        #Step 2: Solve the Master Problem
        TransformationFactory('mpec.simple_disjunction').apply_to(Parent.Master)
        bigm_xfrm.apply_to(Parent.Master) 
        res = opt.solve(Parent.Master)
        if res.solver.termination_condition !=TerminationCondition.optimal:
            raise RuntimeError("ERROR! ERROR! Master: Could not find optimal solution")

        for i in Parent.xu_star:
            Parent.xu_star[i]=Parent.Master.xu[i].value    
        for i in Parent.yu_star:
            Parent.yu_star[i]=Parent.Master.yu[i].value    
        for i in Parent.xl0_star:
            Parent.xl0_star[i]=Parent.Master.xl0[i].value
        for i in Parent.yl0_star:
            Parent.yl0_star[i]=Parent.Master.yl0[i].value

        LB=value(Parent.Master.Theta_star) 
        if not quiet:
            print(f'Iteration {k}: Master Obj={LB} UB={UB}')

        #Step 3: Terminate?
        flag = check_termination(LB, UB, atol, rtol, quiet)
        if flag:
            elapsed = time.time() - t
            break

        if not quiet:
            print("Step 4")
        #Step 4: Solve first subproblem
        results1=opt.solve(Parent.sub1) 
        
        if results1.solver.termination_condition !=TerminationCondition.optimal:
            raise RuntimeError("ERROR! ERROR! Subproblem 1: Could not find optimal solution")
        Parent.theta=value(Parent.sub1.theta)

        for i in Parent.xl_hat:
            Parent.xl_hat[i]=Parent.sub1.xl[i].value 
        for i in Parent.yl_hat:
            Parent.yl_hat[i]=int(round(Parent.sub1.yl[i].value)) 
        
        if not quiet:
            print("Step 5")
        #Step 5: Solve second subproblem
        results2=opt.solve(Parent.sub2)
        
        if results2.solver.termination_condition==TerminationCondition.optimal: #If Optimal
            for i in Parent.xl_star:
                Parent.xl_star[i]=Parent.sub2.xl[i].value
            for i in Parent.yl_star:
                Parent.yl_star[i]=int(round(Parent.sub2.yl[i].value))
                Parent.yl_arc[i]=int(round(Parent.sub2.yl[i].value))
            Parent.Theta_0=value(Parent.sub2.Theta_0)

     
            UB=min(UB,value(UBnew(Parent)))
            
        elif results2.solver.termination_condition==TerminationCondition.infeasible or results2.solver.termination_condition==TerminationCondition.infeasibleOrUnbounded: #If infeasible
            for i in Parent.yl_arc:
                Parent.yl_arc[i]=Parent.yl_hat[i]  
        else: 
             raise RuntimeError("ERROR! Unexpected termination condition for Subproblem2: %s.  Expected an infeasible or optimal solution." % str(results2.solver.termination_condition)) 
        
        if not quiet:
            print("Step 6")
        #Step 6: Add new constraints
        k = k+1
        for i in Parent.yl_arc:  #range(nZ):
            Parent.Master.Y[(i,k)]=Parent.yl_arc[i] #Make sure yl_arc is int or else Master.Y rejects
        Master_add(Parent, k, epsilon)

        if not quiet:
            print(f'Iteration {k}: Step 7 Obj={LB} UB={UB}')
        #Step 7: Loop 
        flag = check_termination(LB, UB, atol, rtol, quiet)
        if flag:
            elapsed = time.time() - t
            break

    #Output Information regarding objective and time/iterations to convergence    
    elapsed = time.time() - t
        
    results.solver.name = 'PCCG'
    results.solver_time = elapsed
    if k>= maxit:
        if not quiet:
            print('Maximum Iterations Reached')
        results.solver.termination_condition = TerminationCondition.maxIterations
    elif k< maxit and flag !=1:
        if not quiet:
            print(f'Optimal Solution Found in {k} iterations and {elapsed} seconds: Obj={UB}')
        results.solver.termination_condition = pyomo_util.pyomo2pao_termination_condition(TerminationCondition.optimal)
        results.best_feasible_objective = UB

    return Parent.Master.xu, Parent.Master.yu, Parent.Master.xl0, Parent.Master.yl0
