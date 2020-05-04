#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# A stochastic bilevel interdiction adapted from:
#   "A Stochastic Program for Interdicting Smuggled Nuclear Material"
#   by F. Pan, W. S. Charlton, D. P. Morton
#   DRAFT August 19, 2002
#
# example contributor: She'ifa Punla-Green

import time

from pyomo.environ import *
from pao.bilevel import *

#A incidence matrix
#n number of nodes
#e number of edges
#b budget
#c cost of edge sensor placement
#ns number of starting nodes
#nt number of ending nodes

def pyomo_create_model():
    b = 1
    e = 16
    n = 12
    ns = 3
    nt = 4

    E = [(2, 1), (3, 1), (3, 2), (3, 6), (4, 3), (4, 5), (5, 2), (5, 6), (6, 4), (6, 7), (6, 8), (6, 9), (9, 11), (10, 6),
         (11, 10), (11, 12)]
    W = [(4, 1), (4, 7), (4, 8), (4, 12), (5, 1), (5, 7), (5, 8), (5, 12), (11, 1), (11, 7), (11, 8), (11, 12)]
    FS = {}
    FS[1] = []
    FS[2] = [(2, 1)]
    FS[3] = [(3, 1), (3, 2), (3, 6)]
    FS[4] = [(4, 3), (4, 5)]
    FS[5] = [(5, 2), (5, 6)]
    FS[6] = [(6, 4), (6, 7), (6, 8), (6, 9)]
    FS[7] = []
    FS[8] = []
    FS[9] = [(9, 11)]
    FS[10] = [(10, 6)]
    FS[11] = [(11, 10), (11, 12)]
    FS[12] = []

    RS = {}
    RS[1] = [(2, 1), (3, 1)]
    RS[2] = [(3, 2), (5, 2)]
    RS[3] = [(4, 3)]
    RS[4] = [(6, 4)]
    RS[5] = [(4, 5)]
    RS[6] = [(3, 6), (5, 6), (10, 6)]
    RS[7] = [(6, 7)]
    RS[8] = [(6, 8)]
    RS[9] = [(6, 9)]
    RS[10] = [(11, 10)]
    RS[11] = [(9, 11)]
    RS[12] = [(11, 12)]

    P = {}
    P[(4, 1)] = 1 / (ns * nt)
    P[(4, 7)] = 1 / (ns * nt)
    P[(4, 8)] = 1 / (ns * nt)
    P[(4, 12)] = 1 / (ns * nt)
    P[(5, 1)] = 1 / (ns * nt)
    P[(5, 7)] = 1 / (ns * nt)
    P[(5, 8)] = 1 / (ns * nt)
    P[(5, 12)] = 1 / (ns * nt)
    P[(11, 1)] = 1 / (ns * nt)
    P[(11, 7)] = 1 / (ns * nt)
    P[(11, 8)] = 1 / (ns * nt)
    P[(11, 12)] = 1 / (ns * nt)

    c = 1

    p = {}
    p[(2, 1)] = 0.5
    p[(3, 1)] = 0.5
    p[(3, 2)] = 0.5
    p[(3, 6)] = 0.5
    p[(4, 3)] = 0.25
    p[(4, 5)] = 0.25
    p[(5, 2)] = 0.25
    p[(5, 6)] = 0.25
    p[(6, 4)] = 0.25
    p[(6, 7)] = 0.5
    p[(6, 8)] = 0.5
    p[(6, 9)] = 0.5
    p[(9, 11)] = 0.25
    p[(10, 6)] = 0.5
    p[(11, 10)] = 0.25
    p[(11, 12)] = 0.25

    q = {}
    q[(2, 1)] = 0.25
    q[(3, 1)] = 0.25
    q[(3, 2)] = 0.25
    q[(3, 6)] = 0.25
    q[(4, 3)] = 0.1
    q[(4, 5)] = 0.1
    q[(5, 2)] = 0.1
    q[(5, 6)] = 0.1
    q[(6, 4)] = 0.1
    q[(6, 7)] = 0.25
    q[(6, 8)] = 0.25
    q[(6, 9)] = 0.25
    q[(9, 11)] = 0.1
    q[(10, 6)] = 0.25
    q[(11, 10)] = 0.1
    q[(11, 12)] = 0.1

    m=ConcreteModel()
    m.nodeset=RangeSet(1,n)
    m.edgeset=RangeSet(1,e)
    m.Edgeset=Set(initialize=E)
    m.Omega=RangeSet(1,nt*ns)
    m.OmegaSet=Set(initialize=W)
    m.Node=Block(m.nodeset)

    for j in m.nodeset:
        m.Node[j].FS=Set(initialize=FS[j])
        m.Node[j].RS=Set(initialize=RS[j])

    m.b=Param(initialize=b)
    m.c=Param(m.Edgeset,initialize=c)
    m.P=Param(m.OmegaSet,initialize=P)
    m.p=Param(m.Edgeset,initialize=p)
    m.q=Param(m.Edgeset,initialize=q)
    m.x=Var(m.Edgeset,within=Binary)
    m.h=Var(m.OmegaSet,within=NonNegativeReals)

    def Obj(m):
        value=sum(m.P[(i,j)]*m.h[(i,j)] for (i,j) in m.OmegaSet) #h[i,j] is the objective of subproblem (i,j)
        return value
    m.Obj=Objective(rule=Obj,sense=minimize)

    def C(m):
        value=sum(m.c[(i,j)]*m.x[(i,j)] for (i,j) in m.Edgeset)
        return value <= b
    m.C=Constraint(rule=C)


    m.Sub=Block(m.OmegaSet)
    for (s,t) in m.OmegaSet:
        m.Sub[(s,t)].sub=SubModel(fixed=m.x)
        m.Sub[(s,t)].sub.stset=Set(initialize=[s,t])
        m.Sub[(s,t)].sub.y=Var(m.Edgeset,within=NonNegativeReals, bounds=(0,1))
        m.Sub[(s,t)].sub.z=Var(m.Edgeset,within=NonNegativeReals, bounds=(0,1))
        def obj(sub):
            value=m.h[(s,t)]
            return value
        m.Sub[(s,t)].sub.obj=Objective(rule=obj, sense=maximize)

        def cb(sub):
            value=sum(m.Sub[(s,t)].sub.y[(i,j)]+m.Sub[(s,t)].sub.z[(i,j)] for (i,j) in m.Node[s].FS)
            return value==1
        m.Sub[(s,t)].sub.cb=Constraint(rule=cb)

        def cc(sub,i):
            value=sum(m.Sub[(s,t)].sub.y[i,j]+m.Sub[(s,t)].sub.z[(i,j)] for (i,j) in m.Node[i].FS)
            value=value - sum(m.p[(j,i)]*m.Sub[(s,t)].sub.y[(j,i)]+m.q[(j,i)]*m.Sub[(s,t)].sub.z[(j,i)] for (j,i) in m.Node[i].RS)
            return value == 0
        m.Sub[(s,t)].sub.cc=Constraint(m.nodeset-m.Sub[(s,t)].sub.stset,rule=cc)

        def cd(sub):
            value = sum (m.p[(j,i)]*m.Sub[(s,t)].sub.y[(j,i)]+m.q[(j,i)]*m.Sub[(s,t)].sub.z[(j,i)] for (j,i) in m.Node[t].RS)
            return m.h[(s,t)]-value == 0
        m.Sub[(s,t)].sub.cd=Constraint(rule=cd)

        def ce(sub,i,j):
            return m.Sub[(s,t)].sub.y[(i,j)] <= 1- m.x[(i,j)]
        m.Sub[(s,t)].sub.ce=Constraint(m.Edgeset,rule=ce)

        def cf(sub,i,j):
            return m.Sub[(s,t)].sub.z[(i,j)] <= m.x[(i,j)]
        m.Sub[(s,t)].sub.cf=Constraint(m.Edgeset,rule=cf)

    m.weights = {m.Sub[k].name+'.sub':v for k,v in m.P.items()}

    return m

if __name__ == "__main__":
    m = pyomo_create_model()
