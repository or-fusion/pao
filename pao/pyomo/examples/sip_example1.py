#
# A stochastic bilevel interdiction adapted from:
#   "A Stochastic Program for Interdicting Smuggled Nuclear Material"
#   by F. Pan, W. S. Charlton, D. P. Morton
#   DRAFT August 19, 2002
#
# example contributor: She'ifa Punla-Green
#
import pyomo.environ as pe
from pao.pyomo import *

#A incidence matrix
#n number of nodes
#e number of edges
#b budget
#c cost of edge sensor placement
#ns number of starting nodes
#nt number of ending nodes

def create():
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

    m = pe.ConcreteModel()
    m.nodeset = pe.RangeSet(1, n)
    m.edgeset = pe.RangeSet(1, e)
    m.Edgeset = pe.Set(initialize=E)
    m.Omega = pe.RangeSet(1, nt*ns)
    m.OmegaSet = pe.Set(initialize=W)
    m.Node = pe.Block(m.nodeset)

    for j in m.nodeset:
        m.Node[j].FS = pe.Set(initialize=FS[j])
        m.Node[j].RS = pe.Set(initialize=RS[j])

    m.b=pe.Param(initialize=b)
    m.c=pe.Param(m.Edgeset,initialize=c)
    m.P=pe.Param(m.OmegaSet,initialize=P)
    m.p=pe.Param(m.Edgeset,initialize=p)
    m.q=pe.Param(m.Edgeset,initialize=q)
    m.x=pe.Var(m.Edgeset,within=pe.Binary)
    m.h=pe.Var(m.OmegaSet,within=pe.NonNegativeReals)

    def Obj(m):
        value=sum(m.P[i,j]*m.h[i,j] for (i,j) in m.OmegaSet) #h[i,j] is the objective of subproblem (i,j)
        return value
    m.Obj = pe.Objective(rule=Obj, sense=pe.minimize)

    def C(m):
        value=sum(m.c[i,j]*m.x[i,j] for (i,j) in m.Edgeset)
        return value <= b
    m.C=pe.Constraint(rule=C)


    m.Sub = pe.Block(m.OmegaSet)
    for (s,t) in m.OmegaSet:
        m.Sub[s,t].sub = SubModel(fixed=m.x)
        m.Sub[s,t].sub.stset = pe.Set(initialize=[s,t])
        m.Sub[s,t].sub.y = pe.Var(m.Edgeset,within=pe.NonNegativeReals, bounds=(0,1))
        m.Sub[s,t].sub.z = pe.Var(m.Edgeset,within=pe.NonNegativeReals, bounds=(0,1))
        def obj(sub):
            return m.h[s,t]
        m.Sub[(s,t)].sub.obj = pe.Objective(rule=obj, sense=pe.maximize)

        def cb(sub):
            value=sum(m.Sub[(s,t)].sub.y[(i,j)]+m.Sub[(s,t)].sub.z[(i,j)] for (i,j) in m.Node[s].FS)
            return value==1
        m.Sub[(s,t)].sub.cb=pe.Constraint(rule=cb)

        def cc(sub,i):
            value=sum(m.Sub[(s,t)].sub.y[i,j]+m.Sub[(s,t)].sub.z[(i,j)] for (i,j) in m.Node[i].FS)
            value=value - sum(m.p[(j,i)]*m.Sub[(s,t)].sub.y[(j,i)]+m.q[(j,i)]*m.Sub[(s,t)].sub.z[(j,i)] for (j,i) in m.Node[i].RS)
            return value == 0
        m.Sub[(s,t)].sub.cc=pe.Constraint(m.nodeset-m.Sub[(s,t)].sub.stset,rule=cc)

        def cd(sub):
            value = sum (m.p[(j,i)]*m.Sub[(s,t)].sub.y[(j,i)]+m.q[(j,i)]*m.Sub[(s,t)].sub.z[(j,i)] for (j,i) in m.Node[t].RS)
            return m.h[(s,t)]-value == 0
        m.Sub[(s,t)].sub.cd=pe.Constraint(rule=cd)

        def ce(sub,i,j):
            return m.Sub[(s,t)].sub.y[(i,j)] <= 1- m.x[(i,j)]
        m.Sub[(s,t)].sub.ce=pe.Constraint(m.Edgeset,rule=ce)

        def cf(sub,i,j):
            return m.Sub[(s,t)].sub.z[(i,j)] <= m.x[(i,j)]
        m.Sub[(s,t)].sub.cf=pe.Constraint(m.Edgeset,rule=cf)

    m.weights = {m.Sub[k].name+'.sub':v for k,v in m.P.items()}

    return m

if __name__ == "__main__":
    M = create()

    opt = Solver('pao.pyomo.FA')
    opt.solve(M, solver='gurobi', tee=True)

    print(M.x)
    M.x.pprint()
    print(M.h)
    M.h.pprint()

    for s,t in M.OmegaSet:
        print(M.Sub[s,t])
        M.Sub[s,t].sub.y.pprint()
        M.Sub[s,t].sub.z.pprint()

    #    self.assertTrue(math.isclose(M.xR.value, 2))
    #    self.assertTrue(math.isclose(M.L.xR.value, 100))
