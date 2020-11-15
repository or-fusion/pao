#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *
from pao.bilevel import *
from pao.bilevel.plugins.collect import BilevelMatrixRepn

# Test of RHS vector in matrix_repn
#
# Near Optimal Robust Bilevel Optimization
# By M. Besancon, M. Anjos, L. Brotcorne


M = ConcreteModel()
M.x=Var(within=NonNegativeReals,bounds=(0,10000))
M.y=Var(within=NonNegativeReals,bounds=(0,10000))
M.v=Var(within=NonNegativeReals,bounds=(0,10000))
M.c1 = Constraint(expr=-M.x+4*M.v <= 11)
M.c2 = Constraint(expr= M.x+2*M.v <= 13)
M.o = Objective(expr=M.x+10*M.v)

M.sub = SubModel(fixed=(M.x))
M.sub.o  = Objective(expr=M.v, sense=minimize)
M.sub.c3 = Constraint(expr=-2*M.x - M.v <= -5)
M.sub.c4 = Constraint(expr= 5*M.x - 4*M.v <= 30)
M.sub.c5 = Constraint(expr=-2*M.x - M.v -M.y <= -3)
M.sub.c6 = Constraint(expr=-2*M.x + M.y <= 0)


matrix_repn = BilevelMatrixRepn(M)
print('---------------------')
print('--Variable Vector----')
print('---------------------')
_name = [var.name for idx,var in matrix_repn._all_vars.items()]
print(_name)
print('\n')
print('---------------------')
print('-Grouped By Variable-')
print('---------------------')
for submodel in M.component_objects(SubModel):
    for var in M.component_objects(Var):
        (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
        (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
        print('---------------------')
        print('submodel name: {}'.format(submodel.name))
        print('variable name: {}'.format(var.name))
        print('A linear matrix coefficients: ')
        print(A)
        print('A bilinear matrix coefficients: ')
        print(A_q.toarray())
        print('Sense (=, <=, >=): ')
        print(sign)
        print('Rhs: ')
        print(b)
        print('C linear coefficient: ')
        print(C)
        print('C bilinear vector coefficients: ')
        print(C_q)
        print('C constant coefficient: ')
        print(C_constant)
print('\n')
print('---------------------')
print('--Grouped By Sense---')
print('---------------------')
for submodel in M.component_objects(SubModel):
    for var in M.component_objects(Var):
        for sense in ['e', 'l']:
            (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var, sense=sense)
            print('---------------------')
            print('submodel name: {}'.format(submodel.name))
            print('variable name: {}'.format(var.name))
            print('A linear matrix coefficients: ')
            print(A)
            print('A bilinear matrix coefficients: ')
            print(A_q.toarray())
            print('Sense (=, <=, >=): ')
            print(sign)
            print('Rhs: ')
            print(b)

