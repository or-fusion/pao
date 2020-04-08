4 Var Declarations
    c1 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals
    c2 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals
    c3 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals
    c4 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals

1 Objective Declarations
    o : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : minimize : 3*y[1] + 2.5*y[2] - (100.0 - y[1])*c1 - (100.0 - y[2])*c2 - 100.0*c3 - 100.0*c4

2 Constraint Declarations
    x[1] : Size=1, Index=None, Active=True
        Key  : Lower : Body                      : Upper : Active
        None :  -Inf : 4.44*c1 + 4*c3 + 3*c4 + 3 :   0.0 :   True
    x[2] : Size=1, Index=None, Active=True
        Key  : Lower : Body                           : Upper : Active
        None :  -Inf : 6.67*c2 + 2.86*c3 + 6*c4 + 2.5 :   0.0 :   True

7 Declarations: c1 c2 c3 c4 o x[1] x[2]
