2 Var Declarations
    c1 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :  None :  None : False :  True : NonNegativeReals
    c2 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :  None :  None : False :  True : NonNegativeReals

1 Objective Declarations
    o : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : maximize : (5.0 - (2*x2 + b.x1))*c1 + (3.0 - x2)*c2

1 Constraint Declarations
    x1 : Size=1, Index=None, Active=True
        Key  : Lower : Body          : Upper : Active
        None :  -Inf : 4*c1 + c2 - 6 :   0.0 :   True

4 Declarations: c1 c2 o x1
