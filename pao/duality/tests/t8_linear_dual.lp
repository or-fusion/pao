3 Var Declarations
    c : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :  None : False :  True :  Reals
    x_upper_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals
    y_upper_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals

1 Objective Declarations
    o : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : minimize : - x_upper_ - y_upper_ + u

2 Constraint Declarations
    x : Size=1, Index=None, Active=True
        Key  : Lower : Body             : Upper : Active
        None :  -Inf : c + x_upper_ + 1 :   0.0 :   True
    y : Size=1, Index=None, Active=True
        Key  : Lower : Body             : Upper : Active
        None :  -Inf : - u*c + y_upper_ :   0.0 :   True

6 Declarations: c x_upper_ y_upper_ o x y
