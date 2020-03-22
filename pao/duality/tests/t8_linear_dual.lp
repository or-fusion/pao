5 Var Declarations
    c : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :  None : False :  True :  Reals
    x_lower_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :  None :  None : False :  True : NonNegativeReals
    x_upper_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals
    y_lower_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :  None :  None : False :  True : NonNegativeReals
    y_upper_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals

1 Objective Declarations
    o : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : minimize : - x_upper_ - y_upper_

2 Constraint Declarations
    x : Size=1, Index=None, Active=True
        Key  : Lower : Body                        : Upper : Active
        None :   0.0 : c + x_upper_ + x_lower_ + 1 :   0.0 :   True
    y : Size=1, Index=None, Active=True
        Key  : Lower : Body                        : Upper : Active
        None :   0.0 : - u*c + y_upper_ + y_lower_ :   0.0 :   True

8 Declarations: c x_upper_ x_lower_ y_upper_ y_lower_ o x y
