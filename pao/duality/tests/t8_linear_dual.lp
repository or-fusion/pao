5 Var Declarations
    c : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :  None : False :  True :  Reals
    x_upper_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals
    y_lower_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :  None :  None : False :  True : NonNegativeReals
    z_lower_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :  None :  None : False :  True : NonNegativeReals
    z_upper_ : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :     0 : False :  True : NonPositiveReals

1 Objective Declarations
    o : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : minimize : - x_upper_ - z_upper_ + z_lower_ + y_lower_ + u

3 Constraint Declarations
    x : Size=1, Index=None, Active=True
        Key  : Lower : Body             : Upper : Active
        None :  -Inf : c + x_upper_ + 1 :   0.0 :   True
    y : Size=1, Index=None, Active=True
        Key  : Lower : Body             : Upper : Active
        None :   0.0 : - u*c + y_lower_ :  +Inf :   True
    z : Size=1, Index=None, Active=True
        Key  : Lower : Body                      : Upper : Active
        None :   0.0 : - c + z_upper_ + z_lower_ :   0.0 :   True

9 Declarations: c x_upper_ z_upper_ z_lower_ y_lower_ o x z y
