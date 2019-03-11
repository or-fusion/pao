\* Source Pyomo model name=Test *\

min 
o:
-5 c1
-3 c2
-4 c3
-11 x2_upper_
+13 x3_lower_
-13 x3_upper_

s.t.

c_u_x1_:
+4 c1
+1 c2
<= -6

c_e_x2_:
+2 c1
+1 c2
+1 c3
+1 x2_upper_
= -4

c_e_x3_:
+1 c1
+1 c3
+1 x3_lower_
+1 x3_upper_
= -2

c_e_x4_:
+101 c1
+103 c3
= 0

c_l_y(1)_:
+14 c1
>= -7

c_l_y(2)_:
+15 c2
>= 0

c_l_y(3)_:
+16 c3
>= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= c1 <= +inf
    -inf <= c2 <= +inf
    -inf <= c3 <= 0
    -inf <= x2_upper_ <= 0
   0 <= x3_lower_ <= +inf
    -inf <= x3_upper_ <= 0
end
