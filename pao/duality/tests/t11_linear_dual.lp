\* Source Pyomo model name=Test *\

min 
o:
-5 c1
-3 c2(2)
-4 c3(4_4)
+3 c5_lb_
+7 c5_ub_
+15 c6(7)
+8 c6_lb_(6)
+12 c6_ub_(6)
-11 x2_upper_
+13 x3_lower_
-13 x3_upper_
-14 x5_lower_

s.t.

c_u_x1_:
+4 c1
+1 c2(2)
-4 c5_lb_
-4 c5_ub_
-12 c6(7)
-9 c6_lb_(6)
-9 c6_ub_(6)
<= -6

c_e_x2_:
+2 c1
+1 c2(2)
+1 c3(4_4)
+1 x2_upper_
= -4

c_e_x3_:
+1 c1
+1 c3(4_4)
+1 x3_lower_
+1 x3_upper_
= -2

c_e_x4(4_4)_:
+101 c1
+103 c3(4_4)
-5 c5_lb_
-5 c5_ub_
-13 c6(7)
-10 c6_lb_(6)
-10 c6_ub_(6)
= 0

c_e_x5_:
+105 c1
+1 x5_lower_
= 0

c_l_y(1)_:
+14 c1
>= -7

c_l_y(2)_:
+15 c2(2)
>= 0

c_l_y(3)_:
+16 c3(4_4)
-6 c5_lb_
-6 c5_ub_
-14 c6(7)
-11 c6_lb_(6)
-11 c6_ub_(6)
>= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= c1 <= +inf
    -inf <= c2(2) <= +inf
    -inf <= c3(4_4) <= 0
   0 <= c5_lb_ <= +inf
    -inf <= c5_ub_ <= 0
    -inf <= c6(7) <= +inf
   0 <= c6_lb_(6) <= +inf
    -inf <= c6_ub_(6) <= 0
    -inf <= x2_upper_ <= 0
   0 <= x3_lower_ <= +inf
    -inf <= x3_upper_ <= 0
   0 <= x5_lower_ <= +inf
end
