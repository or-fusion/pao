\* Source Pyomo model name=unknown *\

min 
o:
+ [
+6 x1 * z
+5 x2 * z
] / 2

s.t.

c_e_sub_dual_equiv_objs_:
-100 sub_dual_c1
-100 sub_dual_c2
-100 sub_dual_c3
-100 sub_dual_c4
+ [
-3 x1 * z
-2.5 x2 * z
]
= 0

c_u_sub_dual_x1_:
+4.4400000000000004 sub_dual_c1
+4 sub_dual_c3
+3 sub_dual_c4
+3 z
<= 0

c_u_sub_dual_x2_:
+6.6699999999999999 sub_dual_c2
+2.8599999999999999 sub_dual_c3
+6 sub_dual_c4
+2.5 z
<= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x1 <= +inf
   0 <= x2 <= +inf
   1 <= z <= 2
    -inf <= sub_dual_c1 <= 0
    -inf <= sub_dual_c2 <= 0
    -inf <= sub_dual_c3 <= 0
    -inf <= sub_dual_c4 <= 0
end
