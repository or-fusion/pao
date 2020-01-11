\* Source Pyomo model name=unknown *\

max 
o:
+ [
+12 x1 * z
+8 x2 * z
+4 x3 * z
] / 2

s.t.

c_e_sub_dual_equiv_objs_:
+5 sub_dual_c1
+3 sub_dual_c2
+4 sub_dual_c3
+ [
-6 x1 * z
-4 x2 * z
-2 x3 * z
]
= 0

c_u_sub_dual_x1_:
+4 sub_dual_c1
+1 sub_dual_c2
-6 z
<= 0

c_u_sub_dual_x2_:
+2 sub_dual_c1
+1 sub_dual_c2
+1 sub_dual_c3
-4 z
<= 0

c_u_sub_dual_x3_:
+1 sub_dual_c1
+1 sub_dual_c3
-2 z
<= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x1 <= +inf
   0 <= x2 <= +inf
   0 <= x3 <= +inf
    -inf <= z <= 0
   0 <= sub_dual_c1 <= +inf
   0 <= sub_dual_c2 <= +inf
   0 <= sub_dual_c3 <= +inf
end
