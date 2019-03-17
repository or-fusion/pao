\* Source Pyomo model name=Test *\

max 
o:
+5 c1
+3 c2

s.t.

c_u_x1_:
+4 c1
+1 c2
<= 6

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= c1 <= +inf
   0 <= c2 <= +inf
end
