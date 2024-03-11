
# Linear model for the coins' example (with binary variables!)
#
# n, C and nC need to be defined before running this script
#
# AM

using JuMP
using GLPK
print("Coins example with binary variables\n");
model = Model(GLPK.Optimizer);
@variable(model,x[1:N] >= 0,Int)
@objective(model, Max, sum( x[i] for i in 1:n))
@constraint(model,constr1, sum( C[i]*x[i] for i in 1:n) == 11.4 )
@constraint(model,constr2, sum( x[i] for i in 1:n) <= 20 )
@constraint(model,constr3[i in 1:n], x[i] <= nC[i])
JuMP.optimize!(model)
objval = 0.0;
println("Optimal Solution:")
for i in 1:n
  val = JuMP.value(x[i]);
  println("x[$i] = ",val);
  global objval = objval + val;
end
println("Objective function in the solution: $objval");

