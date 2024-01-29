# Linear model for the diet problem
using JuMP
using Clp
print("Linear diet example\n");
n = 4;
meat = [0.6,0.0,0.3,0.3];
fish = [0.0,0.5,0.3,0.3];
rice = [0.1,0.5,0.0,0.3];
pasta = [0.3,0.0,0.4,0.1];
weight = [1.0,0.8,0.6,0.7];
diet = Model(Clp.Optimizer);
@variable(diet,x[1:n]);
set_lower_bound.(x,0.0);
set_upper_bound.(x,1.0);
@objective(diet, Min, sum( weight[i]*x[i] for i in 1:n ));
@constraint(diet,Meat, sum( meat[i]*x[i] for i in 1:n ) >= 0.2);
@constraint(diet,Fish, sum( fish[i]*x[i] for i in 1:n ) >= 0.3);
@constraint(diet,Rice, sum( rice[i]*x[i] for i in 1:n ) <= 0.3);
@constraint(diet,Pasta, sum( pasta[i]*x[i] for i in 1:n) <= 0.2);
@constraint(diet,Plate, sum( x[i] for i in 1:n ) <= 1.0);
JuMP.optimize!(diet)
println("Optimal Solution:")
for i in 1:n
  println("x[$i] = ",JuMP.value(x[i]))
end

