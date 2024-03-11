
# Linear program for Adaptive Maps
#
# n, p, q and D need to be defined in Julia before running this script
#
# AM

using JuMP;
using Clp;

# defining the linear program
admaps = Model(Clp.Optimizer);
@variable(admaps,x[1:n] >= 0)
@variable(admaps,y[1:n] >= 0)
@variable(admaps,z[1:n,1:n] >= 0)
for i in 1:n
   for j in 1:n
      if i < j
         North = q[i] - q[j] <= 0.0;
         South = !North;
         West = p[i] - p[j] >= 0.0;
         East = !West;
         if North && West
            @constraint(admaps,c1,z[i,j] >= y[j] - y[i] + x[i] - x[j] - D[i,j])
            @constraint(admaps,c2,z[i,j] >= D[i,j] - y[j] + y[i] - x[i] + x[j])
         elseif North && East
            @constraint(admaps,c1,z[i,j] >= y[j] - y[i] + x[j] - x[i] - D[i,j])
            @constraint(admaps,c2,z[i,j] >= D[i,j] - y[j] + y[i] - x[j] + x[i])
         elseif South && West
            @constraint(admaps,c1,z[i,j] >= y[i] - y[j] + x[i] - x[j] - D[i,j])
            @constraint(admaps,c2,z[i,j] >= D[i,j] - y[i] + y[j] - x[i] + x[j])
         elseif South && East
            @constraint(admaps,c1,z[i,j] >= y[i] - y[j] + x[j] - x[i] - D[i,j])
            @constraint(admaps,c2,z[i,j] >= D[i,j] - y[i] + y[j] - x[j] + x[i])
         end
         unregister(admaps,:c1)
         unregister(admaps,:c2)
      end
   end
end
@objective(admaps,Min,sum(z[i,j] for i in 1:n, j in 1:n))

# solving the linear program
JuMP.optimize!(admaps)
xc = zeros(n);
yc = zeros(n);
for i in 1:n
  xc[i] = JuMP.value(x[i]);
  yc[i] = JuMP.value(y[i]);
end

# printing the result
println("Optimal Solution:")
println("x = ",xc);
println("y = ",yc);

