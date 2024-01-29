# full solution for example in julia2-linear-systems.md
using LinearSolve
A = [5.1 -5.1; 4.7 4.7];
d = [5700.0,5700.0];
lp = LinearProblem(A,d);
sol = solve(lp);
println("The solution to the 'flying' linear system is ",sol);

