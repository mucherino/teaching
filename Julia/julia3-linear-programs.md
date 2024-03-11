
# Solving linear programs with Julia

In this lesson, we are going to write and solve a linear program, which is 
an optimization problem where the objective function and constraints are 
linear functions. For the theory behind linear programs, you can refer to
this [wikipedia page](https://en.wikipedia.org/wiki/Linear_programming).
We remark that, for this kind of problems, efficient and widely tested 
methods exist, whose implementations are also available in 
[Julia](https://julialang.org/).

As in the [previous lesson](./julia2-linear-systems.md), the problem,
as well as an approach to solve it in [Julia](https://julialang.org/),
are introduced through an example.

## The diet problem

This is a classical problem often used in lectures to introduce linear 
programming. We are going to consider a slightly modified version.

Suppose you often have lunch in a restaurant where the payment of your meal 
is based on the weight of the food you place on your plate. This restaurant
offers 4 different kinds of dishes, and you are free to serve yourself, 
with the quantity of food from each dish that you like. One constraint,
of course, is given by the size of the plate you keep in your hands while 
serving yourself.

The restaurant changes its offer very week, and every week the information 
about the composition and weight of the dishes is given. Although you generally 
like all of them, your physician has suggested you to reduce the quantity of 
pasta and rice you eat every day, and hence your choice should mainly be for 
dishes richer in meat or fish. However, especially when meat is an ingredient, 
the weight of the food that you finally collect on your plate increases, and 
as a consequence the price to pay is more important. 

The 4 meal offers for the upcoming week were just published, together with
their composition and weights. We can use these data to define the following
arrays in [Julia](https://julialang.org/):

	meat = [0.6,0.0,0.3,0.3];
	fish = [0.0,0.5,0.3,0.3];
	rice = [0.1,0.5,0.0,0.3];
	pasta = [0.3,0.0,0.4,0.1];
	weight = [1.0,0.8,0.6,0.7];

For example, ```meat[1] = 0.6``` indicates that the 60% of the composition
of the dish labeled with 1 is meat; similarly we have the same information
for the other main ingredients. Finally, the weights are expressed in units
of 100 grams, and they represent the dish weight when it entirely fills
the plate.

Now, your main problem consists in reducing the total cost of your lunch
bill, while some constraints on the food that you choose need to be imposed.

Let us introduce some real variables $x$ to indicate how much of the 4 proposed
dishes you should put on your plate. Since these quantities can be expressed in
terms of percentages (you don't take any of dish 1, then $x_1 = 0$; otherwise
you entirely fill the plate with dish 2, then $x_2 = 1$; and so on), we can
suppose that these variables are constrained in the real interval $[0,1]$.
In order to make our model easily extendable, we introduce the constant $n = 4$,
which corresponds to the number of proposed dishes, but also to the number of
ingredient categories. Our *objective function* looks like:

$$
f(x) = \sum_{i=1}^n {\tt weight[i]} \cdot x_i .
$$

We can remark that this function is linear, and hence it can be easily
optimized (either minimized, or maximized). However, if we attempted optimizing
it without including constraints, the solution of the minimization problem would 
consist of the 4 variables $x_i$ all equal to 0, while the solution to the 
maximization problem would bring all these variables to 1. The suggestions from 
your physician, together with your own wishes (as long as they don't contradict 
the physician suggestions), can help you define a set of constraints on the set
of variables $x$. For example:

$$
\left\{
\begin{array}{l}
\displaystyle\sum_{i=1}^n {\tt meat[i]} \cdot x_i \ge 0.3 , \\
\displaystyle\sum_{i=1}^n {\tt fish[i]} \cdot x_i \ge 0.6 , \\
\displaystyle\sum_{i=1}^n {\tt rice[i]} \cdot x_i \le 0.4 , \\
\displaystyle\sum_{i=1}^n {\tt pasta[i]} \cdot x_i \le 0.3 .
\end{array}
\right.
$$

Finally, we need to make sure that the sum of percentages $x_i$ do not end
up summing to a value greater than 1:

$$
\sum_{i=1}^n x_i \le 1 .
$$

After the introduction of these constraints, finding the optimal values for 
each variable $x_i$ does not seem to be so trivial! The function $f$, together 
with the constraints above, form a linear program. Fortunately, there exist tools 
that are able to efficiently find solutions to these linear programs for us, and 
they are integrated in [Julia](https://julialang.org/). 

The first thing to do to this purpose is to give the details of our linear 
program to [Julia](https://julialang.org). This can be done by using the package 
```JuMP```:

	using JuMP

At the time the linear program is instantiated, it is convenient to indicate which 
is the software tool (also integrated in [Julia](https://julialang.org)) that we 
are going to use for its solution. Our first choice falls on the ```Clp``` solver:

	using Clp

At this point, our linear program, named ```diet```, can be instantiated 
with the following syntax:

	diet = Model(Clp.Optimizer);

At this stage, the ```diet``` model is still empty. In order to add variables, 
we can use the macro ```@variable```, as follows:

	@variable(diet,x[1:n]);

where we suppose that ```n = 4```. We can notice that our vector ```x``` of variables 
is represented in [Julia](https://julialang.org) as a standard vector. But we need
to pay attention to the fact that the type of its elements does not correspond to
a standard ```Int64``` or ```Float64```, it is rather ```VariableRef```. This special
type indicates that the vector ```x``` collects variables in a model, and we cannot 
treat them as generic numbers:

	julia> x
	4-element Vector{VariableRef}:
	 x[1]
	 x[2]
	 x[3]
	 x[4]

Variables refer by default to *real* numbers, spanning from $-\infty$ to $+\infty$. 
However, our variables actually represent percentages, so we need to add the following
bounds:

	set_lower_bound.(x,0.0);
	set_upper_bound.(x,1.0);

Notice the use of the little dot "```.```". It indicates that the operation (setting
up the bounds) needs to be performed on all the elements of the vector ```x```.

The objective function can be added to the model with the following syntax:

	@objective(diet, Min, sum( weight[i]*x[i] for i in 1:n ));

Notice that ```Min``` indicates that the objective function is to be minimized.
After the definition, we can visualize the objective function as follows:

	julia> objective_function(diet)
	x[1] + 0.8 x[2] + 0.6 x[3] + 0.7 x[4]

Finally, we can include the constraints to our ```diet``` model by using the 
```@constraint``` macro:

	@constraint(diet,Meat, sum( meat[i]*x[i] for i in 1:n ) >= 0.3);
	@constraint(diet,Fish, sum( fish[i]*x[i] for i in 1:n ) >= 0.6);
	@constraint(diet,Rice, sum( rice[i]*x[i] for i in 1:n ) <= 0.4);
	@constraint(diet,Pasta, sum( pasta[i]*x[i] for i in 1:n) <= 0.3);
	@constraint(diet,Plate, sum( x[i] for i in 1:n ) <= 1.0);

For the visualization of these constraints, we can simply type their names
in [Julia](https://julialang.org)'s prompt. For example:

	julia> Meat
	Meat : 0.6 x[1] + 0.3 x[3] + 0.3 x[4] >= 0.2

or:

	julia> Plate
	Plate : x[1] + x[2] + x[3] + x[4] <= 1

Now our linear program is ready! We can solve it by invoking:

	JuMP.optimize!(diet)

where the exclamation mark indicates that [Julia](https://julialang.org) is allowed 
to make changes to the arguments passed to the function ```optimize```. The result 
is directly printed on the screen:

	Coin0506I Presolve 5 (0) rows, 4 (0) columns and 16 (0) elements
	Clp0006I 0  Obj 0 Primal inf 0.93333313 (2)
	Clp0006I 3  Obj 0.58222222
	Clp0000I Optimal - objective value 0.58222222
	Clp0032I Optimal objective 0.5822222222 - 3 iterations time 0.002

We can notice that some information about the optimization process are automatically 
printed. For example, we can read that the solver needed to iterate only 3 times: 
our problem is after all quite easy to solve. The value of the objective function 
$f(x)$ in the found solution is about 0.582, but how much of each dish should we 
finally take? This information is stored in the variables. As mentioned earlier,
however, we cannot have a direct access to the content of the variable by simply 
typing ```x```. It is necessary to use a specific ```JuMP``` function:

	for i in 1:n
	  println("x[$i] = ",JuMP.value(x[i]))
	end

The entire code in [Julia](https://julialang.org) for the definition and the solution
of this linear program can be found in the file [diet.jl](./diet.jl).

## Links

* [Next: adaptive maps](./julia4-adaptive-maps.md)
* [Summary](./README.md)

