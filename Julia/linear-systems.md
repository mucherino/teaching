
# Solving linear systems with Julia

We introduce the problem of solving a system of linear equations with an example.
The content below is part of a Numerical Analysis course, whose slides can be 
[downloaded here](https://www.antoniomucherino.it/download/slides/NumAnalysis.pdf).
The analytic solution to the linear system defined below can also be found on those 
slides (but not in this lecture).

## Flying with or against wind

Suppose an aircraft flies from Paris to Rio, and then it comes back. Suppose the wind 
is constant during the whole travel and it is able to influence the speed of the aircraft.  

We have the following information:

- the distance $d$ covered by the aircraft is 5700 miles;
- when flying from Paris to Rio, our aircraft is flying in the same direction as the 
  wind, and the time $t_1$ is 5.1 hours;
- when flying back instead, i.e. from Rio to Paris, the aircraft is flying in the opposite
  direction of the wind, and the time $t_2$ is 4.7 hours;
- you may have noticed that we are dealing with a supersonic aircraft, but this fact will
  not influence our methodology;
- you may also wonder whether earth rotation has an impact on the flying time: yes it does.
  But in order to make things easier here, we will neglect this detail.

Our aim is to answer to the following question. 
*By exploiting the available data, how can we find the average speed of the aircraft, as well
 as the average speed of the wind?* 

Let $x$ be the average speed of the aircraft, and let $y$ be the average speed of the wind.
We can remark that:

- the actual aircraft speed is $x + y$ when it flies with the wind;
- the actual aircraft speed is $x - y$ when it flies against the wind;
- the distance $d$ covered during each travel can be computed as the product between the
  time ($t_1$ or $t_2$) and the actual speed.

Therefore, we can define the following system of equations, where $t_1$, $t_2$ and $d$ are
parameters, and where $x$ and $y$ are the unknowns (the variables):

$$
\left[
\begin{array}{l}
t_1 (x - y) = d , \\
t_2 (x + y) = d . \\
\end{array}
\right.
$$

This system is so small that it may be solved analytically. However, how to deal with much
larger systems, involving many variables and equations? Several methods for solving linear
systems have been proposed in the past years, and it is out of the scope of this course to
go through them. Below, a simple program in [Julia](https://julialang.org/) for the 
solution of this linear system is presented.

## Julia program for linear systems

Linear systems can be expressed in [Julia](https://julialang.org/) with a matrix and a 
vector (exactly the same data structures used in our [previous lesson](./julia1-basics.md)).
In particular, every row of the matrix $A$ represents the coefficients of the 
corresponding equation, whose constant terms are rather collected in the vector, here
denoted with the symbol $d$. In case you're not really familiar with systems of linear 
equations, there is a lot of online material that can help you. See for example this 
[wikipedia page](https://en.wikipedia.org/wiki/System_of_linear_equations).

In order to define our system in [Julia](https://julialang.org/), therefore, we can 
simply define the matrix

	A = [5.1 -5.1; 4.7 4.7];

and the vector

	d = [5700.0,5700.0];

The data are now in memory, and only two additional steps are required for solving the 
linear system. First, we need to define the problem, which is, the problem of solving the 
linear system specified by the matrix $A$ and the vector $d$. In [Julia](https://julialang.org/),
the *problem* is by itself a data structure, and can be constructed by invoking the specific 
function:

	lp = LinearProblem(A,d);

It's the variable ```lp``` that contains the information about the system of linear 
equations. In order to solve it, we can simply invoke the function ```solve```:

	sol = solve(lp);

Since all our commands end with a semi-colon, no information has been printed on
the screen so far. In order to visualize the solution of the linear system, we can
simple type ```sol``` in the [Julia](https://julialang.org/)'s prompt:

	u: 2-element Vector{Float64}:
	 1165.2065081351689
	   47.55944931163945

So, it turns out that the aircraft's speed is about 1165 miles per hour, while the
speed of the wind is about 47 miles per hour. Notice the data type of the solution 
vector: it is ```Float64```, which is equivalent to ```double``` in C and other
programming languages.

An alternative way to solve the linear system is to collect all these commands in a 
[Julia](https://julialang.org/)'s program (see [flying-system.jl](./flying-system.jl) 
file), and launch the execution as follows:

	$ julia flying-system.jl 
	The solution is [1165.2065081351689, 47.55944931163945]

In both cases, we should not forget to indicate to [Julia](https://julialang.org/) 
that we need to use the special package for solving linear problems:

	using LinearSolve

We can remark that [Julia](https://julialang.org/) offers a rather simple way to 
define and solve linear systems. 

## Links

* [Next: linear programs](./linear-programs.md)
* [Back to math programming lectures](./README.md)
* [Back to main repository page](../README.md)

