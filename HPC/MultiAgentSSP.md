
# A multi-agent approach for the SSP

This exercise has as main aim to develop a Java program where
several independent agents will attempt to coordinate their efforts
for the solution of a very well-known combinatorial problem.
The exercise focuses on the Java multithreading implementation: 
it is supposed that the reader is familiar with the notion of 
*threads* as well as with the traditional Java approaches to
multithreaded concurrent programming. Thus, the text below is
going to explain in details the problem and the chosen approach 
for its solution, while the multithreaded Java implementation 
is left to the reader to develop.

## The Subset Sum Problem (SSP)

Given a set $S$ of $n$ positive integers and a target $t$, the
Subset Sum Problem (SSP) asks whether there exists a subset of
$S$ such that the sum of the elements in the subset corresponds
to the target $t$. In other words, if

$$
S = (s_1,s_2,\dots,s_n) ,
$$

the problem asks to find a binary vector $x = (x_1,x_2,\dots,x_n)$
such that

$$
\sum_{i=1}^n x_i s_i = t ,
$$

where every $x_i$ indicates whether the integer $s_i$ takes part to
an SSP solution or not.

## QUBO reformulation of the SSP

Our multi-agent approach is based on a specific reformulation of
the original SSP. To perform this reformulation, we can simply
move the target $t$ on the left-side of the equation above, and 
to square the entire expression:

$$
\left( \sum_{i=1}^n x_i s_i - t \right)^2 = 0 .
$$

Some simple calculations (in order to train your math skills,
please try it yourself!), we can obtain the following equation:

$$
\sum_{i \le j} Q_{ij} x_i x_j + t^2 = 0 ,
$$

where:

$$
Q_{ij} = \left[
\begin{array}{ll}
2 s_i s_j & {\rm if} i \ne j , \\
s_i^2 - 2 t s_i & {\rm otherwise} . \\
\end{array}
\right.
$$

Solving this quadratic equation is not easier than solving the 
original (linear) equation. However, after this transformation,
we can formulate an *optimization* problem where the objective
is to minimize the following quadratic function:

$$
H(x) = \sum_{i \le j} Q_{ij} x_i x_j .
$$

Notice that we can neglect the term $t^2$ obtained above, because 
it is a constant, and hence it doesn't play any role in the optimization 
process.

But how to solve this optimization problem now? You'll find the details
about our approach below; meanwhile, let's begin to develop our Java
class!

By the way, our optimization problem is a **QUBO**, which stands for 
"Quadratic Unconstrained Binary Optimization", where the objective
is to find the binary vector $x$ that better minimizes $H(x)$. The
SSP is only one of the combinatorial problems that can be formulated
as a QUBO.

## Beginning the development of our Java class

So let's begin the development of our Java class. It is recommended that 
you include in the class at least the following attributes:

	private long target;  // the SSP target
	private long[] integers;  // the integers s_i
	private boolean[] solution;  // the x variables (one boolean per variable)
	private double[][] matrix;  // the QUBO matrix Q
	private long error;  // attribute used to store the current error

Your class should also include a constructor that creates a random instance
of the SSP, and generates the associated matrix $Q$. 

In order to create an SSP instance, simply generate random integers having
possible values between 0 and a relatively small $n$ (100 for example), 
and select some of these randomly generated integers (with probability
$\frac{1}{2}$) to construct a feasible target $t$.

After the construction of the QUBO matrix, the constructor should also
normalize its terms, so that all elements of the matrix are finally 
contained in the interval $[-1,1]$.

The Boolean vector (representing a solution to the SSP) is initialized
so that all its elements are set of *false* (i.e. 0).

## What is the error?

Before beginning with the development of our multi-agent approach for solving 
the SSP, let's write a Java method that is capable to compute the error 
caused by a given solution (actually by the one that is currently stored in 
the array ```solution```). The error $e$ can be computed as:

$$
e = t - \sum_{i=1}^n x_i s_i .
$$

Recall that, in the Java code, the integer values $s_i$ are stored in the
array named ```integers```, while the target $t$ is in the attribute ```target```.

## Understanding the QUBO terms

In order to understand how our agents will need to be programmed for 
providing their contribution for the solution of the SSP, let's give a 
deeper look at the terms of our QUBO formulation.

We begin by taking into account a term that is related to a diagonal 
element $Q_{ii}$ of the QUBO matrix. The generic matrix element is:

$$
Q_{ii} = s_i^2 - 2 t s_i ,
$$

and it is multiplied by $x_i$ (in theory it is multiplied by its square,
but the square of a binary variable is equal to its argument). As a 
consequence, when $x_i$ is 1, the value of $Q_{ii}$ contributes to $H(x)$;
it doesn't when $x_i$ is instead 0. If our purpose is to *minimize* the 
value of $H(x)$, therefore, we can consider to set $x_i$ to 1 **only if**
the term $Q_{ii}$ is negative! In other words, a negative $Q_{ii}$ should
encourage us to fix $x_i$ to 1, whereas a positive $Q_{ii}$ should rather
encourage us to fix it to 0.

A similar reasoning can be done for the other terms $Q_{ij}$, where $i$ 
and $j$ are now different. The QUBO matrix element is:

$$
Q_{ij} = 2 s_i s_j ,
$$

and it is multiplied by both $x_i$ and $x_j$. When $Q_{ij}$ is positive,
therefore, it is interesting to set at least one of the two variables
$x_i$ or $x_j$ to 0. Since we obtained our QUBO formulation from an
SSP, we know that these terms must all be positive. The magnitude of the
term contribution in $H(x)$ strictly depends on the numerical values of
the original integers $s_i$ and $s_j$.

Naturally, these remarks on the terms of the QUBO matrix and their
implications on the variables $x_i$ are *only local*, because they 
only take into consideration one term per time. Our agents will act 
exactly in this manner: they will look at their local information,
and they will make a change in the solution vector on the basis of
this local information. The hope is that, when several "local" agents
act in parallel for the solution of the same problem, they manage
to find a *global* agreement where the function $H(x)$ is (globally)
minimized.

## The verif agent

Let's begin now with the implementation of our parallel agents. To
this purpose, inner classes will be included in the main Java class in
order to implement the behavior of each agents. 

The first agent will have the task to (repetitively) verify whether the
current binary vector (stored in the attribute ```solution``` under the
form of an array of Boolean variables) represents, or not, a solution
to the SSP.

For this agent, the entire implementation is provided:

	private class VerifAgent extends Thread
	{
	   @Override
	   public void run()
	   {
	      while (mainClass.this.error != 0L)  mainClass.this.computeError();
	   }
	}

Notice that ```mainClass``` is the name of your main class, while ```computeError```
makes reference to the method you developed above for the computation of
the error.

## The spin agents

For a reason that will become probably evident to some of you only 
when you'll reach the end of this page (no worries, the information is 
not so important at the current stage), the agents to which we assign
the terms $Q_{ii}$ are named *spin agents*.

In order to include these agents in our Java project, we add another inner 
class to the main class, and we make it inherit from ```Thread```:

	private class SpinAgent extends Thread
	{
	   int i;
	   double bias;
	   SpinAgent(int i,double bias)
	   {
	      this.i = i;
	      this.bias = bias;
	   }
	
	   @Override
	   public void run()
	   {
	      ...

In this code snippet, the term $Q_{ii}$ is named ```bias```. It's just a
commonly used name; you can use another name if you prefer. 

The behavior of the agent needs to be implemented in the ```run``` method.
The agent will first of all verify what is the "encouraged" value for $x_i$
from its corresponding $Q_{ii}$, and then store it in a local variable. As 
long as the SSP solution has not been found yet, the agent will keep verifying
whether its $x_i$ corresponds to the encouraged value, and if not, it will
set it to the encouraged value with a probability that depends directly
on the absolute value of the *bias* (recall the QUBO terms have been normalized
in the constructor).

## The coupling agents

It's time now to implement the *coupling* agents. At this point, you have enough 
experience with both the problem and the multithreading implementation approach,
so ... the implementation of these agents is left totally to you.

## Our multi-agent solver

Back to the main class, we can now implement the following method:

	public void solve() throws InterruptedException

The method will be in charge of initializing and launching all involved
threads. It will make sure that the ```verif``` thread will have finished 
the execution before returning. 

## Synchronization issues

Even if it may not seem the case, your implementations could suffer of 
synchronization issues. Even if most of the agents act locally, there is
a special variable that needs to be accessed by all of them (which is it?),
while spin and coupling agents may require to have access, exactly at
the same time, to the same binary variables. Maybe you cannot see this 
so obviously, but a bad implementation may even lead to a deadlock ...

Please make sure that your final implementation has no synchronization
issues. Of course, the implementation solution where your multithreaded
program reduces to a sequential program is not acceptable!

## Some final comments

The SSP is used also in the [first lecture](../GPU/intro_ssp.md)
about GPU programming. Another approach to solve the combinatorial
problem is employed in that lecture.

Were you wondering why the letter $H$ was employed in the QUBO reformulation
for making reference to the objective function of our optimization problem?
Well, the function $H(x)$ is usually referred to as "the Hamiltonian", after 
Hamilton, a famous physicist. The Hamiltonian represents the total energy in 
a closed physical system. You may wonder at this point what is the relationship
between this energy and our combinatorial problem ... Well, an approach to the 
SSP (and to several other combinatorial problems that admit a QUBO reformulation) 
is to find the minimum energy of the Hamiltonian $H(x)$ by using *adiabatic 
quantum computing* (see for example the machines developed by the 
[D-wave](https://www.dwavequantum.com/) company. In this quantum approach,
the binary variables are encoded via *qbits*, their values are controled by 
the *spin* of these qbits, while the coupling effects between pairs of qbits 
are regulated through the *entanglement* phenomenon.

---------

