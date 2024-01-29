
# Julia course

This course is intended to be an introduction to the use of [Julia](https://julialang.org/)
programming language. 

The course begins with several examples, where the main objective is to show how 
writing programs in [Julia](https://julialang.org/) can be much less tedious 
than programming at lower level. We begin by exploring the syntax for storing
vectors and matrices, and for performing some basic *linear algebra* operations 
with them. The solution of *linear systems* and *linear programs* are at the 
center of the examples given thereafter. 

We can remark that some of the discussed functionalities are very similar (they 
actually influenced [Julia](https://julialang.org/)'s syntax) to those provided by 
[Matlab](https://www.mathworks.com/products/matlab.html),
[Scilab](https://www.scilab.org/), and other software tools. In some occasions,
a syntax comparison with C programming language is given during the lectures. 
In C in fact, there are specific implementations for linear algebra, mainly the 
[BLAS](http://www.netlib.org/blas/) library, which provides some basic matrix
operations, and [LAPACK](http://www.netlib.org/lapack/) for more advanced operations,
such as the solution to linear systems. The good news that these implementations 
are integrated into [Julia](https://julialang.org/), which is therefore capable 
to exhibit performances in terms of computational time and quality of the solutions
that are very close to those given by C implementations.

## Table of contents

1. [Basics and linear algebra](./julia1-basics.md);
2. [Solving linear systems](./julia2-linear-systems.md);
3. [Solving linear programs](./julia3-linear-programs.md);
4. More is coming soon...

## Links

* [Back to PA course](../README.md)

------------------------------

