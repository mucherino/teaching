
# Julia course

This course is intended to be an introduction to the use of [Julia](https://julialang.org/)
programming language. 

The course contains several examples, where the main objective is to show how 
writing programs in [Julia](https://julialang.org/) can be much less tedious 
than programming at lower level. We begin by exploring the syntax for storing
vectors and matrices, and for performing some basic *linear algebra* operations 
with them. The solution of *linear systems* and *linear programs* are at the 
center of the examples given thereafter. 

We can remark that some of Julia functionalities discussed in this initial 
part of our course are very similar (they actually influenced 
[Julia](https://julialang.org/)'s syntax) to those provided by 
[Matlab](https://www.mathworks.com/products/matlab.html),
[Scilab](https://www.scilab.org/), and other software tools. In some occasions,
a syntax comparison with C programming language is given during the lectures. 
In C in fact, there are specific implementations for linear algebra, mainly the 
[BLAS](http://www.netlib.org/blas/) library, which provides some basic matrix
operations, and [LAPACK](http://www.netlib.org/lapack/) for more advanced operations,
such as the solution to linear systems. The good news is that these implementations 
are integrated into [Julia](https://julialang.org/), which is therefore capable 
to exhibit performances in terms of computational time and quality of the solutions
that are very close to those given by C implementations. We also remark that there
exist other software tools for modeling in linear programming, such as the 
[AMPL](https://ampl.com/) modeling language. One main advantage of Julia is
the capability of integrating various functionalities in one unique language,
which also supports other paradigms. 

## Table of contents

1. [Basics and linear algebra](./julia1-basics.md);
2. [Linear systems](./julia2-linear-systems.md);
3. [Linear programs](./julia3-linear-programs.md);
4. [Adaptive maps](./julia4-adaptive-maps.md);
5. [Integer linear programs](./julia5-integer-programs.md).

## What more?

There are several more things that one can consider to say about 
[Julia](https://julialang.org/) programming language:

1. encapsulation and abstraction in Julia's style;
2. memory management in Julia's style;
3. operator and method overloading via multiple dispatch (Julia is not object-oriented!);
4. compilation to llvm;
5. compilation to assembly code.

These are some of the topics covered in the next lessons.

## Links

* [Back to Advanced Programming course](../Advanced.md)
* [Back to main repository page](../README.md)

