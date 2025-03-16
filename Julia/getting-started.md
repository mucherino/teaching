
# Getting started with Julia

We are going to introduce some simple [Julia](https://julialang.org/) syntax
through an example concerning some basic manipulations of vectors and matrices.

## Simple manipulations of vectors and matrices

Consider we have a squared matrix ```A``` having ```n``` rows (as well as 
```n``` columns), and a vector ```b```. Suppose we are working on a program 
that, at a certain point during its execution, needs to perform the multiplication
of ```A``` by ```b```.

In low-level programming languages, there exist software libraries for 
this kind of matrix manipulations. In C programming language, for example,
the [BLAS](http://www.netlib.org/blas/) library collects functions for 
performing this kind of operations on matrices, it is heavily tested and 
provides very good performances. However, the programmer who wishes to use at 
least one of the library functions is supposed to read the corresponding
documentation, link the library to the software project, learn how to call the
specific functions, and adapt the data structures to the arguments of the call. 
When [BLAS](http://www.netlib.org/blas/) functions (or the functions of any 
other software library) are necessary several times during the execution of our 
new program, then the programmer may feel the need to invest in the aforementioned
tasks. Instead, when the functionality is required only a few times, maybe
even only one time on a small example, then it is more likely that the programmer 
opts for writing its own code. In our example, it is a matrix-by-vector operation 
that we need to implement. If the involved matrix $A$ and vector $b$ are indeed
relatively small, the following C code would do the required job, just fine:

	for (i = 0; i < n; i++)
	{
	   c[i] = 0.0;
	   for (j = 0; j < n; j++)
	   {
	      c[i] = c[i] + A[i][j]*b[j];
	   };
	};

This code is correct, but in other situations, where the matrix (and vector)
are larger, and when it needs to be executed several times, then it would run
much slower than the implementation provided in the BLAS library. In fact,
the C code written by our programmer is not optimized (it does not control, 
for example, how the matrix and vector elements are accessed during the 
calculations: if you are wondering what this means, please jump to 
[this lesson](../lowlevel/matrix-by-matrix.md)).

Hey, but here's the first good news about Julia programming language: Julia
*contains* these BLAS implementations! So if our programmer were writing the 
code in Julia (instead of C), then performing operations such as the product 
between a matrix $A$ and a vector $b$ would be as simple as writing:

	c = A*b;

and the function that is executed in the background is actually the optimized 
implementation contained in the BLAS library.

OK, but now you might object that this is actually the syntax for multiplying two 
numbers. How does Julia know that ```A``` is a matrix, and that ```b``` is a vector? 
This is shown  a few lines below: one reason for beginning by showing only this single
line of Julia code (```c = A*b```) is that we are, at the moment, focusing only on how 
to *perform* the multiplication, and not (yet) on how to load the data. This line of 
code in Julia (```c = A*b```) is exactly the equivalent of the C code reported above.

But there is also another (quite important) point related to Julia programming.
The one-liner ```c = A*b``` is interpreted by Julia as a matrix-by-vector
multiplication because of the types associated to the two variables ```A``` 
and ```b```. If ```A``` and ```b``` were both of type ```Int64``` (the standard
64-bit signed integers, the equivalent of ```long``` in C) for example, then 
the operation that Julia would execute is the simple multiplications of two 
integer numbers. We will devote an [entire lesson](./multiple-dispatch.md)
to the way Julia selects the *methods* to execute on the basis of the input
arguments. Also notice that the type of ```c``` is consequently inferred.

So, how to prepare our two arguments ```A``` and ```b``` in order to actually
perform the matrix-by-vector multiplication? In Julia, the elements of a matrix 
can be loaded as follows:

	A = [1 2 3; 4 5 6; 7 8 9];

The elements belonging to the same row need to be separated by a blank character, 
while a semi-colon indicates that we are about to step from one row to the following. 
The ending semi-colon indicates to Julia that we *do not* want the result of the 
performed operation (in this case, the operation simply consists in loading the
elements of the matrix) to appear on the screen. Without the use of the ending 
semi-colon, Julia would have answered to our command with:

	3×3 Matrix{Int64}:
	 1  2  3
	 4  5  6
	 7  8  9

```Matrix{Int64}``` is the inferred type for ```A```. It is a *generic* type: it
is a ```Matrix``` whose elements, in our particular case, are of type ```Int64```. 

For the vector, the syntax is similar, but the elements need to be separated by 
a comma, for Julia to recognize that the desired data type is ```Vector``` (and 
not ```Matrix```):

	b = [1,1,1]

The answer from Julia to this command is:

	3-element Vector{Int64}:
	 1
	 1
	 1

Finally, if we execute the line ```c = A*b```, we get the following answer:

	3-element Vector{Int64}:
	  6
	 15
	 24

which has type ```Vector{Int64}```.

We can remark that this Julia syntax and basic functionalities are very similar 
(they are likely to have influenced Julia's developers, actually) to those of
[Matlab](https://www.mathworks.com/products/matlab.html),
[Scilab](https://www.scilab.org/), as well as other software tools. 
The fact that the BLAS library, together with other low-level software libraries, are 
part of Julia, implies that the performances of Julia on the implementation operations
are comparable to those of low-level programming languages, such as C.

To conclude this part, we point out that other (more complex) operations involving 
vectors and matrices can be performed in Julia (with a quite simple syntax!), such as 
the solution of linear systems, as well as the solution of some classes of optimization
problems.

## Links

* [Next: Structures and functions](./structs-and-funs.md)
* [Summary](./README.md)

