
# Getting started with Julia

## Simple manipulations of vectors and matrices

Consider we have a squared matrix $A$ having $n$ rows (as well as $n$
columns), and a $n$-vector $b$. Suppose we are working on a program that,
at a certain point during its execution, needs to perform the multiplication
of $A$ by $b$.

In low level programming languages, there exist software libraries that
implement this kind of operations. In C programming language, for example,
the [BLAS](http://www.netlib.org/blas/) library provides functions for 
performing this kind of operations on matrices, it is heavily tested and 
gives very good performances. However, the programmer that wishes to use
at least one of these functions is supposed to go and read the corresponding
documentation, add the library to the software project, learn how to call the 
specific functions, and adapt the data structures to the arguments of the call. 
When [BLAS](http://www.netlib.org/blas/) functions (or the functions of any 
other software library) are necessary several times during the execution of 
our new program, then the programmer may feel the need to invest in these tasks. 
When instead the matrix-by-vector operation is required only once, and it
is going to be performed on a relatively small matrix $A$ and vector $b$,
then it is much more likely that the programmer is going to actually opt 
for the following C code:

	for (i = 0; i < n; i++)
	{
	   c[i] = 0.0;
	   for (j = 0; j < n; j++)
	   {
	      c[i] = c[i] + A[i][j]*b[j];
	   };
	};

This code is correct and would work just fine for small matrices and vectors.
In other situations, however, this short code would be much slower to perform
the operation than the implementations provided in the BLAS library. In fact,
the short code above does not control how the matrix and vector elements are 
accessed during the calculations (if you are wondering what this means,
please jump to [this lecture](../lowlevel/matrix-by-matrix.md).

Hey, but here's the first good news about Julia programming language: Julia
*contains* these BLAS implementations! So if the programmer opted to write 
the code in Julia, then performing operations such as a product between a matrix 
$A$ and a vector $b$ is as simple as writing:

	c = A*b;

and the implementation that is run in the background is BLAS optimized 
implementation!

OK, you may object that this is actually the syntax for multiplying two numbers.
How does Julia know that $A$ is a matrix, and that $b$ is a vector? This is shown 
a few lines below: the reason why only the line of Julia code (```c = A*b```) appear 
at first, is that this line is the equivalent of the C code reported above. We didn't 
specify either how to load the data in C in our little code snippet.

So, in Julia, the elements of a matrix can be loaded as follows:

	A = [1 2; 2 1];

The elements belonging to the same row need to be separated by a blank character, 
while a semi-colon indicates that we are about to step from one row to the following. 
The ending semi-colon indicates to Julia that we do not want the result of the performed 
operation to appear on the screen. In this easy case, the operation consists in the 
simple loading of the matrix elements in memory. Therefore, without the use of the 
ending semi-colon, Julia would have answered to the command with:

	2×2 Matrix{Int64}:
	 2  1
 	 1  2

Notice that ```Int64``` is the equivalent of a ```long``` in C. For the vector $b$, 
the syntax is similar, but the elements need to be separated by a comma, in order for 
Julia to recognize that the data structure is a vector (and not a matrix):

	b = [3,3]

The answer from Julia for this command is:

	2-element Vector{Int64}:
	 3
	 3

Finally, if we execute the line ```c = A*b``` at this point, we obtain the following
answer:

	2-element Vector{Int64}:
	 9
	 9

which is itself a vector of ```Int64``` types.

We can remark that this Julia syntax and basic functionalities are very similar 
(they are likely to have influenced Julia's developers actually) to those you can find
in [Matlab](https://www.mathworks.com/products/matlab.html),
[Scilab](https://www.scilab.org/), and other software tools. 
The fact that the BLAS library, together with other low-level software libraries, are 
part of Julia, implies that the performances of Julia on the implementation operations
are comparable to those of low-level programming languages, such as C.

## Links

* [Next: Structures](./structs.md)
* [Summary](./README.md)

