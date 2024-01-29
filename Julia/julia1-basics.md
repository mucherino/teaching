
# Easy manipulations of vectors and matrices in Julia

Consider we have a squared matrix $A$ having $n$ rows (as well as $n$
columns), and a $n$-vector $b$. Suppose we are working on a program that,
at a certain point during its execution, needs to perform the product
between $A$ and $b$.

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
When instead a matrix-by-vector operation is required only once, then it is
likely the programmer would actually opt for the following code:

	for (i = 0; i < n; i++)
	{
		c[i] = 0.0;
		for (j = 0; j < n; j++)
		{
			c[i] = c[i] + A[i][j]*b[j];
		};
	};

This code is correct and would work fine for small matrices. But for 
situations where the matrix $A$ (and hence the vector $b$) are much larger, 
this code is not optimized, because it does not control how the matrix and 
vector elements are accessed during the calculations. Since, in general, 
the order in which the sums and products are performed is not important, 
it is possible to identify specific orders in which to have access to 
the matrix and vector elements so that the number of times the computer 
needs to communicate with its main memory is reduced. In this way, the 
total execution time (don't forget that retrieving data from the memory 
is much slower than performing computations) is reduced; in other words, 
the computational time is optimized. The [BLAS](http://www.netlib.org/blas/) 
function for the multiplication of a matrix by a vector attempts indeed 
to perform such an optimization.

Hey, but here's the good news: [Julia](https://julialang.org/) contains 
BLAS implementations! So if our programmer uses [Julia](https://julialang.org/) 
to perform operations such as a product between a matrix $A$ and a vector $b$, 
we can take advantage of the benefits mentioned above, together with others. 
But how to perform the operation in [Julia](https://julialang.org/)? Here's 
the code:

	c = A*b;

OK, you may object that this is the syntax for multiplying two real numbers.
How does Julia know that $A$ is a matrix, and that $b$ is a vector? This
is shown a few lines below: the reason why only this line of 
[Julia](https://julialang.org/) code appears at first, is that this line 
is equivalent to the C code reported above. We didn't specify either how 
to load the data in C in our little code snippet.

In [Julia](https://julialang.org/), the elements of a matrix can be 
loaded as follows:

	A = [1 2; 2 1];

The elements belonging to the same row need to be separated by a blank
character, while a semi-colon indicates that we are about to step from
one row to the following. The ending semi-colon indicates to 
[Julia](https://julialang.org/) that we do not want the result of the 
performed operation to appear on the screen. In this easy case, the 
operation consists in the simple loading of the matrix elements in memory. 
Therefore, without the use of the ending semi-colon, 
[Julia](https://julialang.org/) would have answered to the command with:

	2×2 Matrix{Int64}:
	 2  1
 	 1  2

Notice that ```Int64``` is the equivalent of a ```long``` in C. For the
vector $b$, the syntax is similar, but the elements need to be separated 
by a comma, in order for [Julia](https://julialang.org/) to recognize that 
the data structure is a vector (and not a matrix):

	b = [3,3]

The answer from [Julia](https://julialang.org/) for this command is:

	2-element Vector{Int64}:
	 3
	 3

Finally, if we execute the line ```c = A*b``` at this point, we obtain the following
answer from [Julia](https://julialang.org/):

	2-element Vector{Int64}:
	 9
	 9

which is itself a vector of ```Int64``` types.

## Links

* [Next: linear systems](./julia2-linear-systems.md)
* [Summary](./README.md)

