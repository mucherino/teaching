
# Low-level programming

In this section, we are going to explore a few possible
ways for improving our programs when working at low level.
The C language will be the programming language at the highest
level that we're going to use in these lessons. 

In some of these lessons, we will use the 
[```clang```](https://clang.llvm.org) compiler to generate
[llvm](https://llvm.org) and assembly codes. The main idea is 
to study these low-level codes and see whether they can be 
"optimized" for the small examples that we'll be taking into 
consideration.

## Changing the operation order in the matrix-by-matrix algorithm

We take the basic algorithm for the multiplication of two matrices:

	void mbm_simple(size_t n,double **A,double **B,double **C)
	{
	   assert(n > 0UL);
	   for (size_t i = 0; i < n; i++)
	   {
	      for (size_t j = 0; j < n; j++)
	      {  
	         for (size_t k = 0; k < n; k++)
	         {  
	            C[i][j] = C[i][j] + A[i][k]*B[k][j];
	         };
	      };
	   };
	};

and we ask ourselves whether we can improve the performances of this 
C implementation by "simply" changing the *order* of the required 
operations. [This lesson](./matrix-by-matrix.md) explores the 
possibility to optimize the access to the cache memories contained 
in our modern CPUs, and to have a reduction in computational time 
up to 25% while performing exactly the same operations.

## My own absolute value function

This absolute value function is a little peculiar, because it 
doesn't return the opposite of the input integer number, but 
rather 0, when this integer is negative. The same integer 
number is returned otherwise:

	int myabs(int a)
	{
	   if (a > 0)  return a;
	   return 0;
	};

The [llvm](https://llvm.org) and assembly codes can be found 
in the page devoted to [this lesson](./myabs.md). We will also 
verify whether some changes and improvements can be performed
on these small codes.

## Multiplying several 4-bit integers by 15

Imagine we need to multiply several positive integer numbers,
each consisting of 4 bits maximum, by a constant value. Let 
us suppose that this constant value happens to be the number 15.
If our positive integer numbers are stored in ```array```, then
the most obvious code that every programmer would write will
look like:

	for (size_t i = 0; i < n; i++)  array[i] = 15*array[i];

In [this lesson](./xfifteen.md), we are going to verify whether
we can devise an optimized code in assembly for performing this 
operation.

## Links

* [Back to main repository page](../README.md)

