
# Low-level programming

In this section, we are going to explore a few possible
ways for improving our codes when programming at low level.
The C language will be the programming language at the highest
level that we're going to use in these lessons. Starting from
C, we will first of all generate (through compilation, the 
[```clang```](https://clang.llvm.org) compiler can be employed 
for this purpose) some codes in [llvm](https://llvm.org) and 
assembly. The main idea is to study these low-level codes and 
see whether they can be "optimized" for the small examples 
that we'll be taking into consideration.

## My own absolute value function

This absolute value function is a little peculiar, because
it doesn't return the opposite of the input real number,
but rather 0 when the real is negative. The same real number
is returned otherwise:

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
The most obvious code that every programmer may write at 
first would look like:

	for (size_t i = 0; i < n; i++)  array[i] = 15*array[i];

In [this lesson](./xfifteen.md), we are going to verify whether
we can devise an optimized code in assembly for this operation.

## Links

* [Back to main repository page](../README.md)

