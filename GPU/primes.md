
# Exercise: prime factorization on GPU

## Prime factorization

Prime factorization is one of the hardest known mathematical problems.
Finding a general and efficient solution for this problem would allow
people to break many of the cryptographic systems that are currently
being used to secure for example our online purchases by credit card, 
but not only.

Prime factorization is the process of expressing a positive integer
number as the product of its prime factors. A *prime factor* is a prime 
number that divides the given number evenly. For example, the prime 
factorization of 36 is $2 \cdot 2 \cdot 3 \cdot 3$.

## The naive algorithm

The naive algorithm for prime factorization of a given number $n$ is 
based on the idea to try **all** possible divisors of $n$, and to
keep those which are prime and divide $n$ evenly.

The [provided code](./primes.cu) implements this naive algorithm in 
sequential for finding all *divisors* of a given randomly generated 
positive integer number. Notice that, in order to make things easier
(but of course less efficient), the code does not verify whether the 
divisors are prime numbers or not. In case you'd like to improve your
code (which would be great, actually!), please pay attention to the 
fact that the same improvement should be implemented also on the GPU
version of your code, in order to perform thereafter fair performance
comparisons.

Please begin by reviewing the provided code, and then fill the empty
parts (marked with the comment ```TO BE COMPLETED```). This exercise
is conceived for not only to make you write a GPU kernel, but also to
deal with the memory allocation on the global memory, and with 
the memory transfers.

## Compiling and executing

In order to compile your CUDA code, you need to use the command:

	nvcc -o primes primes.cu

You can launch the execution in the "traditional" way:

	./primes

Is your execution on GPU faster than the one performed on the CPU?

## Coalesced memory access

Do you think the access to the global memory during the execution
of the kernel may become faster if the access is guaranteed to be *coalesced*?
It would probably be a good idea to make sure that your kernel enforces
a coalesced access to the data.

## Links

* [Back to GPU programming](./README.md)
* [Back to main repository page](../README.md)

