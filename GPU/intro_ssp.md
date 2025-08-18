
# Introduction to GPU programming

We introduce GPU programming in CUDA (with the C programming
language) through an example. The combinatorial problem considered 
in this lecture is one of the easiest to understand but it belongs to 
the category of the hardest problems to solve.

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

There is a very large scientific literature on the SSP, and there 
exist several algorithms that, in spite of its complexity, are able 
to solve the SSP efficiently. These algorithms exploit some particular 
properties that SSP instances may satisfy, and they are generally able 
to provide *only one* solution to the problem, i.e., only one subset 
of $S$ whose elements sum up to the target $t$. Instead, our approach 
is going to be different: our interest is in counting the total number 
of solutions for a given SSP instance. In other words, we want to 
count how many subsets of the original $S$ will have elements whose sum 
corresponds to the given target $t$. For more information about the SSP, 
you can refer to this 
[wikipedia page](https://en.wikipedia.org/wiki/Subset_sum_problem).

## The basic recursive algorithm

How to explore the entire solution set for a given SSP? 
Take the elements of $S$ in the given order. Suppose that the
first element, $s_1$, is not included in a solution. Then, solve
another SSP instance where the original set of integers is replaced 
by $S \setminus (s_1)$. However, we need to consider 
also the possibility that $s_1$ is included in a solution. In
this case, take note of the current partial sum, and again, solve 
the SSP instance having $S \setminus (s_1)$ as set of 
integers. In general for an element $s_i$, we need to consider
both possibilities, and to proceed to the element $s_{i+1}$ once
the partial sum (stored in the variable ```partial``` in the code
below) has been updated. Every time a total sum is completed, 
the solution counter is updated if this total sum corresponds
to the target.

This is a simple implementation in C of the recursive algorithm:

	void recursive(size_t i,size_t n,unsigned long target,unsigned long partial,unsigned long *set,size_t *count)
	{
	   // are we at the last element yet?
	   if (i == n)
	   {
	      if (partial == target)  (*count)++;
	      return;
	   };

	   // we do not consider the current integer
	   recursive(i+1,n,target,partial,set,count);

	   // we do consider the current integer
	   partial = partial + set[i];
	   recursive(i+1,n,target,partial,set,count);
	};

## CUDA implementation

Our approach for this CUDA implementation is particularly tailored
to the GPU structure. Instead of running our program on a single 
core of a CPU, we are going to have, on our GPU, several threads
running simultaneously and in cooperation for solving a given SSP
instance. Since the number of parallel threads can importantly
grow when using more modern GPUs, we can consider to assign *each
sum to compute* to a single thread. 

The function that is supposed to run on the GPU is named **kernel**,
and this is how it looks like for our specific problem:

	__global__ void ssp_on_gpu(size_t n,unsigned long *set_gpu,unsigned long *sum_gpu)
	{
	   unsigned long sum = 0UL;
	   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   size_t copy = id;
	   for (size_t i = 0; i < n; i++)
	   {
	      sum = sum + (id%2)*set_gpu[i];
	      id = id >> 1;
	   };
	   sum_gpu[copy] = sum;
	};

You can notice that there are no recursive calls in this CUDA implementation.

If you're new with CUDA programming, there are several details
you need to know to fully understand this code. If you're one of
my students, please consider to participate to the lectures. Here
below, only the some *main points* will be briefly mentioned in
the following subsections.

Code snippets are given below but you can refer to the full program 
in the file [ssp.cu](./ssp.cu). Notice that the kernel function given 
above only computes the sums; it's the CPU that will be in charge of
verifying how many sums actually correspond to the given target $t$.

### Thread identifiers

The way identifiers are assigned to the various threads in a GPU
is very peculiar. Please make reference to this [CUDA code](./identifiers.cu)
in order to have a clearer understanding. Basically, for architectural
reasons, threads are grouped in blocks: a predefined number of threads
can be found in every block, and a predefined number of blocks can be
found in every GPU. Things are a little more complex than that, actually,
but for the moment, this is all we need to know. Consider, however, 
that the number of blocks and available threads may differ from our
needs when we want to run a specific program.

In our kernel, each thread needs to have a *unique* identifier,
because its identifier also corresponds to the binary vector $x$
representing the solution the thread is supposed to compute ...

### GPU global memory

The two arrays of type ```unsigned long``` in our kernel function
are marked with the label ```_gpu```, because they are actually
supposed to be allocated on the global memory of our GPU device.
It is said that this memory is *global* because all threads are
able to read and write on it. Differently from the CPU, however,
which is able to exchange information directly with the RAM, it
is necessary here to manually move data from and to the global
memory before and/or after performing computations on the GPU.

For our example, it is therefore necessary to move the original 
set of integers on the global memory. Before transferring the 
data, we need to allocate memory on the GPU:

	cudaMalloc((void**)&set_gpu,n*sizeof(unsigned long));
	cudaMalloc((void**)&sum_gpu,total_threads*sizeof(unsigned long));

Then, we can transfer the content of the array ```set```, 
allocated on the RAM, to the array ```set_gpu``` that we have
just allocated on the global memory:

	cudaMemcpy(set_gpu,set,n*sizeof(unsigned long),cudaMemcpyHostToDevice);

The second array, named ```sum_gpu```, that we have allocated on
the global memory, is going to be used to store all the computed
sums. When the computations are finally done on the GPU, it is 
necessary to transfer these sums from the global memory to the RAM:

	cudaMemcpy(sum,sum_gpu,total_threads*sizeof(unsigned long),cudaMemcpyDeviceToHost);

otherwise the CPU won't be able to have access to them.

### Launching the kernel

From the main C function (running on CPU), it's the following line 
of code that launches the kernel:

	ssp_on_gpu<<<nblocks,nthreads>>>(n,set_gpu,sum_gpu);

Apart from the standard arguments for the C function, there are 
other additional arguments related to the particular setup we wish 
to have for our GPU. The argument ```nblocks``` indicates how many 
blocks we intend to allocate on the GPU for running our kernel. 
The argument ```nthreads``` indicates moreover the number of threads 
to allocate in each block. 

Before transferring the results to the RAM, it is fundamental
to verify that every thread has actually finished its computations:

	cudaDeviceSynchronize();

## Links

* [Next: GPU programming overview](./vectorsum.md)
* [Back to HPC lectures](./README.md)
* [Back to main repository page](../README.md)

