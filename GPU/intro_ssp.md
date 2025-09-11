
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

This recursive algorithm is able to count the number of solution of a given
SSP instance. Among the function arguments, ```n``` is the size of the instance,
```target``` is the SSP target, ```partial``` contains the partial sums obtained
with the integers selected in previous recursive calls, and ```set``` is the array
containing all the integers numbers in the original set. Finally, ```i``` is the
index indicating to which integer in ```set``` we are referring at the current
recursive call, and ```count``` will at the end contain the result. Notice that
the pointer to ```count``` is given in argument, because all recursive call will
have to access to the same memory location.

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

Our approach for this CUDA implementation is particularly tailored to the 
GPU structure. Instead of running our program on a single core of a CPU, 
we are going to have, on our GPU, several threads running simultaneously 
and in cooperation for solving a given SSP instance. Since we can have
several threads running in parallel when using modern GPUs, we can consider
to assign to each thread *only one subset sum to compute*.

The function that is supposed to run on the GPU is named **kernel**, and 
this is how it looks like for our specific problem:

	__global__ void ssp_on_gpu(size_t n,unsigned long *set_gpu,bool *is_solution_gpu,unsigned long target)
	{
	   unsigned long sum = 0UL;
	   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   size_t copy = id;
	   for (size_t i = 0; i < n; i++)
	   {
	      sum = sum + (id%2)*set_gpu[i];
	      id = id >> 1;
	   };
	   is_solution_gpu[copy] = (sum == target);
	};

You can notice that there are no recursive calls in this CUDA implementation.

If you're new with CUDA programming, there are several details you 
need to know to fully understand this code. If you're one of my 
students, please consider to participate to the lectures. Here below, 
only some of the *main points* will be briefly mentioned.

Code snippets are given below but you can refer to the full program in 
the file [ssp.cu](./ssp.cu). Notice that this kernel code is executed by 
all threads involved in the computation, where each of them makes reference 
to a different subset. Each thread is supposed to verify whether the 
subset that is *assigned to the thread* is, or not, a solution to the SSP 
(i.e., it verifies if the sum of its elements actually corresponds to the 
given target). This GPU approach requires an extra computing step that is
finally executed on the CPU.

### Thread identifiers

The way identifiers are assigned to the various threads in a GPU
is very peculiar. Please make reference to this [CUDA code](./identifiers.cu)
in order to have a clearer understanding. Basically, for architectural
reasons, threads are grouped in blocks: a predefined number of threads
can be found in every block, and a predefined number of blocks can be
found in every GPU. Things are a little more complex than that, actually,
but for the moment, this is all we need to know. Consider, however, that 
the number of *physical* blocks, as well as the number of available threads 
per block, may differ from our specific needs.

In our kernel, each thread needs to have a *unique* identifier, because 
its identifier also corresponds to the binary vector $x$ representing 
the solution (the subset) that the thread is supposed to verify ...

### GPU global memory

The arrays that are supposed to be allocated on the global memory of 
the GPU are marked with the label ```_gpu```. We say that the memory 
is *global* because all GPU threads are able to read from, and write 
on it. Differently from the CPU, however, which is able to exchange 
information directly with the RAM, it is necessary here to manually 
move the data from and to the global memory before and/or after 
performing computations on the GPU.

For our example, it is therefore necessary to move the original set 
of integers to the global memory. Before transferring the data, it is
necessary to allocate the memory on the GPU global memory:

	cudaMalloc((void**)&set_gpu,n*sizeof(unsigned long));

Once the memory allocated, we can transfer the content of the array 
```set```, previously defined on the RAM, to the array ```set_gpu```,
just allocated on the global memory:

	cudaMemcpy(set_gpu,set,n*sizeof(unsigned long),cudaMemcpyHostToDevice);

The second array, named ```is_solution```, is going to hold the information 
on whether the subset assigned to a given thread is an SSP solution or not. 
It is an array of booleans:

	cudaMalloc((void**)&is_solution_gpu,total_threads*sizeof(bool));

This information is the result of the computations that are going to be 
executed on the GPU. After the computations, the content of this array can 
be retrieved from the global memory and stored in the RAM:

	cudaMemcpy(is_solution,is_solution_gpu,total_threads*sizeof(bool),cudaMemcpyDeviceToHost);

Both ```set``` and ```is_solution``` are "traditional" arrays allocated on the 
RAM through a call to ```calloc```.

### Launching the kernel

From the main C function (running on CPU), it's the following line of code 
that launches the kernel:

	ssp_on_gpu<<<nblocks,nthreads>>>(n,set_gpu,is_solution_gpu,target);

Apart from the standard arguments for the C function, we can notice other 
additional arguments related to the particular setup we wish to have for our 
GPU. The argument ```nblocks``` indicates how many blocks we intend to allocate 
on the GPU for running our kernel. The argument ```nthreads``` indicates 
moreover the number of threads to allocate in each block. 

Before transferring the results to the RAM, it is fundamental to verify that 
every thread has actually finished its computations:

	cudaDeviceSynchronize();

## Performance of GPU version

If you run this code on GPU, you are probably not going to observe an 
extremely high improvement in the computational speed. In fact, in order 
to keep things as easy as possible at the moment, we made the choice to 
assign *only one subset* to each involved thread in our little CUDA code.

Can you think of a more complex implementation where every thread is
actually in change to work on a chunk of possible subsets of the SSP?
Recall that the subsets are represented in our code through the binary
representations of the thread identifiers: this binary representation 
may actually be used to define the *family* of subsets that are assigned
to a given thread. Naturally, at this point, additional bits will be 
necessary to enumarate the full SSP solution set.

In case you're discovering CUDA programming with this lecture, then it
is probably not a good idea to work on this extension of the code right away. 
You're rather invited to make the other exercises available on this repository,
and to come back to this code extension only when you'll feel more familiar 
with GPU programming and CUDA.

## Links

* [Next: GPU programming overview](./vectorsum.md)
* [Back to GPU lectures](./README.md)
* [Back to main repository page](../README.md)

