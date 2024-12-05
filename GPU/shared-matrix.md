
# Shared memory

One of the possible ways to enhance the performances of our CUDA programs
is to take advantage of a special memory that is contained in every 
*physical* block of threads. This is the so-called "shared memory". The
memory is therefore shared by all the threads that are contained in the
block. Recall, moreover, that blocks of GPU devices are formed by a
certain number of *warps*. Therefore, it is important to remark that
some groups of threads sharing this common memory are perfectly 
synchronized (because they have a common controler), but this 
synchronization does not apply to threads of the block that belong
to different warps.

The shared memory is designed in such a way to ensure a very fast access
to the data it holds. We are going to study an example where the use
of shared memory will give us a clear advantage in terms of performances.

## Several matrix-by-matrix multiplications

Given a real number $p$ and two matrices $A$ and $B$, this 
[CUDA code](./shared-matrix.cu) computes the following matrix:

$$
   C = (A \times B) + (A \times B) + \dots + (A \times B) ,
$$

where the resulting matrix $A \times B$ is multiplied by itself $p$ times.
Of course, this calculation can be trivially simplified:

$$
   C = p \cdot (A \times B) .
$$

However, we are not going to exploit this simplication, because we wish to
take into consideration the more complex situation where the two matrices
$A$ and $B$ can actually vary in each matrix multiplication. In the code,
$A$ and $B$ are randomly generated at the beginning, and used in every
multiplication. You can change the code and have two new matrices generated
for each new multiplication, but this will not have any impact on the
observed performances (we won't count, of course, the additional complexity 
for the generation of the new matrices).

The [CUDA code](./shared-matrix.cu) compares two kernels in charge of 
performing the operations on the matrices. The first kernel does not use 
the shared memory, while the second actually exploits it for enhancing
the performances. In this page, we are going to focus our attention only 
on the latter.

Notice that the code has a similar structure, and it satisfies assumptions 
(matrix size, assignment of matrix elements to threads) similar to those
in the code presented in the [previous lecture](./matrix-by-matrix.md). 

## Allocating shared memory at block level

Shared memory can only be allocated at run time (differently from what we
have learned about the global memory). There are two possible approaches
for performing this allocation. When the size of the memory is constant
for all executions, the static approach is recommended. We can for example
write:

	__shared__ float shared_mem[128];

However, the size of the memory cannot always be predicted in advance 
(in our code, we are in fact in this second situation). But unfortunately,
dynamical memory allocation as we know it cannot be applied to shared 
memory. The second approach for shared memory allocation requires that
the size of the memory (*in terms of bytes*) must be known at the 
moment of launching the kernel. This is the approach used in our code.

When we launch the kernel, we need therefore to specify the size of the 
shared memory (in each block):

	multi_shared<<<blocks,threads,nthreads*nthreads*sizeof(float)>>>(AG,BG,CG,p);

and then, when allocating the memory, we need to adapt the syntax in the 
kernel as follows:

	extern __shared__ float shared_mem[];

Our kernel continues then with the identification of memory location 
associated to each thread (pay attention to the fact that this memory is 
shared only by the threads in the same block, so we can use local thread
identifiers!), and the initialization to zero for each of these elements:

	size_t myelement = index(threadIdx.x,threadIdx.y,blockDim.x);
	shared_mem[myelement] = 0.0f;

Since only the threads belonging to the same warp are perfectly synchronized, 
it is necessary to enforce the synchronization of the entire block before 
continuing with the computations, in order to ensure data consistency:

	__syncthreads();

## Performing the computations

Once allocated, the shared memory can be accessed exactly like the global memory,
i.e. through the given pointer. In the line below, the kernel calls an internal
function for the multiplication of two matrices, ```AG``` and ```BG```, and
it cumulates the result in ```shared_mem```:

	for (size_t ip = 0; ip < p; ip++)  mbm_shared(AG,BG,shared_mem);

Finally, if the obtained result is in the shared memory, it is necessary to move 
it to the global memory before considering to transfer the data to the RAM. In our 
code, this is implemented in the following function:

	retrieve_matrix(shared_mem,CG);

## Twice faster

By partially exploiting the use of the shared memory (the matrices $A$ and $B$ 
are still directly accessed from the global memory), we are able to reduce by 
a half the execution time on GPU:

	p*(Matrix-by-Matrix) on GPU with CUDA
	Comparing versions : sequential, CUDA w/out shared memory, and CUDA with shared memory
	Two-dimensional thread grid structure: [blocks (32,32), threads (8,8)]
	Setting up memory space on RAM ... done
	Sequential version ... done in 5.28392 seconds
	Memory allocation and memory transfer ... done
	CUDA version without shared memory ... done in 0.022269 seconds
	CUDA version with shared memory ... done in 0.011299 seconds

## But we can do even better!

Have you ever heard of *banked* memories? Well, if you are following this lecture 
because it is part of your course program, then your teacher will explain this to 
you. Otherwise, there is a lot of material on the web!

Once the concept of banked memory is clear to you, you may imagine an upgraded
version of our CUDA code where sub-matrices (and not single elements) are assigned 
to each thread. But in this case, how to define these sub-matrices in order to 
ensure a quick access to the shared memory?

## Links

* [Next: CUDA streams](./sierpinski.md)
* [Back to HPC lectures](./README.md)
* [Back to main repository page](../README.md)

