
# A GPU programming overview

In order to give an initial overview on GPU programming, we are
going to study the implementation of some simple operations on vectors. 
More complex GPU programs will be presented in the next lectures, when 
a more detailed explanation of the topics only mentioned in the current
lecture will be given. Please make sure that the basic content of the 
[previous lecture](./intro_ssp.md) is clear to you before continuing
your reading.

The [CUDA code](./vectorsum.cu) is available for download. Below, we
will make reference to some code snippets from this CUDA code, and 
we will show the (partial) execution of the code on 
[Google Colab](https://colab.research.google.com/). As for January 2025,
Colab gives limited access to Nvidia T4 GPU cards.

## The sum of two vectors

Our very first task is to compute the sum of two vectors. The two vectors
are supposed to have the same length, and to hold single-precision integer 
numbers. If ```x``` and ```y``` are the two arrays of ```int``` types, 
representing the two vectors to sum up, and ```z```  is another array of 
the same type and length, then in sequential we can simply write:

	for (i = 0; i < n; i++)  z[i] = x[i] + y[i];


On Google Colab, when the length of the vector is about 3 hundred millions
(327,680,000), one CPU core can compute this vector sum in about 1 second
and a half:

	Problem1: computing the sum z = x + y
	Computations on CPU ...  done: elapsed time =  1.5568; verification: 9 + 4 = 13

## Vector sums with contiguous chunks

For an implementation on GPU where we can take the advantage of having
several computing resources in parallel, we can consider to assign a "chunk"
of contiguous vector elements to every running thread. Since the sum of each 
vector chunk is independent of the sums of other chunks, this parallelization 
strategy can be highly effective by allowing concurrent computation across 
multiple threads.

Once the size of the chunk (stored in the variable ```nchunk``` in the CUDA 
code) has been defined, the memory on the GPU has been correctly allocated and 
the content of ```x``` and ```y``` have been moved to the GPU, the following 
kernel can be launched:

	// computing the sum of two vectors on GPU
	// this kernel is based on a global memory access which is not coalesced
	__global__ void vectorsum_noncoalesced(size_t nchunk,int *xgpu,int *ygpu,int *zgpu)
	{
	   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   for (size_t i = id*nchunk; i < (id+1)*nchunk; i++)  zgpu[i] = xgpu[i] + ygpu[i];
	};

The label ```_noncoalesced``` in the function name indicates that we haven't (yet) 
really exploited the power of our GPU device. In fact, when running the code with
512 blocks, each allocating 128 threads, and with each thread working on a vector 
chunk having size 5000, we get the following result:

	Vector sums with CUDA
	nblocks = 512, nthreads = 128, nchunk = 5000
	total number of threads is 65536
	hence vector size is 327680000
	Problem1: computing the sum z = x + y
	Computations on CPU ...  done: elapsed time =  1.5568; verification: 9 + 4 = 13 
	Computations on GPU (non-coalesced) ...  done: elapsed time =  0.7136; verification: 1 + 7 = 8

We have (only) doubled the speed.

## Vector sums with coalesced memory access

In order to understand the idea behind **coalesced memory access**, it is necessary 
to point out that the running threads in every GPU are not only organized in blocks, 
but actually every thread belongs first of all to a smaller group of threads named 
*warp*, and that a *block* can then be seen as a "group of warps".

What relationship with memory access? Well, when a thread needs to get access to 
a specific piece of data stocked in the GPU global memory, it is not the thread 
itself that makes the job of retrieving the data (differently from a CPU core, 
which is instead able to do that job by itself). It's actually the warp that takes 
care of the necessary data exchanges between the thread and the global memory.
But a warp does not have only one thread to take care about. The warp ensures the 
data transfer for all its threads at once. 

At the first sight, the code below might seem to have threads jumping from a given 
memory location to other distant locations. But actually, it allows the warps to
perform **contiguous** memory accesses! Can you see that?

	// computing the sum of two vectors on GPU
	// this kernel is based on a global memory access which is coalesced
	__global__ void vectorsum_coalesced(size_t n,int *xgpu,int *ygpu,int *zgpu)
	{
	   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   size_t nthreads = gridDim.x*blockDim.x;
	   for (size_t i = id; i < n; i = i + nthreads)  zgpu[i] = xgpu[i] + ygpu[i];
	};

In fact, if you run the code again, you will see that this kernel is faster than the 
previous one, even if the number of operations it performs has not changed at all. 
It's the optimized access to the global memory that allows it to get faster:

	Problem1: computing the sum z = x + y
	Computations on CPU ...  done: elapsed time =  1.5568; verification: 9 + 4 = 13 
	Computations on GPU (non-coalesced) ...  done: elapsed time =  0.7136; verification: 1 + 7 = 8
	Computations on GPU (coalesced) ...  done: elapsed time =  0.0206; verification: 2 + 1 = 3

## Conditional vector sums

Dealing with conditions is a little tricky on GPU devices. We have already
mentioned to the fact that threads are organized in *warps*, which are then
organized in *blocks*. We pointed out in the previous section that warps are
in charge to retrieve the data from the global memory for their threads.
Another important task performed by warps is that of **controling** all 
their threads. In other words, each isolated thread is only capable to perform 
computations (logic and arithmetic operations), but not to interpret the
machine code and hence to select which operations to perform. This task is
rather given to the warp, which, again, ensures that the job is done for all 
of its threads at once. As a consequence, **all threads belonging to the same 
warp perform exactly the same operation at each given instant of time**.
They are perfectly synchronized.

But how to deal then with situations where the code contain branching conditions?
Depending on the data on which each thread is working, a different code
branch may be selected. How the warp can deal with situations? It is not able
to assign different operations to its threads. At most, it can communicate 
to some of its threads to *do nothing*.

This kernel is example of code divergence caused by conditional branching:

	// computing the conditional sum of two vectors on GPU
	// !! this kernel suffers of divergence issues !!
	// in order to introduce the divergence, we suppose that:
	// - threads with id%4 == 0 simply set up all z components to 0
	// - threads with id%4 == 1 copy the content of the x component
	// - threads with id%4 == 2 copy the content of the y component
	// - threads with id%4 == 3 perform the sum of the two components of x and y
	__global__ void vectorsum_divergent(size_t n,int *xgpu,int *ygpu,int *zgpu)
	{
	   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   size_t nthreads = gridDim.x*blockDim.x;
	   size_t k = id%4;
	   for (size_t i = id; i < n; i = i + nthreads)
	   {
	      if (k == 0)
	         zgpu[i] = 0;
	      else if (k == 1)
	         zgpu[i] = xgpu[i];
	      else if (k == 2)
	         zgpu[i] = ygpu[i];
	      else
	         zgpu[i] = xgpu[i] + ygpu[i];
	   };
	};

This kernel is slower than the previous one. One may think that the increased 
computational time is simply due to the conditions that are included in this new
kernel. But actually the kernel given below, which underwent refactoring so that 
it does not suffer anymore of code divergence, performs exactly like a kernel 
without conditions:

	// computing the sum of two vectors on GPU
	// this kernel performs the same operations as above
	// **but without causing code divergence**
	__global__ void vectorsum_without_divergence(size_t n,int *xgpu,int *ygpu,int *zgpu)
	{
	   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   size_t nthreads = gridDim.x*blockDim.x;
	   size_t alphax = id & 1UL;
	   size_t alphay = (id >> 1) & 1UL;
	   for (size_t i = id; i < n; i = i + nthreads)
	   {
	      zgpu[i] = alphax*xgpu[i] + alphay*ygpu[i];
	   };
	};

This is the execution on Google Colab:

	Problem2: computing the conditional sum z = {either x, or x, or y, or x + y}
	Computations on GPU (divergent) ...  done: elapsed time =  0.0287; verification: x = 1, y = 4, z = 0 (index modulo is 0)
	Computations on GPU (non-divergent) ...  done: elapsed time =  0.0202; verification: x = 6, y = 9, z = 6 (index modulo is 1)

Notice that the refactored kernel replaces conditions with logic and arithmetic 
operations.

## Shared memory

At block level, all warps (and hence all its threads) share a common local
memory called the *shared memory*. The access to this memory is much faster
than the access to the global memory. Using and exploiting in full what the
shared memory offers to CUDA programmers is not trivial; in this preliminary
overview, we will simply take advantage of the fact that its access by the
warps is fast.

To this purpose, we consider the problem of summing up all the elements in a
given vector. The kernel below does not make use of the shared memory:

	// computing the sum of all vector elements
	// in this kernel, each thread computes its partial sum, which is then 'sent' 
	// to the CPU for the final computation
	__global__ void vectorsum_partial_reduction(size_t n,int *xgpu,int *pgpu)
	{
	   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   size_t nthreads = gridDim.x*blockDim.x;
	   int thread_partial = 0;
	   for (size_t i = id; i < n; i = i + nthreads)  thread_partial = thread_partial + xgpu[i];
	   pgpu[id] = thread_partial;  // there are as many partial sums as the number of threads
	};

The computations are finalized on the GPU, after the transfer of the partial 
sums from the global memory to the RAM:

	int sum_gpu = 0;
	for (size_t i = 0; i < total_threads; i++)  sum_gpu = sum_gpu + partial[i];

This may seem a reasonable algorithmic solution, but if the number of threads is
very large, the computational time for performing the final sums on the CPU is
not negligible.

What about storing the partial sums computed by each thread in the corresponding
shared memory, and then let only one of threads in each block to compute the total 
sum *for its own block*?

	// computing the sum of all vector elements
	// this kernel uses shared memory to store partial sums calculated by each thread;
	// thread 0 of each block then reads these partial sums from shared memory and computes 
	// the final sum for the block
	__global__ void vectorsum_shared_reduction(size_t n,int *xgpu,int *pgpu)
	{
	   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   size_t nthreads = gridDim.x*blockDim.x;
	   extern __shared__ int shared_mem[];  // expected to have the size of a block
	   shared_mem[threadIdx.x] = 0;
	   for (size_t i = id; i < n; i = i + nthreads)  shared_mem[threadIdx.x] = shared_mem[threadIdx.x] + xgpu[i];
	   __syncthreads();
	   int block_partial = 0;
	   if (threadIdx.x == 0)
	   {
	      for (size_t i = 0; i < blockDim.x; i++)  block_partial = block_partial + shared_mem[i];
	      pgpu[blockIdx.x] = block_partial;  // only one partial sum per block goes back to the CPU!
	   };
	};

The final computations are still performed by the CPU, but less data need to be
transferred (only one partial sum per block), and less data need to be considered
in the final for loop executed by the CPU:

	Problem3: computing the sum of all z elements
	Computations on GPU (partial reduction) ...  done: elapsed time =  0.0073 (gpu)  0.0001 (transfer)  0.0002 (cpu); verification: OK
	Computations on GPU (reduction with shared memory) ...  done: elapsed time =  0.0072 (gpu)  0.0000 (transfer)  0.0000 (cpu); verification: OK

We remark that the computations of ```block_partial``` in each block does not have 
any impact on the total computational time on GPU. 

## To go further

The next lectures will explore in more details these main concepts about
CUDA programming:

- more about coalesced memory access: [Lecture 3](./matrix-by-matrix.md);
- more about shared memory: [Lecture 4](././shared-matrix.md).

## Links

* [Next: matrix-by-matrix in CUDA](./matrix-by-matrix.md)
* [Back to HPC lectures](./README.md)
* [Back to main repository page](../README.md)

