
# A GPU programming overview

In order to give an initial overview on GPU programming, we are
going to study the implementation of some simple operations on vectors. 
More complex GPU programs will be presented in the next lectures, when 
a more detailed explanation of the topics simply mentioned in the current
lecture will be given. Please make sure that the basic content of the 
[previous lecture](./intro_ssp.md) is clear to you before continuing
your reading.

## The sum of two vectors

Our first program performs the sum of two vectors having the same length. 
The values held by the two vectors are single-precision integer numbers. 
If ```x``` and ```y``` are the two arrays of ```int``` types, representing 
the two vectors to sum up, and ```z```  is another array of the same type 
(with pre-allocated memory corresponding to the size of the other two 
arrays), then in sequential we can simply write:

	for (i = 0; i < n; i++)  z[i] = x[i] + y[i];

## GPU implementation with contiguous chunks

For an implementation on GPU where we can take the advantage of having
several computing resources in parallel, we can consider to assign a "chunk"
of contiguous vector elements to every running thread. Since the sum of each 
vector chunk is independent of the sums of other chunks, this parallelization 
strategy can be highly effective by allowing concurrent computation across 
multiple threads.

Once the size of the chunk (stored in the variable ```nchunk``` in the 
snippet below) has been defined, and the memory on the GPU has been 
correctly allocated and the content of ```x``` and ```y``` have been
moved to the GPU, the following kernel can be launched:

	__global__ void vectorsum_noncoalesced(size_t nchunk,int *xgpu,int *ygpu,int *zgpu)
	{
	   size_t i;
	   unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   for (i = id*nchunk; i < (id+1)*nchunk; i++)  zgpu[i] = xgpu[i] + ygpu[i];
	};

The label ```_noncoalesced``` in the function name indicates that we
haven't (yet) really exploited the power of our GPU device. In spite of
this, if you run the program [vectorsum.cu](./vectorsum.cu) on GPU, you
will see that the execution is already much faster than a simple
sequential execution on CPU!

## GPU implementation with coalesced memory access

In order to understand the meaning of "coalesced memory access", it is 
necessary to point out that the running threads in every GPU are not
only organized in blocks, but actually every thread belongs first of 
all to a smaller group of threads named **warp**, and that a **block**
can then be seen as a "group of warps".

Why is this important to understand the idea behind coalesced memory
access? Well, when a thread needs to get access to a specific piece of data
stocked in the GPU global memory, it is not the thread itself that makes
the job of retrieving the data (differently from a CPU core, which is
instead able to do that job by itself). It's actually the warp that takes 
care of the necessary data exchanges between the thread and the global memory.
But a warp does not have only one thread to take care about. The warp
ensures the data transfer for all its threads at once. 

At the first sight, the code below might seem to be jumping from given memory 
locations to other distant locations. But actually, for every group of threads 
belonging to the same warp, this code implies that every warp performs a 
**contiguous** memory access! Can you see that?

	__global__ void vectorsum_coalesced(size_t n,int *xgpu,int *ygpu,int *zgpu)
	{
	   size_t i;
	   unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   unsigned int nthreads = gridDim.x*blockDim.x;
	   for (i = id; i < n; i = i + nthreads)  zgpu[i] = xgpu[i] + ygpu[i];
	};

In fact, if you run the code again, you will see that this kernel is faster 
than the previous one, even if the number of operations it performs has not
changed. It's the optimized access to the global memory that allows it to get 
faster!

The next [lecture](./matrix-by-matrix.md) explains in much more details 
the concept of coalesced memory access. When possible, the use of shared 
memory, treated in [this lecture](./shared-matrix.md), is also recommended 
for speeding up the computations on GPU. Basically, our programs can really 
attain a great performance if we pay particular attention on how the data
are transferred to the running threads during the execution of the kernel.

## Links

* [Next: matrix-by-matrix in CUDA](./matrix-by-matrix.md)
* [Back to HPC lectures](./README.md)
* [Back to main repository page](../README.md)

