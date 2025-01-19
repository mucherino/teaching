
# A GPU programming overview

In order to make an initial overview on GPU programming, we are
going to study the implementation of some simple operations on vectors. 
More complex GPU programs will be presented in the next lectures, when 
a more detailed explanation of the topics simply mentioned in the current
lecture will be treated. Please make sure that the basic content of the 
[previous lecture](./intro_ssp.md) is clear to you before continuing
your reading.

## The sum between two vectors

Our first program performs the sum between two vectors having the
same length. The values held by the two vectors are single-precision
integer numbers. If ```x``` and ```y``` are the two arrays of ```int``` 
types, representing the two vectors to sum up, and ```z```  is another 
array of the same type (with pre-allocated memory corresponding to 
the size of the other two arrays), then in sequential we can simply
write:

	for (i = 0; i < n; i++)  z[i] = x[i] + y[i];

## GPU implementation with contiguous chunks

For an implementation on GPU where we can take the advantage of having
several computing resources in parallel, we can consider to assign a "chunk"
of contiguous vector elements to every running thread. The sums required 
for computing every chunk of the resulting vector ```z``` are completely 
independent from the sums performed in other chunks, hence implying that
this parallelization strategy is highly effective. 

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

The label ```_noncoalesced``` in the function name indicate that we
haven't exploited at the maximum the power of our GPU device. In spite of
this, if you run the program [vectorsum.cu](./vectorsum.cu) on GPU, we
will see that the execution is already much faster than a simple 
sequential execution on CPU!

## GPU implementation with coalesced memory access

In order to understand the meaning of "coalesced memory access", it is 
necessary to point out that the running threads in every GPU are not
only organized in blocks, but actually every thread belongs first of 
all to a smaller group of threads named **warp**, and that a **block**
can then be seen as a group of warps.

Why is this important to understand the idea behind coalesced memory
access? When a thread needs to get access to a specific piece of data
stocked in the GPU global memory, it is not the thread itself that makes
the job of retrieving the data (differently from a CPU core, which is
instead able to do that job by itself). It's the warp that takes care
of the necessary data exchanges between the thread and the global memory.
But a warp does not have only one thread to take care about. The warp
ensures the data transfer for all its threads at once. 

At the first sight, the code below seem to be jumping from given memory 
locations to other distant locations. Actually, for every group of threads 
belong to the same warp, this code requires a *contiguous* memory access!

	__global__ void vectorsum_coalesced(size_t n,int *xgpu,int *ygpu,int *zgpu)
	{
	   size_t i;
	   unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
	   unsigned int nthreads = gridDim.x*blockDim.x;
	   for (i = id; i < n; i = i + nthreads)  zgpu[i] = xgpu[i] + ygpu[i];
	};

If you run the code again, we will see that this kernel is faster than the
previous one, even if the number of operations it performs is exactly
the same. Therefore, it's the access to the global memory that allows it
to get faster.

The next [lecture](./matrix-by-matrix.md) gives much more details about 
memory coalesced access.

## Links

* [Next: matrix-by-matrix in CUDA](./matrix-by-matrix.md)
* [Back to HPC lectures](./README.md)
* [Back to main repository page](../README.md)

