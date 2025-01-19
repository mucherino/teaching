
# Exercise: matrix transposition on GPU

In this exercise, we are going to consider a simple yet potentially
expensive operation to perform on matrices. The transpose $A^T$ of a
given $n \times m$ matrix $A$ is another matrix, having size $m \times n$,
and such that

$$
   A^T (i,j) = A(j,i) .
$$

In this exercise, the idea is *not* to modify the original matrix $A$ so that 
to transform it into its own transpose, but rather to create a new matrix where 
the transpose of $A$ is constructed. Please make reference to the partially-filled 
code [mattranspose.cu](./mattranspose.cu).

## Using a one-dimensional array for a matrix

The first function that you'll find in the provided ```cuda``` file is
the following:

	size_t index(size_t i,size_t j,size_t n);

which converts a pair of two-dimensional indices ```i``` and ```j``` 
into one unique mono-dimensional index ```k```. This indicates that
the matrices on which we are going to work are actually stored in
one-dimensional arrays of ```float``` types, and not in the 
typical two-dimensional arrays we may use in sequential programs.
The function ```index``` is therefore employed for performing the
necessary conversion from any pair $(i,j)$ of indices corresponding
to the standard matrix form, to the unique index $k$ corresponding to
$(i,j)$ in the mono-dimensional array.

Do you see why using a mono-dimensional array turns out to be
a good idea when working with GPUs? Think for example to the 
necessary step of transferring data from the RAM to the global
memory, and vice versa. Also notice the use of both ```__host__``` 
and ```__device__``` labels for the ```index``` function.

## Main steps

Please identify all missing parts in the provided code (marked with the 
comment ```TO BE COMPLETED``` in [mattranspose.cu](./mattranspose.cu)), and 
fill them in with your own implementations. The main steps the ```main``` 
function is expected to perform are the following:

1. generate a matrix by filling it either with random numbers, or rather
   by defining its elements through a predefined formula involving
   its indices;

2. compute the transpose of the generated matrix on the CPU;

3. transfer the transposed matrix to the global memory of the GPU;

4. transpose the matrix currently on the GPU global memory by invoking
   your kernel;

5. transfer back the obtained matrix;

6. compare the two matrices that are currently stored in the RAM:
   if your implementations are correct, the two matrices should be
   identical (modulo little round-off errors).

For simplicity, you can suppose that your matrices are square matrices.

## A two-dimensional topology for the GPU grid

Since the objects we are working with have a two-dimensional nature, 
it makes sense to consider to have a two-dimensional organization of the 
threads in our GPU. This two-dimensional topology occurs at more than
one level:

- the thread blocks can be organized in a two-dimensional topology 
  (this implies that there will be two indices to identify them, and
  not only one);

- thereafter, inside each block, the threads can also be organized in
  a two-dimensional topology (so they will also have two indices);

- finally, each thread can work on a two-dimensional *chunk* of matrix 
  elements, for which two indices need to be use in order to refer to each 
  single element.

## Pushing the limits

A relatively small setup is predefined at the beginning of our ```main```
function:

	size_t nblocks = 8;   // nblocks^2 = 64
	size_t nthreads = 8;  // nthreads^2 = 64
	size_t nchunks = 16;  // nchunks^2 = 256

It is a very good exercise to play with these numbers to see how they can
affect the performances of your program. While playing, pay attention to the 
fact that ```nblocks``` and ```nthreads``` are constrainted by the architecture 
of your GPU, while ```nchunks``` may push the global memory to the limits of its 
storage capabilities.

## Links

* [Back to GPU programming](./README.md)
* [Back to main repository page](../README.md)

