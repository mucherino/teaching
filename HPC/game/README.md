
# Convey's Game of Life

This page proposes some exercises on the famous *game of life* devised 
by the British mathematician John Horton Conway (1937-2020). A simple 
sequential implementation in C programming language of the game is provided
(see file [gamelife.c](./gamelife.c)). The proposed exercises have as a 
main aim to explore different possible ways to improve the performances of 
the game in different programming languages and paradigms.

## Introducing to the "game"

The game of life is a cellular automaton, represented in its original form
as a 2-dimensional torus where cells admit only two possible states: *dead*
or *alive*. The game of life is based on some simple rules that allow the torus 
to automatically evolve to the next generations. From an initial torus state 
(i.e. from a known state for all the cells forming the torus), each cell may 
be found in new different states at every new computed generation. Cells 
surrounded by too many other alive cells may in fact die as by overpopulation; 
other cells, in different conditions, may instead become alive. For some initial 
states, the game of life is infinite, i.e. no future generation exists where the
torus will completely be formed by dead cells (no extinction is possible, in 
order words). This is one of the reasons why people are interested in computing 
many many new generations in the game of life, and to compute them fast.

There is a lot of material on the Internet related to the game of life.
This [wikipedia page](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
contains a lot of information about the game of life, and you can also find
on it the main rules of the game. If you're interested in knowing more
about the topic, the YouTube channel 
[Numberphile](https://www.youtube.com/@numberphile)
has devoted several videos to Convey and its game of life. This is 
[one of the videos](https://www.youtube.com/watch?v=R9Plq-D1gEk).

# A simple C version

The file [gamelife.c](./gamelife.c) proposes a simple implementation in
C programming language of the game of life. The torus is represented by
using the following data structure:

	typedef struct
	{
	   size_t n;
	   size_t m;
	   bool* t;
	}  torus_t;

where the array of Boolean variables indicates the state of each cell 
(0 for dead; 1 for alive). Notice the use of a one-dimensional array for
storing a two-dimensional object. This choice is not only motivated by the
need of speeding up memory access (recall we want to improve performances), 
but it will also become useful later on when we will work on some parallel
implementations of the game. In order to convert 2-dimensional indices $(x,y)$
into a unique 1-dimensional index $z$, the following function is provided:

	size_t torus_index(size_t n,size_t m,size_t x,size_t y,int dx,int dy)

Notice also the presence of a C function capable to create random initial 
states for the torus, as well as of another which can read predefined models 
(these models may be particularly useful when debugging your code, because 
their behavior is a priori known, see the 
[wikipedia page](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)).
Text files containing some of these models are available in the current
folder of the repository.

## The game in MPI

The idea is to adapt the provided sequential implementation in such a way 
that the calculations are performed in parallel by a cluster of computers.
We will spread the cells of the torus over the memories of the various
involved processors, so that each processor will be able to work on its
own block of contiguous rows in the torus. However, we can immediately 
remark that the computations for each row block cannot be performed in
a completely independent way by the several processors, and therefore
a communication step is necessary when stepping from one generation
to another. Communications are likely to slow down our parallel 
executions, but a gain in speed can in any case be observed when 
working with tori having a very large size. In phase of development,
however, it is recommended to use the provided models (even if they are
quite small in size) and to compare the results with the sequential 
version.

In order to make our "life" a little easier, we will suppose that the 
total number of rows in the torus is divisible by the number of processors
involved in the computation. Moreover, we will suppose that the number
of processors is a power of 2, because this will allow us to devise a 
simpler communication scheme. Finally, we will verify that the number of
rows in the torus is not smaller than the total number of allocated
processors:

	if (rank == 0)
	{
	   if (size%2 != 0)
	   {
	      fprintf(stderr,"This MPI version can only work with an even number of processors!\n");
	      exit = true;
	   };
	   if (torus1->n / size == 0)
	   {
	      fprintf(stderr,"Too many processors (%d) for such a small torus (%lu rows) !\n",size,torus1->n);
	      exit = true;
	   };
	   if (torus1->n%size != 0)
	   {
	      fprintf(stderr,"The torus row size (%lu) needs to be divisible by the number of processors (%d)\n",torus1->n,size);
	      exit = true;
	   };
	};
	MPI_Barrier(MPI_COMM_WORLD);

We will allocate memory for storing the entire torus (and not only its 
row blocks) on the RAM associated to every processor. In this way, it will 
be easy for each processor to point to its row block and the surrounding
cells.

Your MPI implementation needs to execute the following main steps (tasks
already implemented in the sequential version are not mentioned):

- The MPI environment is initialized so that a dedicated process will run
  on each of the allocated processors.
- The assumptions given above are verified by each process (code snippet
  is provided), and the execution is aborted if the verification fails
  (remember that each processor is already running its own process at this
  stage).
- All processes read the preselected text files containing the initial
  state for the torus (you can reuse the sequential code).
- Process 0 prints the current torus.
- Each process identifies the row block on which it is going to work;
  basically the first and last row index delimiting each row block are computed.
- In the main ```for``` loop, every process works on its own row block.
- Every process exchanges with its neighbors the information necessary for
  working on the next generations: this is a one-to-one communications, which
  involves all processes.
- The final torus is constructed in the memory of process 0 by combining the 
  various row blocks on which the processes have worked (this final 
  communication step is of type all-to-one).
- Process 0 prints the obtained torus.

### Bonus in MPI

Once you're done with the implementation in MPI of the game of life, you may
try to identify the lines of your code that you'd need to modify in order to
avoid allocating memory for the entire torus for every process. In other words,
how to deal with the indices when process 0 has the entire torus, while the
others only store the rows that they strictly need?

## The game on GPU

The number of threads that can run in parallel on a GPU device is generally
much larger than the total number of processors that we can have for an 
MPI execution. Therefore, we focus our attention now on a possible CUDA 
implementation of the game of life where every cell of the torus is assigned 
to *one unique* thread during a parallel execution on GPU. The thread's 
topology needs for this reason to be two-dimensional, so that each thread
actually has two indices, which correspond in our case to the indices of
the cells in our torus. 

In order to start working on this CUDA implementation, please answer to 
the following questions:

- Which C functions in the [sequential version](./gamelife.c) can directly
  be (meaning: without any change) reused on the GPU? If you find any of such
  functions, don't forget to mark it with the proper label (e.g. ```___device___```).

In order to make our "life" easier, we will not try to transfer the data
type ```torus_t``` to the GPU global memory, but rather its array of Boolean's,
where the torus is actually stored. Suppose that the prototype for our kernel is:

	__global__ void torus_on_gpu(bool *torus1,bool *torus2);

- How to find out in the kernel how many rows and columns our torus has?

We can remark that some operations performed by the C functions in the sequential 
version will have to be rewritten in the kernel.

- Should we pay a particular attention on how these operations are re-implemented?
  Would it be important to avoid code divergence as much as possible?

Finally, a question about the synchronization of the threads.

- Where and how, in your CUDA code, it is convenient to synchronize the threads?

Please follow the main lines below for your CUDA implementation:

- Verify the specifications of your GPU and manually set up a grid 
  that will correspond to the shape of your torus.
- Use the C function for a random generation of the torus in order to
  create a torus which exactly fits with your grid.
- Run the game of life in sequential on the random torus for a predetermined
  number of generations.
- Print the obtained torus: you will use this torus to verify that your
  CUDA implementation is correct.
- Transfer the original random torus on the GPU global memory.
- Invoke the kernel.
- Retrieve the obtained torus from the global memory.
- Print the result.
- (*Optional*) Instead of printing, you can consider to write a C function,
  in sequential, that compares the two results, one obtained in sequential,
  the other on GPU.

## More games!

In case you'd write new text files containing some other interesting models 
(initial states for the torus), please don't hesitate to send them to your 
teacher, or to directly add them to the repository via a "push request".

## Links

* [Back to main repository page](../../README.md)

