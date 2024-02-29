
# Convey's Game of Life

This page proposes some exercises on the famous *game of life* devised 
by the British mathematician John Horton Conway (1937-2020) in 1970. A 
simple sequential implementation in C programming language of the game 
is available in the repository: [gamelife.c](./gamelife.c). The proposed 
exercises have as a main aim to explore different possible ways to improve 
the performances of the game in different programming languages and paradigms.

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
torus will completely be formed by dead cells (no extinction possible, in order 
words). This is one of the reasons why people are interested in computing many 
many new generations in the game of life, and to compute them fast.

There is a lot of material on the Internet related to the game of life.
This [wikipedia page](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
contains a lot of information on the game of life, and you can also find
on it the main rules of the game. If you're interested in knowing more
about the topic, the YouTube channel 
[Numberphile](https://www.youtube.com/@numberphile)
has devoted several videos to Convey and its game of life. This is 
[one of these videos](https://www.youtube.com/watch?v=R9Plq-D1gEk).

# A simple C version

The file [gamelife.c](./gamelife.c) proposes a simple implementation in
C programming language of the game of life. The torus is represented by
using the following data structure:

	// torus structure
	typedef struct
	{
	   size_t n;
	   size_t m;
	   bool* t;
	}  torus_t;

where the array of boolean variables indicates the state of each cell 
(0 for dead; 1 for alive). Notice the use of a one-dimensional array for
storing a two-dimensional object. This choice is not only motivated by the
need of speeding up memory access (recall we want to improve performances), 
but it will also become useful later on when we will work on some parallel
implementations of the game. In order to convert 2-dimensional indices $(x,y)$
into a unique 1-dimensional index $z$, the following function is provided:

	size_t torus_index_convert(torus_t *torus,size_t x,size_t y,int dx,int dy);

Notice also the presence of a function capable to create random initial 
states for the torus, as well as another which can read predefined models 
(these models may be particularly useful when debugging your code, because 
their behavior is a priori known, see the 
[wikipedia page](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)).
Text files containing some of these models are available in the current
folder of the repository.

## The game in MPI

The idea is to adapt the provided sequential implementation so that the 
calculations can be performed in parallel by a cluster of computers.
We will spread the cells of the torus over the memories of the various
involved processors, so that each processor will be able to work on its
own block of contiguous rows in the torus. However, we can immediately 
remark that the computations for each row block cannot be performed in
a completely independent way by the several processors, and therefore
a communication step is necessary when stepping from one generation
to another. The necessity to perform these communications implies that
our parallel version is going to be more performing of the sequential one 
only on very large toruses, but you can test in any case your new 
implementation in MPI on small examples (such as the models provided
in this folder).

In order to make our "life" a little easier, we will suppose that the 
total number of rows in the torus is divisible by the number of processors
involved in the computation. Moreover, we will suppose that the number
of processors is a power of 2, because this will allow us to devise a 
simpler communication scheme. Finally, we will allocate memory for storing 
the entire torus (and not only its row blocks) on the RAM associated to 
every processor. In this way, it will be easy for each processor to predict 
the row block borders for its neighbors.

Your MPI implementation needs to execute the following main steps (tasks
already implemented in the sequential version are not mentioned):

- the MPI environment is initialized so that a dedicated process will run
  on each of the allocated processors;
- the assumptions mentioned in the paragraph above are verified by each
  process, and the execution is aborted should the verification fail;
- all processes read the preselected text files containing the initial
  state for the torus;
- process 0 prints the current state;
- each process identifies the row block on which it is going to work;
  basically the first and last row index delimiting each row block are computed;
- in the main ```for``` loop, every process works on its own row block;
- every process exchanges with its neighbors the information necessary for
  working on the next generations (one-to-one communications, involving 
  all processes);
- the final torus is constructed in the memory of process 0 by combining the 
  various row blocks on which the processes have worked (this final 
  communication step is of type all-to-one);
- process 0 prints the final result.

## Bonus in MPI

Once you're done with the implementation in MPI of the game of life, you may
try to identify the lines of your code that you'd need to modify in order to 
avoid allocating memory for the entire torus for every process. In other words,
how to deal with the indices when process 0 has the entire torus, while the 
others only store the rows that they strictly need?

## More games!

In case you'd write new text file containing some other interesting initial
states for the torus, please don't hesitate to send them to your teacher, or 
to add them to the repository via a "push request".

## Links

* [Back to main repository page](../../README.md)

