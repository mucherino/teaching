
# GPU programming

The CUDA programs below are used during the lectures for 
explaining some fundamental facts about GPU programming. 
In case you have the possibility, you should try to run 
these programs on GPU, and play a little with the proposed
configurations (number of blocks, number of threads).

## Reading the identifiers of GPU threads

This simple program in CUDA outputs the following thread 
identifiers:

- the identifier of each thread in its block;
- the block identifier for each thread;
- and it also computes a unique identifier for each thread.

It is supposed that the thread topology is linear (i.e. one 
dimensional).

The code is in the file [identifiers.cu](./identifiers.cu).

## Subset Sum Problem (SSP) on GPU

Given a set of integers $S$ and a target $t$, the Subset Sum 
Problem (SSP) asks whether there exists a subset of $S$ such
that the sum of the elements in the subset corresponds to the 
target $t$. This is a fundamental problem in computer science. 
For more information, please visit this
[wikipedia page](https://en.wikipedia.org/wiki/Subset_sum_problem).

This CUDA program compares two simple methods for the SSP:

- the first method is sequential and implements the basic recursive 
  approach for counting the number of solutions admitted by an 
  instance of the SSP (i.e. the number of subsets of $S$ summing up 
  to $t$);
- the second method consists in allocating as many threads on the
  GPU as the number of potential subsets, and to let every thread
  compute *only one* sum (it's the CPU then that collects the
  sums and verifies whether they correspond to the target).

Notice that the order in which the operations are performed in the
two implementations have been adapted to the structure of GPU programs.

The code is in the file [ssp.cu](./ssp.cu).

## Computing the sum of two vectors

This program in CUDA is for performing a very simple operation: the sum 
of two $n$-dimensional vectors. This CUDA program is here proposed with 
the aim of focusing our attention on how threads can collect data from the
global memory during the executions.

The program initially generates the two vectors (of given length $n$)
and then it computes the sum of the two vectors in three different 
ways:

1. it performs the computations in sequential on one core of
   the CPU;
2. it performs the computations on GPU, without paying
   any attention to the way the vector elements are accessed 
   by the threads on the GPU;
3. it performs the computations on GPU by making sure that
   the access to the vector elements is *coalesced*.

The code is in the file [vectorsum.cu](./vectorsum.cu).

## Exercises

- [Computing the natural log of 2 on GPU](./log2series.md);
- [Matrix transposition on GPU](./mattranspose.md).

## Links

* [Back to HPC course](../HPC.md)
* [Back to main repository page](../README.md)

