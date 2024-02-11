
# GPU programming

Two simple programs in CUDA are currently available on this page.

## Reading GPU threads identifiers

This simple program in CUDA outputs the following thread 
identifiers:

- the identifier of each thread in its block;
- the block identifier for each thread;
- and it also computes a unique identifier for each thread.

It is supposed that the thread topology is linear (i.e. one 
dimensional).

The code can be downloaded [here](./identifiers.cu)

## Performing the sum of two vectors

This program in CUDA generates two vectors of length $n$ 
containing integer elements ranging from 0 to 9, and it
performs the sum of the two vectors in three different ways:

1. it performs the computations in sequential on one core of
   the CPU;
2. it performs the computations on GPU, without paying
   any attention to the way the vector elements are accessed 
   by the threads on the GPU;
3. it performs the computations on GPU by making sure that
   the access to vector elements is *coalesced*.

The code can be downloaded [here](./vectorsum.cu)

## Links

* [Back to HPC course](../HPC.md)
* [Back to main repository page](../README.md)

