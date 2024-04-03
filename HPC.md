
# High Performance Computing

The content of this page represents an entire course for the 
[ESIR](https://esir.univ-rennes.fr/en/welcome-eisr-graduate-school-excellence)
engineering school of University of Rennes, and partially for the parallel 
computing course taugh at [ISTIC](https://istic.univ-rennes.fr/).
This second course is given in collaboration with my colleague
[Cédric Tedeschi](http://people.irisa.fr/Cedric.Tedeschi/index.html).

## Lectures on GPU programming

Please find below the main lecture line:

* Comparisons between CPU and GPU architectures
* Something new we need to pay attention at:
  - code divergence
  - coalesced memory access
* Measuring the performance of GPU programs
* Some simple examples (in pseudo-code):
  - vector sum
  - scalar product
* Main steps in GPU programming
* CUDA language:
  - memory allocation on GPU
  - data transfer between CPU and GPU
  - thread identifiers
  - the kernel
* Two complete examples in CUDA:
  - the sum of two vectors
  - the solution to a combinatorial problem
* The misteries of shared memory

Some of the codes proposed during the lectures can be found
in [this page](./GPU/README.md).

## Exercises on GPU programming

- [Computing the natural log of 2 on GPU](./GPU/log2series.md).

## Links

* [Back to main repository page](README.md)

