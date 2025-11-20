
# GPU programming

The content of this page covers the entire GPU programming course that I give 
at the [ESIR](https://esir.univ-rennes.fr/en/welcome-eisr-graduate-school-excellence)
engineering school of University of Rennes, and partially (as long as GPU programming
is concerned) the parallel computing course taught at [ISTIC](https://istic.univ-rennes.fr/) 
(with [Cédric Tedeschi](http://people.irisa.fr/Cedric.Tedeschi/index.html)), as well as
the parallel computing course taught at [SPM](https://spm.univ-rennes.fr/en) (with
[Mariko Dunseath-Terao](https://perso.univ-rennes1.fr/mariko.dunseath-terao/)).

## Lectures on GPU programming in CUDA

- [Lecture 1](./intro_ssp.md) : An introduction through an example (the Subset-Sum Problem);
- [Lecture 2](./vectorsum.md) : A GPU programming overview (Simple operations on vectors);
- [Lecture 3](./matrix-by-matrix.md) : 2D grids and coalesced memory access (Matrix by matrix multiplications);
- [Lecture 4](./shared-matrix.md) : Shared memory (Performing several operations on matrices);
- [Lecture 5](./sierpinski.md) : CUDA streams (Sierpiński fractal).

This [CUDA code](./exploratory.cu) is an initial attempt to cover several aspects 
of GPU programming in one unique program.

## Exercises

- [Prime factorization](./primes.md);
- [Computing the natural log of 2](./log2series.md);
- [Matrix transposition](./mattranspose.md);
- [The game of life](../HPC/game/README.md).

## Links

* [Back to main repository page](../README.md)

