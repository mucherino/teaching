
# Exercise: computing mathematical series on GPU

## Computing $\ln(2)$

An approximation of the natural logarithm of 2 can be computed as:

$$
\ln(2) = \sum_{i=0}^{n-1} \frac{(-1)^i}{i+1} .
$$

In this assignment, we want to compute this approximation for a very 
large value of $n$. To this purpose, we are going to write our program 
in CUDA, in order to exploit the power of a GPU device.

## Getting started

Give a careful look at the [provided code](./log2series.cu) and find all 
locations where the comment ```TO BE COMPLETED``` appears. These are the 
blocks of code that you'll have to fill in.

## Writing the first kernel

For the first kernel, the one named ```log2series_GPUv1```, we wish to 
implement a very simple approach in GPU programming where a chunk of 
consecutive terms of the series is assigned to each thread. The partial 
sum that each thread computes is then stored in the appropriate element 
of the array allocated on the GPU, and subsequently "sent" to the CPU. The 
final computation, i.e. the sum of all partial sums, is performed by the 
CPU (this part of the code is provided).

## Preparing the main

The kernel you have just written needs to be invoked from the main function. 
There are three main steps that need to be completed in the main function in 
order to execute your first kernel:

- the memory allocation on the GPU;
- the call to the kernel;
- the memory transfer from the GPU to the CPU.

Once these three steps implemented, you can comment out the rest of the code 
(except the final calls to ```free``` and ```return```). This way, you'll be 
able to test your first kernel on GPU before implementing the subsequent tasks.

Please compile your program with the ```nvcc``` compiler, and execute it.
Do you remark any improvements? What about the quality of the result? 

## Improving the performances

It is time now to write our second kernel, the one named ```log2series_GPUv2```.
Pay particular attention to the operations that are executed in the first
version of your kernel, and try to find a different way to associate terms
to threads in such a way to reduce the total number of required operations. 
When you'll be confident about your new kernel, complete the main function 
for its execution, compile and test it. Do you remark any changes?

## Improving the quality of the results

Take now your previous kernel, copy its code inside the third kernel, and then 
modify it so that each thread computes the partial sums by taking into consideration 
the smallest terms at first (i.e. the ones where the denominator ```i+1``` is 
larger), and iterating in the "direction" of larger terms. Complete everything 
necessary in the main function to run this third kernel. Then compile, execute, 
and observe the results.

## Links

* [Back to GPU programming](./README.md)
* [Back to main repository page](../README.md)

