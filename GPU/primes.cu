
/*
 * Prime factorization on GPU
 *
 * comparing sequential and GPU versions of the naive algorithm
 * that checks all possible divisors
 *
 * Naive prime factorization algorithm with CUDA on GPU
 * nblocks = 1024, nthreads = 128, nchunks = 1000
 * Randomly generated number = 201607572
 * Sequential version took 0.618820 seconds
 * The divisors are: 2 3 4 6 12 16800631 33601262 50401893 67202524 100803786 
 * GPU version took 0.020606 seconds
 * The divisors are: 2 3 4 6 12 16800631 33601262 50401893 67202524 100803786 
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

// checking divisors in sequential
void check_divisors_seq(unsigned long number,size_t n,bool *isdivisor)
{
   isdivisor[0] = false;
   isdivisor[1] = false;
   for (size_t i = 2; i < n; i++)  isdivisor[i] = !(number%i);  // equivalent to number%i != 0
};

// kernel
__global__ void check_divisors(unsigned long number,size_t nchunks,bool *isdivisorGPU)
{
   // TO BE COMPLETED!
};

// computing CPU time
float compute_time(time_t start,time_t end)
{
   return ((float)((int)end - (int)start))/CLOCKS_PER_SEC;
};

// main
int main(int argc,char *argv[])
{
   size_t warp_size = 32;
   size_t warps_per_block = 4;
   size_t nthreads = warp_size*warps_per_block;
   size_t nblocks = 1024;
   size_t nchunks = 1000;
   size_t total_threads = nblocks*nthreads;
   size_t n = nchunks*total_threads;
   unsigned long number;
   bool *isdivisor,*isdivisorGPU;
   time_t startclock,endclock;
   float time;

   // welcome message
   printf("Naive prime factorization algorithm with CUDA on GPU\n");
   printf("nblocks = %lu, nthreads = %lu, nchunks = %lu\n",nblocks,nthreads,nchunks);

   // random generation of the "number" (between total_threads and its double)
   srand(999);
   number = n + rand()%n;
   printf("Randomly generated number = %lu\n",number);

   // memory allocations for boolean arrays ... TO BE COMPLETED!

   // running the naive algorithm in sequential
   startclock = clock();
   check_divisors_seq(number,n,isdivisor);
   endclock = clock();
   time = compute_time(startclock,endclock);

   // printing the result obtained in sequential
   printf("Sequential version took %f seconds\n",time);
   printf("The divisors are: ");
   for (size_t i = 2; i < n; i++)  if (isdivisor[i])  printf("%ld ",i);
   printf("\n");

   // cleaning up the boolean array
   for (size_t i = 0; i < n; i++)  isdivisor[i] = false;

   // running the naive algorithm on GPU
   startclock = clock();
   check_divisors<<<nblocks,nthreads>>>(number,nchunks,isdivisorGPU);
   cudaDeviceSynchronize();   
   endclock = clock();
   time = compute_time(startclock,endclock);

   // transferring the result to the RAM ... TO BE COMPLETE!

   // printing the result obtained in sequential
   printf("GPU version took %f seconds\n",time);
   printf("The divisors are: ");
   for (size_t i = 2; i < n; i++)  if (isdivisor[i])  printf("%ld ",i);
   printf("\n");

   // freeing memory ... TO BE COMPLETED!

   // ending
   return 0;
};

