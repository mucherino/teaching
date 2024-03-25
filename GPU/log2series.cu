
/* Log2series on GPU with CUDA
 *
 * nblocks = 64, nthreads = 128, nchunk = 1000000
 * ... and hence the total number of series terms to sum up is 8192000000
 * Sequential implementation ... val = 0.6931375265 (error: 10^-5), time = 28.6614
 * Memory allocation ... done
 * GPU implementation version 1 ... val = 0.6931376457 (error: 10^-5), elepsed time = 0.138585
 * GPU implementation version 2 ... val = 0.6928183436 (error: 10^-4), elapsed time = 0.109617
 * GPU implementation version 3 ... val = 0.6931439638 (error: 10^-6), elapsed time = 0.081293
 * Freeing memory ... ending
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// log2series in sequential
float log2series(size_t n)
{
   size_t i;
   short sign = 1.0;
   float val = 0.0;
   for (i = 0; i < n; i++)
   {
      val = val + ((float)sign)/(i+1);
      sign = -sign;
   };
   return val;
};

// GPU kernel version 1 -- we divide the work load in equal chunks
__global__ void log2series_GPUv1(size_t n,size_t nchunk,float *result)
{
   size_t i;
   unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
   // TO BE COMPLETED
};

// GPU kernel version 2 -- a different order for summing up the terms
__global__ void log2series_GPUv2(size_t n,float *result)
{
   size_t i;
   // TO BE COMPLETED
};

// GPU kernel version 3 -- we consider the version 2 but with the inverse order for the sums
__global__ void log2series_GPUv3(size_t n,float *result)
{
   // TO BE COMPLETED
};

// computing CPU time
float compute_time(time_t start,time_t end)
{
   return ((float)((int)end - (int)start))/CLOCKS_PER_SEC;
};

// computing the order of magnitude for the difference between two float numbers
int diff(float a,float b)
{
   int df = 0;
   float d = b - a;
   if (d < 0.0)  d = -d;
   float avg = 0.5*(a + b);
   d = d/avg;
   if (d != 0.0)
   {
      while (d < 1.0)
      {
         d = 10.0*d;
         df++;
      };
   };
   return df;
};

// main
int main(int argc,char *argv[])
{
   unsigned int i;
   size_t warp_size = 32;
   size_t warps_per_block = 4;
   size_t nthreads = warp_size*warps_per_block;
   size_t nblocks = 64;
   size_t total_threads = nblocks*nthreads;
   size_t nchunk = 1000000;
   size_t n = nchunk*total_threads;
   time_t startclock,endclock;
   float *psums,*psumsOnGPU;
   float val;
   float time;

   // welcome message
   fprintf(stderr,"Log2series on GPU with CUDA\n");
   fprintf(stderr,"nblocks = %lu, nthreads = %lu, nchunk = %lu\n",nblocks,nthreads,nchunk);
   fprintf(stderr,"... and hence the total number of series terms to sum up is %lu\n",n);

   // computations on CPU
   fprintf(stderr,"Sequential implementation ... ");
   startclock = clock();
   val = log2series(n);
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"val = %12.10f (error: 10^-%d), time = %g\n",val,diff(log(2),val),time);

   // memory allocation
   fprintf(stderr,"Memory allocation ... ");
   psums = (float*)calloc(total_threads,sizeof(float));
   // memory allocation on GPU ... TO BE COMPLETED ...
   fprintf(stderr,"done\n");

   // invoking the kernel (version 1)
   fprintf(stderr,"GPU implementation version 1 ... ");
   startclock = clock();
   log2series_GPUv1<<<nblocks,nthreads>>>(n,nchunk,psumsOnGPU);  // running on GPU!
   cudaDeviceSynchronize();  // synchronization!

   // moving the partial sums to the RAM ... TO BE COMPLETED ...

   // finalizing the computation on CPU
   val = 0.0;
   for (i = 0; i < total_threads; i++)  val = val + psums[i];
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"val = %12.10f (error: 10^-%d), elepsed time = %g\n",val,diff(log(2),val),time);

   // cleaning CPU array for the partial sums
   for (i = 0; i < total_threads; i++)  psums[i] = 0.0;

   // invoking the kernel (version 2)
   fprintf(stderr,"GPU implementation version 2 ... ");
   startclock = clock();
   // TO BE COMPLETED : THE KERNEL NEEDS TO BE INVOKED HERE
   cudaDeviceSynchronize();  // synchronization!

   // moving the partial sums to the RAM ... TO BE COMPLETED ...

   // finalizing the computation on CPU
   val = 0.0;
   for (i = 0; i < total_threads; i++)  val = val + psums[i];
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"val = %12.10f (error: 10^-%d), elapsed time = %g\n",val,diff(log(2),val),time);

   // cleaning CPU array for the partial sums
   for (i = 0; i < total_threads; i++)  psums[i] = 0.0;

   // invoking the kernel (version 3)
   fprintf(stderr,"GPU implementation version 3 ... ");
   startclock = clock();
   // TO BE COMPLETED : THE KERNEL NEEDS TO BE INVOKED HERE
   cudaDeviceSynchronize();  // synchronization!

   // moving the partial sums to the RAM ... TO BE COMPLETED ...

   // finalizing the computation on CPU
   val = 0.0;
   for (i = 0; i < total_threads; i++)  val = val + psums[i];
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"val = %12.10f (error: 10^-%d), elapsed time = %g\n",val,diff(log(2),val),time);

   // freeing memory and ending
   fprintf(stderr,"Freeing memory ... ");
   cudaFree(psumsOnGPU);
   free(psums);
   fprintf(stderr,"ending\n");
   return 0;
};

