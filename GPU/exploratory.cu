
/* Exploratory CUDA code
 *
 * We implement a useless algorithm with the only aim of exploring 
 * the various possibilities offered by modern GPUs to speed up the 
 * access to data.
 *
 * Even if our data array will be initialized with a constant value,
 * our algorithms suppose that the array elements may actually be 
 * different.
 *
 * Every kernel repeats the implemented algorithms as many times as 
 * the number of blocks in the GPU grid.
 *
 * For simplicity, the implementation is based on a 1D GPU grid.
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>

// kernel 0
// -> every thread inserts the value '1' in the array element assigned to itself
__global__ void initialization(unsigned int *array,bool *answer)
{
   array[blockDim.x*blockIdx.x + threadIdx.x] = 1;
   if (threadIdx.x == 0)  *answer = true;
};

// kernel 1
// -> each thread sums up the terms in its own thread block,
//    and placing it in its own array element
__global__ void kernel1(unsigned int *array)
{
   size_t id_first = blockDim.x*blockIdx.x;
   size_t id_last = id_first + blockDim.x;
   unsigned int sum = 0;
   for (size_t it = 0; it < blockDim.x; it++)
   {
      for (size_t i = id_first; i < id_last; i++)  sum = sum + array[i];
   };
   __syncthreads();
   array[id_first + threadIdx.x] = sum;
};

// kernel 2
// -> same as kernel 1, but the order of the sums is inverted
__global__ void kernel2(unsigned int *array)
{
   size_t id_first = blockDim.x*blockIdx.x;
   size_t id_last = id_first + blockDim.x;
   unsigned int sum = 0;
   for (size_t it = 0; it < blockDim.x; it++)
   {
      for (size_t i = 0; i < blockDim.x; i++)  sum = sum + array[id_last - i - 1];
   };
   __syncthreads();
   array[id_first + threadIdx.x] = sum;
};

// kernel 3
// -> same as kernal 1, but with memory access by 'sliding' coalesced blocks
__global__ void kernel3(unsigned int *array)
{
   size_t id_smallest = blockDim.x*blockIdx.x;
   size_t id_first = id_smallest + threadIdx.x;
   size_t id_largest = id_smallest + blockDim.x;
   unsigned int sum = 0;
   for (size_t it = 0; it < blockDim.x; it++)
   {
      for (size_t i = id_first; i < id_largest; i++)  sum = sum + array[i];  // divergence :-(
      for (size_t i = id_smallest; i < id_first; i++)  sum = sum + array[i];
   };
   __syncthreads();
   array[id_first] = sum;
};

// kernel 4
// -> same as kernel 1, but with memory access by 'jumping' coalesced blocks
//    the jump has the size of a warp
__global__ void kernel4(unsigned int *array)
{
   size_t warp_size = 32;
   size_t id_first = blockDim.x*blockIdx.x;
   size_t id_last = id_first + blockDim.x;
   unsigned int sum = 0;
   for (size_t it = 0; it < blockDim.x; it++)
   {
      for (size_t k = 0; k < warp_size; k++)
      {
         for (size_t i = k + id_first; i < id_last; i+=warp_size)  sum = sum + array[i];
      };
   };
   __syncthreads();
   array[id_first + threadIdx.x] = sum;
};

// kernel 5
// -> same as kernel 4, but we use the shared memory
__global__ void kernel5(unsigned int *array,size_t warp_size)
{
   extern __shared__ unsigned int shared_mem[];
   size_t my_unique_id = blockDim.x*blockIdx.x + threadIdx.x;

   // moving the data in each block's shared memories
   shared_mem[threadIdx.x] = array[my_unique_id];
   __syncthreads();

   // performing the computations
   unsigned int sum = 0;
   for (size_t it = 0; it < blockDim.x; it++)
   {
      for (size_t k = 0; k < warp_size; k++)
      {
         for (size_t i = k; i < blockDim.x; i+=warp_size)  sum = sum + shared_mem[i];
      };
   };
   __syncthreads();
   array[my_unique_id] = sum;
};

// kernel 6
// -> with shared memory, but simply linear access
__global__ void kernel6(unsigned int *array)
{
   extern __shared__ unsigned int shared_mem[];
   size_t my_unique_id = blockDim.x*blockIdx.x + threadIdx.x;

   // moving the data in each block's shared memories
   shared_mem[threadIdx.x] = array[my_unique_id];
   __syncthreads();

   // performing the computations
   unsigned int sum = 0;
   for (size_t it = 0; it < blockDim.x; it++)
   {
      for (size_t i = 0; i < blockDim.x; it++)  sum = sum + shared_mem[i];
   };
   __syncthreads();
   array[my_unique_id] = sum;
};

// kernel 7
// -> each thread verifies whether or not its array element contains 
//    the expected value (we suppose that 'answer' is initialized to true)
__global__ void verification(unsigned int expected,unsigned int *array,bool *answer)
{
   if (expected != array[blockDim.x*blockIdx.x + threadIdx.x])  // divergence occurs only when 
      *answer = false;                                          // the verification fails ...
};

/// computing CPU clock time
float compute_time(time_t start,time_t end)
{
   return 1000.0f*((float)((int)end - (int)start))/CLOCKS_PER_SEC;
};

// main
int main(int argc,char *argv[])
{
   size_t warp_size = 32;
   size_t warps_per_block = 4;
   size_t nthreads = warp_size*warps_per_block;
   size_t nblocks = 512;
   size_t nchunk = 10000;
   size_t total_threads = nblocks*nthreads;
   size_t n = nchunk*total_threads;
   unsigned int *array;
   bool answer_on_RAM;
   bool *answer_on_GPU;
   time_t startclock,endclock;
   float time;

   // welcome message
   printf("Exploratory code in CUDA\n");
   printf("nblocks = %lu, nthreads = %lu, nchunk = %lu\n",nblocks,nthreads,nchunk);
   printf("total number of threads is %lu\n",total_threads);
   printf("... and hence the total array size is %lu\n\n",n);

   // memory allocation on GPU
   cudaMalloc((void**)&array,n*sizeof(unsigned int));
   cudaMalloc((void**)&answer_on_GPU,sizeof(bool));

   // kernel 1
   printf("Kernel1\n");
   printf("-> every threads sums up the elements of its own block\n");
   printf("-> traditional for loop implemented\n");
   initialization<<<nblocks,nthreads>>>(array,answer_on_GPU);
   cudaDeviceSynchronize();
   startclock = clock();
   kernel1<<<nblocks,nthreads>>>(array);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf("Done in %7.4fms\n",time);

   // calling last kernel for the verification
   printf("Verification ... ");
   answer_on_RAM = false;
   verification<<<nblocks,nthreads>>>(nthreads*nthreads,array,answer_on_GPU);
   cudaDeviceSynchronize();
   cudaMemcpy(&answer_on_RAM,answer_on_GPU,sizeof(bool),cudaMemcpyDeviceToHost);
   if (!answer_on_RAM)  printf("NOT ");
   printf("OK\n\n");

   // kernel 2
   printf("Kernel2\n");
   printf("-> the memory access by reverse order\n");
   initialization<<<nblocks,nthreads>>>(array,answer_on_GPU);  
   cudaDeviceSynchronize();
   startclock = clock();
   kernel2<<<nblocks,nthreads>>>(array);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf("Done in %7.4fms\n",time);

   // calling last kernel for the verification
   printf("Verification ... ");
   answer_on_RAM = false;
   verification<<<nblocks,nthreads>>>(nthreads*nthreads,array,answer_on_GPU);
   cudaDeviceSynchronize();
   cudaMemcpy(&answer_on_RAM,answer_on_GPU,sizeof(bool),cudaMemcpyDeviceToHost);
   if (!answer_on_RAM)  printf("NOT ");
   printf("OK\n\n");

   // kernel 3
   printf("Kernel3\n");
   printf("-> coalesced memory access with 'sliding' contiguous blocks\n");
   initialization<<<nblocks,nthreads>>>(array,answer_on_GPU);
   cudaDeviceSynchronize();
   startclock = clock();
   kernel3<<<nblocks,nthreads>>>(array);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf("Done in %7.4fms\n",time);

   // calling last kernel for the verification
   printf("Verification ... ");
   answer_on_RAM = false;
   verification<<<nblocks,nthreads>>>(nthreads*nthreads,array,answer_on_GPU);
   cudaDeviceSynchronize();
   cudaMemcpy(&answer_on_RAM,answer_on_GPU,sizeof(bool),cudaMemcpyDeviceToHost);
   if (!answer_on_RAM)  printf("NOT ");
   printf("OK\n\n");

   // kernel 4
   printf("Kernel4\n");
   printf("-> coalesced memory access with 'jumping' contiguous blocks\n");
   initialization<<<nblocks,nthreads>>>(array,answer_on_GPU);
   cudaDeviceSynchronize();
   startclock = clock();
   kernel4<<<nblocks,nthreads>>>(array);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf("Done in %7.4fms\n",time);

   // calling last kernel for the verification
   printf("Verification ... ");
   answer_on_RAM = false;
   verification<<<nblocks,nthreads>>>(nthreads*nthreads,array,answer_on_GPU);
   cudaDeviceSynchronize();
   cudaMemcpy(&answer_on_RAM,answer_on_GPU,sizeof(bool),cudaMemcpyDeviceToHost);
   if (!answer_on_RAM)  printf("NOT ");
   printf("OK\n\n");

   // kernel 5
   printf("Kernel5\n");
   printf("-> same coalesced access but with shared memory\n");
   initialization<<<nblocks,nthreads>>>(array,answer_on_GPU);
   cudaDeviceSynchronize();
   startclock = clock();
   kernel5<<<nblocks,nthreads,nthreads*sizeof(unsigned int)>>>(array,warp_size);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf("Done in %7.4fms\n",time);

   // calling last kernel for the verification
   printf("Verification ... ");
   answer_on_RAM = false;
   verification<<<nblocks,nthreads>>>(nthreads*nthreads,array,answer_on_GPU);
   cudaDeviceSynchronize();
   cudaMemcpy(&answer_on_RAM,answer_on_GPU,sizeof(bool),cudaMemcpyDeviceToHost);
   if (!answer_on_RAM)  printf("NOT ");
   printf("OK\n\n");

   // kernel 5 again
   printf("Kernel5\n");
   printf("-> we make sure now that the blocks do not match with the bank size\n");
   initialization<<<nblocks,nthreads>>>(array,answer_on_GPU);
   cudaDeviceSynchronize();
   startclock = clock();
   kernel5<<<nblocks,nthreads,nthreads*sizeof(unsigned int)>>>(array,5);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf("Done in %7.4fms\n",time);

   // calling last kernel for the verification
   printf("Verification ... ");
   answer_on_RAM = false;
   verification<<<nblocks,nthreads>>>(nthreads*nthreads,array,answer_on_GPU);
   cudaDeviceSynchronize();
   cudaMemcpy(&answer_on_RAM,answer_on_GPU,sizeof(bool),cudaMemcpyDeviceToHost);
   if (!answer_on_RAM)  printf("NOT ");
   printf("OK\n\n");

   // kernel 6
   printf("Kernel6\n");
   printf("-> using shared memory with a simple linear access\n");
   initialization<<<nblocks,nthreads>>>(array,answer_on_GPU);
   cudaDeviceSynchronize();
   startclock = clock();
   kernel6<<<nblocks,nthreads,nthreads*sizeof(unsigned int)>>>(array);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf("Done in %7.4fms\n",time);

   // calling last kernel for the verification
   printf("Verification ... ");
   answer_on_RAM = false;
   verification<<<nblocks,nthreads>>>(nthreads*nthreads,array,answer_on_GPU);
   cudaDeviceSynchronize();
   cudaMemcpy(&answer_on_RAM,answer_on_GPU,sizeof(bool),cudaMemcpyDeviceToHost);
   if (!answer_on_RAM)  printf("NOT ");
   printf("OK\n\n");

   // ending
   cudaFree(answer_on_GPU);
   cudaFree(array);
   return 0;
};

/* Run on Google Colab (GPU: Nvidia T4)
 *
 * Exploratory code in CUDA
 * nblocks = 512, nthreads = 128, nchunk = 10000
 * total number of threads is 65536
 * ... and hence the total array size is 655360000
 * 
 * Kernel1
 * -> every threads sums up the elements of its own block
 * -> traditional for loop implemented
 * Done in  3.3390ms
 * Verification ... OK
 *
 * Kernel2
 * -> the memory access by reverse order
 * Done in  3.0310ms
 * Verification ... OK
 *
 * Kernel3
 * -> coalesced memory access with 'sliding' contiguous blocks
 * Done in  3.7660ms
 * Verification ... OK
 *
 * Kernel4
 * -> coalesced memory access with 'jumping' contiguous blocks
 * Done in  8.1430ms
 * Verification ... OK
 *
 * Kernel5
 * -> same coalesced access but with shared memory
 * Done in  3.3500ms
 * Verification ... OK
 *
 * Kernel5
 * -> we make sure now that the blocks do not match with the bank size
 * Done in  2.7790ms
 * Verification ... OK
 * 
 * Kernel6
 * -> using shared memory with a simple linear access
 * Done in  1.6470ms
 * Verification ... OK
 *
 */

