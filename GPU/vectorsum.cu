
/*
 * Simple operations on vectors performed on GPU with CUDA
 *
 * Solving the following three problems:
 * - computing the sum of two vectors
 * - computing the conditional sum of two vectors
 * - computing the sum of all elements of a vector (reduction)
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// small kernal to clean up the content of a vector stocked in the global memory
// the memory access is coalesced
__global__ void cleanup(size_t n,int *gpu)
{
   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t nthreads = gridDim.x*blockDim.x;
   for (size_t i = id; i < n; i = i + nthreads)  gpu[i] = 0;
};

// computing the sum of two vectors on GPU
// this kernel is based on a global memory access which is not coalesced
__global__ void vectorsum_noncoalesced(size_t nchunk,int *xgpu,int *ygpu,int *zgpu)
{
   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
   for (size_t i = id*nchunk; i < (id+1)*nchunk; i++)  zgpu[i] = xgpu[i] + ygpu[i];
};

// computing the sum of two vectors on GPU
// this kernel is based on a global memory access which is coalesced
__global__ void vectorsum_coalesced(size_t n,int *xgpu,int *ygpu,int *zgpu)
{
   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t nthreads = gridDim.x*blockDim.x;
   for (size_t i = id; i < n; i = i + nthreads)  zgpu[i] = xgpu[i] + ygpu[i];
};

// computing the conditional sum of two vectors on GPU
// !! this kernel suffers of divergence issues !!
// in order to introduce the divergence, we suppose that:
// - threads with id%4 == 0 simply set up all z components to 0
// - threads with id%4 == 1 copy the content of the x component
// - threads with id%4 == 2 copy the content of the y component
// - threads with id%4 == 3 perform the sum of the two components of x and y
__global__ void vectorsum_divergent(size_t n,int *xgpu,int *ygpu,int *zgpu)
{
   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t nthreads = gridDim.x*blockDim.x;
   size_t k = id%4;
   for (size_t i = id; i < n; i = i + nthreads)
   {
      if (k == 0)
         zgpu[i] = 0;
      else if (k == 1)
         zgpu[i] = xgpu[i];
      else if (k == 2)
         zgpu[i] = ygpu[i];
      else
         zgpu[i] = xgpu[i] + ygpu[i];
   };
};

// computing the sum of two vectors on GPU
// this kernel performs the same operations as above
// **but without causing code divergence**
__global__ void vectorsum_without_divergence(size_t n,int *xgpu,int *ygpu,int *zgpu)
{
   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t nthreads = gridDim.x*blockDim.x;
   size_t alphax = id & 1UL;
   size_t alphay = (id >> 1) & 1UL;
   for (size_t i = id; i < n; i = i + nthreads)
   {
      zgpu[i] = alphax*xgpu[i] + alphay*ygpu[i];
   };
};

// computing the sum of all vector elements
// in this kernel, each thread computes its partial sum, which is then 'sent' 
// to the CPU for the final computation
__global__ void vectorsum_partial_reduction(size_t n,int *xgpu,int *pgpu)
{
   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t nthreads = gridDim.x*blockDim.x;
   int thread_partial = 0;
   for (size_t i = id; i < n; i = i + nthreads)  thread_partial = thread_partial + xgpu[i];
   pgpu[id] = thread_partial;  // there are as many partial sums as the number of threads
};

// computing the sum of all vector elements
// this kernel uses shared memory to store partial sums calculated by each thread;
// thread 0 of each block then reads these partial sums from shared memory and computes 
// the final sum for the block
__global__ void vectorsum_shared_reduction(size_t n,int *xgpu,int *pgpu)
{
   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t nthreads = gridDim.x*blockDim.x;
   extern __shared__ int shared_mem[];  // expected to have the size of a block
   shared_mem[threadIdx.x] = 0;
   for (size_t i = id; i < n; i = i + nthreads)  shared_mem[threadIdx.x] = shared_mem[threadIdx.x] + xgpu[i];
   __syncthreads();
   int block_partial = 0;
   if (threadIdx.x == 0)
   {
      for (size_t i = 0; i < blockDim.x; i++)  block_partial = block_partial + shared_mem[i];
      pgpu[blockIdx.x] = block_partial;  // only one partial sum per block goes back to the CPU!
   };
};

// computing wall clock time
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
   size_t nblocks = 512;
   size_t nchunk = 5000;
   size_t total_threads = nblocks*nthreads;
   size_t n = nchunk*total_threads;
   int *x,*xgpu;
   int *y,*ygpu;
   int *z,*zgpu;
   int *partial,*pgpu;  // partial sums
   time_t startclock,endclock;
   float gpu_time,cpu_time;
   float total_time;
   srand(time(NULL));

   // welcome message
   printf("Vector sums with CUDA\n");
   printf("nblocks = %lu, nthreads = %lu, nchunk = %lu\n",nblocks,nthreads,nchunk);
   printf("total number of threads is %lu\n",total_threads);
   printf("hence vector size is %lu\n",n);

   // memory allocation on CPU
   x = (int*)calloc(n,sizeof(int));
   y = (int*)calloc(n,sizeof(int));
   z = (int*)calloc(n,sizeof(int));

   // initializing data on CPU
   for (size_t i = 0; i < n; i++)  x[i] = 1 + rand()%9;
   for (size_t i = 0; i < n; i++)  y[i] = 1 + rand()%9;

   // Problem1: the sum z of two vectors (x + y)
   printf("Problem1: computing the sum z = x + y\n");

   // computing the vector sum on CPU
   printf("Computations on CPU ... ");
   startclock = clock();
   for (size_t i = 0; i < n; i++)  z[i] = x[i] + y[i];
   endclock = clock();
   total_time = compute_time(startclock,endclock);
   size_t k = rand()%n;
   printf(" done: elapsed time = %7.4f; verification: %d + %d = %d \n",total_time,x[k],y[k],z[k]);

   // cleaning up the solution vector
   for (size_t i = 0; i < n; i++)  z[i] = 0;

   // memory allocation on GPU
   cudaMalloc((void**)&xgpu,n*sizeof(int));
   cudaMalloc((void**)&ygpu,n*sizeof(int));
   cudaMalloc((void**)&zgpu,n*sizeof(int));

   // moving the vectors x and y in the global memory
   cudaMemcpy(xgpu,x,n*sizeof(int),cudaMemcpyHostToDevice);
   cudaMemcpy(ygpu,y,n*sizeof(int),cudaMemcpyHostToDevice);

   // cleaning up the content of z in the global memory (kernel call)
   cleanup<<<nblocks,nthreads>>>(n,zgpu);
   cudaDeviceSynchronize();

   // invoking the first kernel (non-coalesced access)
   printf("Computations on GPU (non-coalesced) ... ");
   startclock = clock();
   vectorsum_noncoalesced<<<nblocks,nthreads>>>(nchunk,xgpu,ygpu,zgpu);
   cudaDeviceSynchronize();
   endclock = clock();
   total_time = compute_time(startclock,endclock);
   printf(" done: elapsed time = %7.4f;",total_time);

   // moving the result to the RAM, and printing one of the sums
   cudaMemcpy(z,zgpu,n*sizeof(int),cudaMemcpyDeviceToHost);
   k = rand()%n;
   printf(" verification: %d + %d = %d\n",x[k],y[k],z[k]);

   // cleaning up (RAM and global memory)
   for (size_t i = 0; i < n; i++)  z[i] = 0;
   cleanup<<<nblocks,nthreads>>>(n,zgpu);
   cudaDeviceSynchronize();

   // invoking the second kernel (coalesced access)
   printf("Computations on GPU (coalesced) ... ");
   startclock = clock();
   vectorsum_coalesced<<<nblocks,nthreads>>>(n,xgpu,ygpu,zgpu);
   cudaDeviceSynchronize();
   endclock = clock();
   total_time = compute_time(startclock,endclock);
   printf(" done: elapsed time = %7.4f;",total_time);

   // moving the result to the RAM, and printing one of the sums
   cudaMemcpy(z,zgpu,n*sizeof(int),cudaMemcpyDeviceToHost);
   k = rand()%n;
   printf(" verification: %d + %d = %d\n",x[k],y[k],z[k]);

   // cleaning up (RAM and global memory)
   for (size_t i = 0; i < n; i++)  z[i] = 0;
   cleanup<<<nblocks,nthreads>>>(n,zgpu);
   cudaDeviceSynchronize();

   // Problem2: the conditional sum z of the two vectors
   printf("Problem2: computing the conditional sum z = {either x, or x, or y, or x + y}\n");

   // invoking the third kernel (with divergence)
   printf("Computations on GPU (divergent) ... ");
   startclock = clock();
   vectorsum_divergent<<<nblocks,nthreads>>>(n,xgpu,ygpu,zgpu);
   cudaDeviceSynchronize();
   endclock = clock();
   total_time = compute_time(startclock,endclock);
   printf(" done: elapsed time = %7.4f;",total_time);

   // moving the result to the RAM, and printing one of the sums
   cudaMemcpy(z,zgpu,n*sizeof(int),cudaMemcpyDeviceToHost);
   k = rand()%n;
   printf(" verification: x = %d, y = %d, z = %d (index modulo is %ld)\n",x[k],y[k],z[k],k%4);

   // cleaning up (RAM and global memory)
   for (size_t i = 0; i < n; i++)  z[i] = 0;
   cleanup<<<nblocks,nthreads>>>(n,zgpu);
   cudaDeviceSynchronize();

   // invoking the fourth kernel (without divergence)
   printf("Computations on GPU (non-divergent) ... ");
   startclock = clock();
   vectorsum_without_divergence<<<nblocks,nthreads>>>(n,xgpu,ygpu,zgpu);
   cudaDeviceSynchronize();
   endclock = clock();
   total_time = compute_time(startclock,endclock);
   printf(" done: elapsed time = %7.4f;",total_time);

   // moving the result to the RAM, and printing one of the sums
   cudaMemcpy(z,zgpu,n*sizeof(int),cudaMemcpyDeviceToHost);
   k = rand()%n;
   printf(" verification: x = %d, y = %d, z = %d (index modulo is %ld)\n",x[k],y[k],z[k],k%4);

   // memory allocation for the partial sums on the RAM and on the global memory
   partial = (int*)calloc(total_threads,sizeof(int));
   cudaMalloc((void**)&pgpu,total_threads*sizeof(int));

   // Problem3: the sum of all vector elements
   printf("Problem3: computing the sum of all z elements\n");

   // computing the sum in sequential (for confirming GPU result)
   int sum_cpu = 0;
   for (size_t i = 0 ; i < n; i++)  sum_cpu = sum_cpu + z[i];

   // invoking the fifth kernel (partial reduction)
   printf("Computations on GPU (partial reduction) ... ");
   startclock = clock();
   vectorsum_partial_reduction<<<nblocks,nthreads>>>(n,zgpu,pgpu);
   cudaDeviceSynchronize();
   endclock = clock();
   gpu_time = compute_time(startclock,endclock);

   // moving the partial sums to the RAM
   startclock = endclock;
   cudaMemcpy(partial,pgpu,total_threads*sizeof(int),cudaMemcpyDeviceToHost);
   endclock = clock();
   float transfer_time = compute_time(startclock,endclock);

   // finalizing the computation
   startclock = endclock;
   int sum_gpu = 0;
   for (size_t i = 0; i < total_threads; i++)  sum_gpu = sum_gpu + partial[i];
   endclock = clock();
   cpu_time = compute_time(startclock,endclock);
   printf(" done: elapsed time = %7.4f (gpu) %7.4f (transfer) %7.4f (cpu);",gpu_time,transfer_time,cpu_time);
   printf(" verification: ");
   if (sum_gpu == sum_cpu)  printf("OK\n");

   // cleaning up (RAM and global memory)
   for (size_t i = 0; i < total_threads; i++)  partial[i] = 0;
   cleanup<<<nblocks,nthreads>>>(total_threads,pgpu);
   cudaDeviceSynchronize();

   // invoking the sixth kernel (reduction with shared memory)
   printf("Computations on GPU (reduction with shared memory) ... ");
   startclock = clock();
   vectorsum_shared_reduction<<<nblocks,nthreads,nthreads*sizeof(int)>>>(n,zgpu,pgpu);
   cudaDeviceSynchronize();
   endclock = clock();
   gpu_time = compute_time(startclock,endclock);

   // moving the partial sums to the RAM
   startclock = endclock;
   cudaMemcpy(partial,pgpu,nblocks*sizeof(int),cudaMemcpyDeviceToHost);
   endclock = clock();
   transfer_time = compute_time(startclock,endclock);

   // finalizing the computation
   startclock = endclock;
   sum_gpu = 0;
   for (size_t i = 0; i < nblocks; i++)  sum_gpu = sum_gpu + partial[i];
   endclock = clock();
   total_time = compute_time(startclock,endclock);
   cpu_time = compute_time(startclock,endclock);
   printf(" done: elapsed time = %7.4f (gpu) %7.4f (transfer) %7.4f (cpu);",gpu_time,transfer_time,cpu_time);
   printf(" verification: ");
   if (sum_gpu == sum_cpu)  printf("OK\n");

   // freeing memory
   cudaFree(xgpu);
   cudaFree(ygpu);
   cudaFree(zgpu);
   cudaFree(pgpu);
   free(x);
   free(y);
   free(z);
   free(partial);

   // ending
   return 0;
};

