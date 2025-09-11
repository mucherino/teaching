
/*
 * Code for loading the thread identifiers with CUDA on GPU
 *
 * This code works ONLY for one-dimensional topologies
 *
 * AM
 */
 
#include <stdio.h>
#include <stdlib.h>

// kernel
__global__ void identifiers(unsigned int *id_in_block,unsigned int *block_id)
{
   unsigned int id = (blockDim.x*blockIdx.x) + threadIdx.x;
   id_in_block[id] = threadIdx.x;
   block_id[id] = blockIdx.x;
};

// main
int main()
{
   unsigned int i;
   size_t warp_size = 32;
   size_t warps_per_block = 4;
   size_t nthreads_per_block = warp_size*warps_per_block;
   size_t nblocks = 8;
   size_t total_threads = nblocks*nthreads_per_block;
   unsigned int *id_in_block,*id_in_block_gpu;
   unsigned int *block_id,*block_id_gpu;

   // welcome message
   printf("Loading the identifiers of each thread on the GPU\n");
   printf("number of blocks is %lu; number of threads per block is %lu\n",nblocks,nthreads_per_block);
   printf("allocating memory (on both CPU and GPU) ...\n");

   // memory allocation on CPU
   id_in_block = (unsigned int*)calloc(total_threads,sizeof(unsigned int));
   block_id = (unsigned int*)calloc(total_threads,sizeof(unsigned int));

   // memory allocation on GPU
   cudaMalloc((void**)&id_in_block_gpu,total_threads*sizeof(unsigned int));
   cudaMalloc((void**)&block_id_gpu,total_threads*sizeof(unsigned int));

   // calling the kernel
   printf("calling the kernel ...\n");
   identifiers<<<nblocks,nthreads_per_block>>>(id_in_block_gpu,block_id_gpu);
   cudaDeviceSynchronize();

   // retrieving the data (all identifiers)
   printf("transferring the data...\n");
   cudaMemcpy(id_in_block,id_in_block_gpu,total_threads*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   cudaMemcpy(block_id,block_id_gpu,total_threads*sizeof(unsigned int),cudaMemcpyDeviceToHost);

   // freeing memory on the GPU
   printf("freeing memory on the GPU...\n");
   cudaFree(id_in_block_gpu);
   cudaFree(block_id_gpu);

   // printing the identifiers
   for (i = 0; i < total_threads; i++)  printf("%u) id in the block is %u; id of the block is %u\n",i,id_in_block[i],block_id[i]);

   // ending
   printf("ending ...\n");
   free(id_in_block);
   free(block_id);
   return 0;
};

