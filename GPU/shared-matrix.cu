
/* Shared matrix
 * 
 * We execute several matrix-by-matrix operations in CUDA, with 
 * the aim of showing the importance of shared memory on GPU.
 *
 * Basically, we compute C = p*A*B, where p is a positive integer.
 * Of course, C can be computed by multiplying A*B by the constant p.
 * Instead, we will compute it with the following approach:
 *
 *    C = (A*B) + (A*B) + ... + (A*B)  [p times]
 *
 * where A*B is recalculated each time. This is of course a 
 * tremendously inefficient approach. Notice however that in real
 * applications the two matrices A and B can vary for each multiplication;
 * instead we will use always the same two randomly generated matrices 
 * in order to avoid to introduce useless complexity in our code. 
 * Our focus is in fact on the CUDA syntax and benefits in using 
 * the shared memory.
 *
 * Part of the code is "inherited" from a previous example. 
 * Again, we suppose that each thread will take care of only one
 * element of resulting matrix C
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>

// converting from (i,j) indices to a unique k index
// -> we suppose that the elements of the matrix are
//    stored row by row
__host__ __device__ size_t index(size_t i,size_t j,size_t n)
{
   return i*n + j;
};

// sequential version of C = C + A*B
void sequential(size_t n,float *A,float *B,float *C)
{
   assert(n > 0UL);
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < n; j++)
      {
         size_t ind = index(i,j,n);
         for (size_t k = 0; k < n; k++)
         {
            C[ind] = C[ind] + A[index(i,k,n)]*B[index(k,j,n)];
         };
      };
   };
};

// sequential version of C = p*A*B
void multi_sequential(size_t n,float *A,float *B,float *C,size_t p)
{
   assert(n > 0UL);
   assert(p > 0UL);
   for (size_t k = 0; k < n*n; k++)  C[k] = 0.0f;
   for (size_t ip = 0; ip < p; ip++)  sequential(n,A,B,C);
};

// CUDA version of C = 0
__device__ void zero_matrix(float *CG)
{
   size_t n = blockDim.x*gridDim.x;
   size_t idx = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t idy = (blockIdx.y*blockDim.y) + threadIdx.y;
   CG[index(idx,idy,n)] = 0.0f;
};

// CUDA version of C = C + A*B without shared memory
__device__ void mbm_no_shared(float *AG,float *BG,float *CG)
{
   size_t n = blockDim.x*gridDim.x;
   size_t idx = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t idy = (blockIdx.y*blockDim.y) + threadIdx.y;
   size_t ind = index(idx,idy,n);

   for (size_t k = 0; k < n; k++)
   {
      CG[ind] = CG[ind] + AG[index(idx,k,n)]*BG[index(k,idy,n)];
   };
};

// CUDA version of C = p*A*B without shared memory
__global__ void multi_no_shared(float *AG,float *BG,float *CG,size_t p)
{
   zero_matrix(CG);
   for (size_t ip = 0; ip < p; ip++)  mbm_no_shared(AG,BG,CG);
};

// matrix-by-matrix with shared memory on GPU
// -> the result is cumulated in the shared memory!
__device__ void mbm_shared(float *AG,float *BG,float *shared)
{
   size_t n = blockDim.x*gridDim.x;
   size_t idx = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t idy = (blockIdx.y*blockDim.y) + threadIdx.y;
   size_t myelement = index(threadIdx.x,threadIdx.y,blockDim.x);

   for (size_t k = 0; k < n; k++)
   {
      shared[myelement] = shared[myelement] + AG[index(idx,k,n)]*BG[index(k,idy,n)];
   };
   __syncthreads();
};

// retrieving the result from the shared memory
__device__ void retrieve_matrix(float *shared,float *CG)
{
   size_t n = blockDim.x*gridDim.x;
   size_t idx = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t idy = (blockIdx.y*blockDim.y) + threadIdx.y;
   CG[index(idx,idy,n)] = shared[index(threadIdx.x,threadIdx.y,blockDim.x)];
   __syncthreads();
};

// CUDA version of C = p*A*B with shared memory
__global__ void multi_shared(float *AG,float *BG,float *CG,size_t p)
{
   // initializing the shared memory in each thread block
   extern __shared__ float shared_mem[];
   size_t myelement = index(threadIdx.x,threadIdx.y,blockDim.x);
   shared_mem[myelement] = 0.0f;
   __syncthreads();

   // performing the computations
   for (size_t ip = 0; ip < p; ip++)  mbm_shared(AG,BG,shared_mem);

   // extracting the results from the shared memory
   retrieve_matrix(shared_mem,CG);
};

// frand (single-precision random number in [0,1])
float frand()
{
   return ((float)rand() / RAND_MAX);
};

// computing CPU clock time
float compute_time(time_t start,time_t end)
{
   return ((float)((int)end - (int)start))/CLOCKS_PER_SEC;
};

// main
int main(int argc,char *argv[])
{
   size_t p = 100;
   size_t nblocks = 32;   // nblocks^2 = 1024
   size_t nthreads = 8;  // nthreads^2 = 64
   size_t n = nblocks*nthreads;  // n^2 = 65536
   float *A,*B,*C;  // pointers to the RAM
   float *AG,*BG,*CG;  // pointers to the global memory
   time_t startclock,endclock;
   float time;

   // welcome message
   fprintf(stderr,"p*(Matrix-by-Matrix) on GPU with CUDA\n");
   fprintf(stderr,"Comparing versions : sequential, CUDA w/out shared memory, and CUDA with shared memory\n");
   fprintf(stderr,"Two-dimensional thread grid structure: ");
   fprintf(stderr,"[blocks (%lu,%lu), threads (%lu,%lu)]\n",nblocks,nblocks,nthreads,nthreads);
   fprintf(stderr,"Setting up memory space on RAM ... ");

   // memory allocation on RAM
   size_t nsquared = n*n;
   A = (float*)calloc(nsquared,sizeof(float));
   B = (float*)calloc(nsquared,sizeof(float));
   C = (float*)calloc(nsquared,sizeof(float));

   // making sure that C is set up to the zero matrix
   for (size_t k = 0; k < nsquared; k++)  C[k] = 0.0f;
   endclock = clock();

   // random initialization of matrices A and B
   for (size_t k = 0; k < nsquared; k++)  A[k] = frand();
   for (size_t k = 0; k < nsquared; k++)  B[k] = frand();
   fprintf(stderr,"done\n");

   // running the sequential version on CPU
   fprintf(stderr,"Sequential version ... ");
   startclock = clock();
   multi_sequential(n,A,B,C,p);
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // allocating memory on GPU global memory
   fprintf(stderr,"Memory allocation and memory transfer ... ");
   cudaMalloc((void**)&AG,nsquared*sizeof(float));
   cudaMalloc((void**)&BG,nsquared*sizeof(float));
   cudaMalloc((void**)&CG,nsquared*sizeof(float));
   cudaMemcpy(AG,A,nsquared*sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpy(BG,B,nsquared*sizeof(float),cudaMemcpyHostToDevice);
   fprintf(stderr,"done\n");

   // running the CUDA version without shared memory
   fprintf(stderr,"CUDA version without shared memory ... ");
   dim3 blocks(nblocks,nblocks);
   dim3 threads(nthreads,nthreads);
   startclock = clock();
   multi_no_shared<<<blocks,threads>>>(AG,BG,CG,p);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // running the CUDA version with shared memory
   fprintf(stderr,"CUDA version with shared memory ... ");
   startclock = clock();
   multi_shared<<<blocks,threads,nthreads*nthreads*sizeof(float)>>>(AG,BG,CG,p);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // freeing memory
   cudaFree(AG);
   cudaFree(BG);
   cudaFree(CG);
   free(A);
   free(B);
   free(C);

   // ending
   return 0;
};

