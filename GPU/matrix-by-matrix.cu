
/* matrix-by-matrix in CUDA
 *
 * We suppose that each thread will take care of only one
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

// sequential version
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

// setting to zero all elements of a given matrix (on GPU)
__global__ void zero_matrix(float *M)
{
   size_t n = blockDim.x*gridDim.x;
   size_t idx = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t idy = (blockIdx.y*blockDim.y) + threadIdx.y;
   M[index(idx,idy,n)] = 0.0f;
};

// matrix-by-matrix GPU version
__global__ void matrix_by_matrix(float *AG,float *BG,float *CG)
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
   size_t nblocks = 32;   // nblocks^2 = 1024
   size_t nthreads = 8;  // nthreads^2 = 64
   size_t n = nblocks*nthreads;  // n^2 = 65536
   float *A,*B,*C;  // pointers to the RAM
   float *AG,*BG,*CG;  // pointers to the global memory
   float eps = 1.e-4;
   time_t startclock,endclock;
   float time;

   // welcome message
   fprintf(stderr,"Matrix-by-Matrix on GPU with CUDA\n");
   fprintf(stderr,"Two-dimensional thread grid structure: ");
   fprintf(stderr,"[blocks (%lu,%lu), threads (%lu,%lu)]\n",nblocks,nblocks,nthreads,nthreads);

   // memory allocation on RAM
   fprintf(stderr,"Memory allocation on RAM ... ");
   size_t nsquared = n*n;
   startclock = clock();
   A = (float*)calloc(nsquared,sizeof(float));
   B = (float*)calloc(nsquared,sizeof(float));
   C = (float*)calloc(nsquared,sizeof(float));
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // making sure that C is set up to the zero matrix
   fprintf(stderr,"Setting up to zero all elements of C matrix on RAM ... ");
   startclock = clock();
   for (size_t k = 0; k < nsquared; k++)  C[k] = 0.0f;
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // random initialization of matrices A and B
   fprintf(stderr,"Random generation of matrices A and B ... ");
   startclock = clock();
   for (size_t k = 0; k < nsquared; k++)  A[k] = frand();
   for (size_t k = 0; k < nsquared; k++)  B[k] = frand();
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // running the sequential version on CPU
   fprintf(stderr,"Sequential version ... ");
   startclock = clock();
   sequential(n,A,B,C);
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // allocating memory on GPU global memory
   fprintf(stderr,"Memory allocation on GPU ... ");
   startclock = clock();
   cudaMalloc((void**)&AG,nsquared*sizeof(float));
   cudaMalloc((void**)&BG,nsquared*sizeof(float));
   cudaMalloc((void**)&CG,nsquared*sizeof(float));
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // making sure that C is set up to the zero matrix
   fprintf(stderr,"Setting up to zero all elements of C matrix on GPU ... ");
   dim3 blocks(nblocks,nblocks);
   dim3 threads(nthreads,nthreads);
   startclock = clock();
   zero_matrix<<<blocks,threads>>>(CG);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // transferring the matrices A and B to the GPU global memory
   fprintf(stderr,"Transferring A and B to the global memory ... ");
   startclock = clock();
   cudaMemcpy(AG,A,nsquared*sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpy(BG,B,nsquared*sizeof(float),cudaMemcpyHostToDevice);
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // running the main GPU kernel
   fprintf(stderr,"CUDA version of matrix-by-matrix ... ");
   startclock = clock();
   matrix_by_matrix<<<blocks,threads>>>(AG,BG,CG);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // transferring the resulting matrix back to the RAM (in A)
   fprintf(stderr,"Transferring the result to the RAM ... ");
   startclock = clock();
   cudaMemcpy(A,CG,nsquared*sizeof(float),cudaMemcpyDeviceToHost);
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // verifying the results
   fprintf(stderr,"Verifying the results ... ");
   startclock = clock();
   bool OK = true;
   for (size_t k = 0; k < nsquared && OK; k++)  if (fabs(A[k] - C[k]) > eps)  OK = false;
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds, results are ",time);
   if (!OK)  fprintf(stderr,"NOT ");
   fprintf(stderr,"OK\n");

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

