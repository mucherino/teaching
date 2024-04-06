
/* 
 * Matrix transpose in CUDA
 *
 * For this exercise, we will explore the possibility to use
 * a 2-dimensional topology for the grid of threads in the GPU.
 * The 'chunk' of matrix elements assigned to each threads will
 * be 2-dimensional as well.
 *
 * Matrix transposition on GPU with CUDA
 * Two-dimensional thread grid structure: [blocks (8,8), threads (8,8), nchunks (128,128)]
 * Sequential implementation ... done in 2.07977 seconds
 * GPU implementation ... done in 0.019636 seconds
 * The two implementation gave coherent results
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

// converting from (i,j) indices to a unique k index
// -> we suppose that the elements of the matrix are stored row by row
__host__ __device__ size_t index(size_t i,size_t j,size_t n)
{
   return i*n + j;
};

// printing (use only for small matrices!)
void print(size_t n,size_t m,float *A)
{
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         printf(" %5.3f",A[index(i,j,n)]);
      };
      printf("\n");
   };
};

// matrix transpose in sequential
void transpose_seq(size_t n,size_t m,float *A,float *B)
{
   // TO BE COMPLETED
};

// matrix transpose on GPU
__global__ void transpose(size_t n,size_t m,float *A,float *B)
{
   unsigned int N = ?  // why do we need another size N here?
   unsigned int M = ?  // the question is to undestand what n and m actually represent here
   unsigned int idx = ?
   unsigned int idy = ?
   // TO BE COMPLETED
};

// computing CPU time
float compute_time(time_t start,time_t end)
{
   return ((float)((int)end - (int)start))/CLOCKS_PER_SEC;
};

// main
int main(int argc,char *argv[])
{
   size_t nblocks = 8;   // nblocks^2 = 64
   size_t nthreads = 8;  // nthreads^2 = 64
   size_t nchunks = 16;  // nchunks^2 = 256
   size_t n = nblocks*nthreads*nchunks;
   size_t m = n;
   float *A,*B,*AG,*BG;
   time_t startclock,endclock;
   float eps = 0.0001;
   float time;

   // welcome message
   fprintf(stderr,"Matrix transposition on GPU with CUDA\n");
   fprintf(stderr,"Two-dimensional thread grid structure: ");
   fprintf(stderr,"[blocks (%lu,%lu), threads (%lu,%lu), nchunks (%lu,%lu)]\n",nblocks,nblocks,nthreads,nthreads,nchunks,nchunks);

   // memory allocation on RAM
   A = (float*)calloc(n*m,sizeof(float));
   B = (float*)calloc(n*m,sizeof(float));

   // filling up the matrix A with float values
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         // TO BE COMPLETED
      };
   };

   // transposing on CPU
   fprintf(stderr,"Sequential implementation ... ");
   startclock = clock();
   transpose_seq(n,m,A,B);
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // allocating memory on GPU global memory
   cudaMalloc((void**)&AG,n*m*sizeof(float));
   cudaMalloc((void**)&BG,n*m*sizeof(float));

   // transferring data to the global memory ... TO BE COMPLETED

   // invoking the kernel
   fprintf(stderr,"GPU implementation ... ");
   dim3 blocks(nblocks,nblocks);
   dim3 threads(nthreads,nthreads);
   startclock = clock();
   // kernel is invoked here  ... TO BE COMPLETED
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);
   fprintf(stderr,"done in %g seconds\n",time);

   // transferring the result back from global memory ... TO BE COMPLETED

   // verifying whether A and B are now identical
   bool coherent = true;
   fprintf(stderr,"The two implementation gave ");
   for (size_t i = 0; i < n && coherent; i++)
   {
      for (size_t j = 0; j < m && coherent; j++)
      {
         coherent = (abs(A[index(i,j,n)] - B[index(i,j,n)]) < eps);
      };
   };
   if (!coherent) fprintf(stderr,"NON-");
   fprintf(stderr,"coherent results\n");

   /* FOR DEBUGGING (consider small matrices)
   if (!coherent)
   {
      print(n,m,A);
      printf("\n");
      print(n,m,B);
   }; */

   // freeing memory
   cudaFree(AG);
   cudaFree(BG);
   free(A);
   free(B);

   // ending
   return 0;
};

