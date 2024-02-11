
/*
 * Vector Sum on GPU with CUDA
 *
 * execution on Nuvolos with GPU Tesla T4:
 *
 * Vector sum with CUDA
 * nblocks = 64, nthreads = 128, nchunk = 100000
 * hence vector size is 819200000
 * Computing the vector sum on CPU ...  done: elapsed time =  2.3357; verification: 3 + 3 = 6 
 * Computing the vector sum on GPU (non-coalesced) ...  done: elapsed time =  0.1816; verification: 8 + 0 = 8 
 * Computing the vector sum on GPU (coalesced) ...  done: elapsed time =  0.0820; verification: 7 + 9 = 16 
 *
 * last update: February 11th, 2024
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// the kernel for the GPU (with non coalesced global memory access)
__global__ void vectorsum_noncoalesced(size_t nchunk,int *xgpu,int *ygpu,int *zgpu)
{
   size_t i;
   unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
   for (i = id*nchunk; i < (id+1)*nchunk; i++)  zgpu[i] = xgpu[i] + ygpu[i];
};

// the kernel for the GPU (with coalesced global memory access)
__global__ void vectorsum_coalesced(size_t n,int *xgpu,int *ygpu,int *zgpu)
{
   size_t i;
   unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
   unsigned int nthreads = gridDim.x*blockDim.x;
   for (i = id; i < n; i = i + nthreads)  zgpu[i] = xgpu[i] + ygpu[i];
};

// computing clock time
float compute_time(time_t start,time_t end)
{
   return ((float)((int)end - (int)start))/CLOCKS_PER_SEC;
};

// main
int main(int argc,char *argv[])
{
   unsigned int i;
   size_t warp_size = 32;
   size_t warps_per_block = 4;
   size_t nthreads = warp_size*warps_per_block;
   size_t nblocks = 8;
   size_t nchunk = 10000;
   size_t n = nchunk*nblocks*nthreads;
   int *x,*xgpu;
   int *y,*ygpu;
   int *z,*zgpu;
   time_t startclock,endclock;
   float time;

   // welcome message
   printf("Vector sum with CUDA\n");
   printf("nblocks = %lu, nthreads = %lu, nchunk = %lu\n",nblocks,nthreads,nchunk);
   printf("hence vector size is %lu\n",n);

   // memory allocation on CPU
   x = (int*)calloc(n,sizeof(int));
   y = (int*)calloc(n,sizeof(int));
   z = (int*)calloc(n,sizeof(int));

   // initializing data on CPU
   for (i = 0; i < n; i++)  x[i] = rand()%10;
   for (i = 0; i < n; i++)  y[i] = rand()%10;

   // computing the vector sum on CPU
   printf("Computing the vector sum on CPU ... ");
   startclock = clock();
   for (i = 0; i < n; i++)  z[i] = x[i] + y[i];
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf(" done: elapsed time = %7.4f; verification: %d + %d = %d \n",time,x[0],y[0],z[0]);

   // cleaning up the solution vector
   for (i = 0; i < n; i++)  z[i] = 0.0;

   // memory allocation on GPU
   cudaMalloc((void**)&xgpu,n*sizeof(int));
   cudaMalloc((void**)&ygpu,n*sizeof(int));
   cudaMalloc((void**)&zgpu,n*sizeof(int));

   // moving the vectors x and y on the global memory
   cudaMemcpy(xgpu,x,n*sizeof(int),cudaMemcpyHostToDevice);
   cudaMemcpy(ygpu,y,n*sizeof(int),cudaMemcpyHostToDevice);

   // invoking the first kernel (non-coalesced access)
   printf("Computing the vector sum on GPU (non-coalesced) ... ");
   startclock = clock();
   vectorsum_noncoalesced<<<nblocks,nthreads>>>(nchunk,xgpu,ygpu,zgpu);  // running on GPU!
   cudaDeviceSynchronize();  // synchronization!
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf(" done: elapsed time = %7.4f;",time);

   // moving the result to the RAM
   cudaMemcpy(z,zgpu,n*sizeof(int),cudaMemcpyDeviceToHost);
   printf(" verification: %d + %d = %d \n",x[nchunk+1],y[nchunk+1],z[nchunk+1]);

   // cleaning up the solution vector
   for (i = 0; i < n; i++)  z[i] = 0.0;

   // invoking the second kernel (coalesced access)
   printf("Computing the vector sum on GPU (coalesced) ... ");
   startclock = clock();
   vectorsum_coalesced<<<nblocks,nthreads>>>(n,xgpu,ygpu,zgpu);  // running on GPU!
   cudaDeviceSynchronize();  // synchronization!
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf(" done: elapsed time = %7.4f;",time);

   // moving the result to the RAM
   cudaMemcpy(z,zgpu,n*sizeof(int),cudaMemcpyDeviceToHost);
   printf(" verification: %d + %d = %d \n",x[2*nchunk+2],y[2*nchunk+2],z[2*nchunk+2]);

   // freeing memory
   cudaFree(x);
   cudaFree(y);
   cudaFree(z);
   free(x);
   free(y);
   free(z);

   // ending
   return 0;
};

