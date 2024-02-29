
/* 
 * The recursive algorithm for the Subset Sum Problem (SSP) on GPU
 *
 * We are only interested in verifying whether solutions exist;
 * we do not intend to output the list of integers forming the 
 * found solution (i.e. the subsets)
 *
 * Execution on Nuvolos with a Tesla T4 GPU:
 *
 * Recursive algorithm for SSP with CUDA on GPU
 * nblocks = 1024, nthreads = 128
 * SSP size = 17, search space size = 131072
 * Recursive algorithm says that 4632 subsets correspond to the SSP target (time  0.0010430000257)
 * GPU parallel approach says that 4632 subsets correspond to the SSP target (time  0.0000450000007)
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* GPU kernel */

// every thread computes the sum of the subset encoded by the binary version
// of its own identifier ;-)

__global__ void ssp_on_gpu(size_t n,int *set_gpu,int *sum_gpu)
{
   int sum = 0;
   unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
   unsigned int copy = id;
   for (size_t i = 0; i < n; i++)
   {
      sum = sum + (id%2)*set_gpu[i];
      id = id >> 1;
   };
   sum_gpu[copy] = sum;
};

// recursive algorithm in sequential
void recursive(size_t i,size_t n,int target,int partial,int *set,int *count)
{
   // are we at the last element yet?
   if (i == n)
   {
      if (partial == target)  (*count)++;
      return;
   };

   // we do not consider the current integer
   recursive(i+1,n,target,partial,set,count);

   // we do consider the current integer
   partial = partial + set[i];
   recursive(i+1,n,target,partial,set,count);
};

// computing CPU time
float compute_time(time_t start,time_t end)
{
   return ((float)((int)end - (int)start))/CLOCKS_PER_SEC;
};

// main
int main(int argc,char *argv[])
{
   size_t warp_size = 32;  // must be power of 2
   size_t warps_per_block = 4;  // idem
   size_t nthreads = warp_size*warps_per_block;
   size_t nblocks = 1024;  // idem
   size_t total_threads = nblocks*nthreads;
   size_t n = 0;
   int count,partial;
   int target,*set,*sum;
   int *set_gpu,*sum_gpu;
   time_t startclock,endclock;
   float time;

   // welcome message
   printf("Recursive algorithm for SSP with CUDA on GPU\n");
   printf("nblocks = %lu, nthreads = %lu\n",nblocks,nthreads);

   // computing SSP size for the given search space (one sum per thread)
   size_t tmp = total_threads;
   while (tmp > 1)
   {
      n++;
      tmp = tmp >> 1;
   };
   printf("SSP size = %lu, search space size = %lu\n",n,total_threads);

   // memory allocation for the CPU
   set = (int*)calloc(n,sizeof(int));
   sum = (int*)calloc(total_threads,sizeof(int));

   // generating a simple SSP instance
   target = 0;
   for (size_t i = 0; i < n; i++)
   {
      set[i] = rand()%10;  // integers between 0 and 9
      if (rand()%2)  target = target + set[i];
   };

   // counting the number of solutions in sequential
   count = 0;
   partial = 0;
   startclock = clock();
   recursive(0,n,target,partial,set,&count);  // recursive calls
   endclock = clock();
   time = compute_time(startclock,endclock);

   // printing the solution obtained in sequential
   printf("Recursive algorithm says that %d subsets correspond to the SSP target (time %16.13lf)\n",count,time);

   // memory allocation on GPU
   cudaMalloc((void**)&set_gpu,n*sizeof(int));
   cudaMalloc((void**)&sum_gpu,total_threads*sizeof(int));

   // moving the set of integers on GPU global memory
   cudaMemcpy(set_gpu,set,n*sizeof(int),cudaMemcpyHostToDevice);

   // launching the kernel
   startclock = clock();
   ssp_on_gpu<<<nblocks,nthreads>>>(n,set_gpu,sum_gpu);
   cudaDeviceSynchronize();
   endclock = clock();
   time = compute_time(startclock,endclock);

   // retrieving the computed sums
   cudaMemcpy(sum,sum_gpu,total_threads*sizeof(int),cudaMemcpyDeviceToHost);

   // counting the number of sums that are equal to the target
   count = 0;
   for (size_t i = 0; i < total_threads; i++)
   {
      if (sum[i] == target)  count++;
   };

   // printing the result obtained with the help of the GPU
   printf("GPU parallel approach says that %d subsets correspond to the SSP target (time %16.13f)\n",count,time);

   // freing memory
   cudaFree(set_gpu);
   cudaFree(sum_gpu);
   free(set);
   free(sum);

   // ending
   return 0;
};

