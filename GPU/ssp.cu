
/* 
 * Recursive algorithm for the Subset Sum Problem (SSP) on GPU
 *
 * We are only interested in verifying whether solutions exist;
 * we do not intend to output the list of integers forming the 
 * found solution (i.e. the subsets).
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* GPU kernel */

// every thread computes the sum of the subset encoded by the binary version
// of its own identifier ;-)
__global__ void ssp_on_gpu(size_t n,unsigned long *set_gpu,unsigned long *sum_gpu)
{
   unsigned long sum = 0UL;
   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t copy = id;
   for (size_t i = 0; i < n; i++)
   {
      sum = sum + (id%2)*set_gpu[i];
      id = id >> 1;
   };
   sum_gpu[copy] = sum;
};

// recursive algorithm in sequential
void recursive(size_t i,size_t n,unsigned long target,unsigned long partial,unsigned long *set,size_t *count)
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

// computing CPU clock time
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
   size_t nblocks = 2048;  // idem
   size_t total_threads = nblocks*nthreads;
   size_t n = 0UL;
   unsigned long count,partial;
   unsigned long target,*set,*sum;
   unsigned long *set_gpu,*sum_gpu;
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
   set = (unsigned long*)calloc(n,sizeof(unsigned long));
   sum = (unsigned long*)calloc(total_threads,sizeof(unsigned long));

   // generating a simple SSP instance
   target = 0UL;
   for (size_t i = 0; i < n; i++)
   {
      set[i] = rand()%10;  // integers between 0 and 9
      if (rand()%2)  target = target + set[i];
   };

   // counting the number of solutions in sequential
   count = 0UL;
   partial = 0UL;
   startclock = clock();
   recursive(0,n,target,partial,set,&count);  // recursive calls
   endclock = clock();
   time = compute_time(startclock,endclock);

   // printing the solution obtained in sequential
   printf("Recursive algorithm says that %lu subsets correspond to the SSP target (time %16.13lf)\n",count,time);

   // memory allocation on GPU
   cudaMalloc((void**)&set_gpu,n*sizeof(unsigned long));
   cudaMalloc((void**)&sum_gpu,total_threads*sizeof(unsigned long));

   // moving the set of integers on GPU global memory
   cudaMemcpy(set_gpu,set,n*sizeof(unsigned long),cudaMemcpyHostToDevice);

   // launching the kernel
   startclock = clock();
   ssp_on_gpu<<<nblocks,nthreads>>>(n,set_gpu,sum_gpu);
   cudaDeviceSynchronize();

   // retrieving the computed sums
   cudaMemcpy(sum,sum_gpu,total_threads*sizeof(unsigned long),cudaMemcpyDeviceToHost);

   // counting the number of sums that are equal to the target
   count = 0UL;
   for (size_t i = 0; i < total_threads; i++)
   {
      if (sum[i] == target)  count++;
   };

   // printing the result obtained with the help of the GPU
   endclock = clock();
   time = compute_time(startclock,endclock);
   printf("GPU parallel approach says that %lu subsets correspond to the SSP target (time %16.13f)\n",count,time);

   // freing memory
   cudaFree(set_gpu);
   cudaFree(sum_gpu);
   free(set);
   free(sum);

   // ending
   return 0;
};

