
/* 
 * Recursive algorithm for the Subset Sum Problem (SSP) on GPU
 *
 * We are only interested in verifying how many solutions exist;
 * we do not intend to output the list of integers forming the 
 * found subsets whose element sum up to the target.
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* GPU kernel */

// every thread computes the sum of the subset encoded by the binary
// representation of its own identifier ;-)
// then, it verifies whether the computed sum corresponds to the target
__global__ void ssp_on_gpu(size_t n,unsigned long *set_gpu,bool *is_solution_gpu,unsigned long target)
{
   unsigned long sum = 0UL;
   size_t id = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t copy = id;
   for (size_t i = 0; i < n; i++)
   {
      sum = sum + (id%2)*set_gpu[i];
      id = id >> 1;
   };
   is_solution_gpu[copy] = (sum == target);
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
   bool *is_solution;
   bool *is_solution_gpu;
   unsigned long count,partial;
   unsigned long target,*set;
   unsigned long *set_gpu;
   time_t startclock,endclock;
   float time,gpu_time,transfer_time;

   // welcome message
   printf("Subset Sum Problem (SSP)\n");
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
   is_solution = (bool*)calloc(total_threads,sizeof(bool));

   // generating a simple SSP instance
   target = 0UL;
   for (size_t i = 0; i < n; i++)
   {
      set[i] = rand()%10;  // integers between 0 and 9
      if (rand()%2)  target = target + set[i];
   };

   // initializing the array is_solution
   for (size_t i = 0; i < total_threads; i++)  is_solution[i] = false;

   // counting the number of solutions in sequential
   count = 0UL;
   partial = 0UL;
   startclock = clock();
   recursive(0,n,target,partial,set,&count);  // recursive calls
   endclock = clock();
   time = compute_time(startclock,endclock);

   // printing the solution obtained in sequential
   printf("Recursive algorithm says that %lu subsets correspond to the SSP target\n",count);
   printf("=> time : %8.7f\n",time);

   // memory allocation on GPU
   cudaMalloc((void**)&set_gpu,n*sizeof(unsigned long));
   cudaMalloc((void**)&is_solution_gpu,total_threads*sizeof(bool));

   // moving the set of integers on GPU global memory
   cudaMemcpy(set_gpu,set,n*sizeof(unsigned long),cudaMemcpyHostToDevice);

   // launching the kernel
   startclock = clock();
   ssp_on_gpu<<<nblocks,nthreads>>>(n,set_gpu,is_solution_gpu,target);
   cudaDeviceSynchronize();
   endclock = clock();
   gpu_time = compute_time(startclock,endclock);

   // retrieving the array is_solution
   startclock = clock();
   cudaMemcpy(is_solution,is_solution_gpu,total_threads*sizeof(bool),cudaMemcpyDeviceToHost);
   endclock = clock();
   transfer_time = compute_time(startclock,endclock);

   // summing up the count values received from each thread
   startclock = clock();
   count = 0UL;
   for (size_t i = 0; i < total_threads; i++)  if (is_solution[i])  count++;
   endclock = clock();
   time = compute_time(startclock,endclock);

   // printing the result obtained with the help of the GPU
   printf("GPU parallel approach says that %lu subsets correspond to the SSP target\n",count);
   printf("=> time : %8.7f ",gpu_time + transfer_time + time);
   printf("[%8.7f (GPU), %8.7f (transfer), %8.7f (CPU)]\n",gpu_time,transfer_time,time);

   // freing memory
   cudaFree(set_gpu);
   cudaFree(is_solution_gpu);
   free(set);
   free(is_solution);

   // ending
   return 0;
};

