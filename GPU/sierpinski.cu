
/* Sierpinski fractal on GPU
 *
 * images are represented by boolean matrices, all matrices are squared
 *
 * the grid/image correspondance is: one pixel per thread for each performed operation
 *
 * the code uses CUDA Streams
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

// kernel for generating a 'white' image
__global__ void white_image(bool *image)
{
   size_t n = blockDim.x*gridDim.x;
   size_t idx = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t idy = (blockIdx.y*blockDim.y) + threadIdx.y;
   size_t k = index(idx,idy,n);
   image[k] = false;
};

// kernel for generating a 'white' image
__global__ void black_image(bool *image)
{
   size_t n = blockDim.x*gridDim.x;
   size_t idx = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t idy = (blockIdx.y*blockDim.y) + threadIdx.y;
   size_t k = index(idx,idy,n);
   image[k] = true;
};

// kernel for shrinking the rows and the columns of the image
// -> n is the original size (n x n); new size is (n/2 x n/2)
// -> one element of the shrunk matrix is assigned to each thread
__global__ void shrink_image(size_t N,bool *original,bool *shrunk)
{
   size_t n = blockDim.x*gridDim.x;
   size_t idx = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t idy = (blockIdx.y*blockDim.y) + threadIdx.y;
   size_t k = index(idx,idy,n);
   size_t h = index(idx << 1,idy << 1,N);
   shrunk[k] = original[h] || original[h + 1] || original[N + h] || original[N + h + 1];
};

// kernel for inserting a smaller image into a larger one
// -> (x,y) is the first pixel where to begin the insertion
// -> each thread takes care of only one element
__global__ void insert_image(size_t N,bool *larger,bool *smaller,size_t x,size_t y)
{
   size_t n = blockDim.x*gridDim.x;
   size_t idx = (blockIdx.x*blockDim.x) + threadIdx.x;
   size_t idy = (blockIdx.y*blockDim.y) + threadIdx.y;
   size_t k = index(idx,idy,n);
   size_t K = index(x + idx,y + idy,N);
   larger[K] = smaller[k];
};

// printing the matrix (ie, the image) on CPU
void print_image(size_t n,bool *image)
{
   assert(n > 0UL);
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < n; j++)
      {
         size_t k = index(i,j,n);
         if (image[k])  printf("x");  else  printf(" ");
      };
      printf("\n");
   };
};

// main
int main(int argc,char *argv[])
{
   size_t ITERATIONS = 5;
   size_t nblocks = 16;
   dim3 blocks(nblocks,nblocks);  // 16 x 16
   size_t nthreads = 8;
   dim3 threads(nthreads,nthreads);  // 8 x 8
   dim3 half_threads(nthreads/2,nthreads/2);  // 4 x 4
   size_t n = nblocks*nthreads;  // 128
   size_t nsquared = n*n;  // 16384 (image is 128 x 128)
   bool *sierpinski;  // RAM
   bool *full_image;  // global memory
   bool *image_part;  // global memory
   cudaStream_t higher;  // streams
   cudaStream_t lower_left;
   cudaStream_t lower_right;

   // welcome message
   fprintf(stderr,"Sierpinski on GPU with CUDA\n");
   fprintf(stderr,"Two-dimensional thread grid structures:\n");
   fprintf(stderr,"* Main grid [blocks (%lu,%lu), threads (%lu,%lu)]\n",nblocks,nblocks,nthreads,nthreads);
   fprintf(stderr,"* Shrunk grid [blocks (%lu,%lu), threads (%lu,%lu)]\n",nblocks,nblocks,nthreads/2,nthreads/2);

   // allocating memory on GPU
   cudaMalloc((void**)&full_image,nsquared*sizeof(bool));
   cudaMalloc((void**)&image_part,nsquared*sizeof(bool));

   // preparing the image on GPU
   black_image<<<blocks,threads>>>(full_image);
   cudaDeviceSynchronize();

   // streams initialization
   cudaStreamCreate(&higher);
   cudaStreamCreate(&lower_left);
   cudaStreamCreate(&lower_right);

   while (ITERATIONS > 0)
   {
      // cleaning up the image part
      white_image<<<blocks,threads>>>(image_part);
      cudaDeviceSynchronize();

      // shrinking the current image
      shrink_image<<<blocks,half_threads>>>(n,full_image,image_part);
      cudaDeviceSynchronize();

      // cleaning up the main image
      white_image<<<blocks,threads>>>(full_image);
      cudaDeviceSynchronize();

      // reassembling on GPU by Sierpinski fractal rule
      insert_image<<<blocks,half_threads,0,higher>>>(n,full_image,image_part,0,n/4);
      insert_image<<<blocks,half_threads,0,lower_left>>>(n,full_image,image_part,n/2,0);
      insert_image<<<blocks,half_threads,0,lower_right>>>(n,full_image,image_part,n/2,n/2);
      cudaDeviceSynchronize();

      // counting the iterations
      ITERATIONS--;
   };

   // destroying the streams
   cudaStreamDestroy(higher);
   cudaStreamDestroy(lower_left);
   cudaStreamDestroy(lower_right);

   // moving the final image to the RAM
   sierpinski = (bool*)calloc(nsquared,sizeof(bool));
   cudaMemcpy(sierpinski,full_image,nsquared*sizeof(bool),cudaMemcpyDeviceToHost);

   // printing Sierpinski fractal
   print_image(n,sierpinski);

   // freeing memory
   cudaFree(image_part);
   cudaFree(full_image);
   free(sierpinski);

   // ending
   return 0;
};

