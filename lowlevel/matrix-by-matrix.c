
/*
 * Matrix by matrix with optimized cache memory access
 *
 * all matrices are supposed to be squared matrices
 *
 * our C functions will compute the matrix C such that C = C + A*B
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

// size of squared matrices (used in the main)
static size_t N = 1000;
static size_t NBLOCKS = 20;

// drand (random number in [0,1] with 3 decimal digits)
double drand()
{
   return (double)(rand()%1000) / 1000.0;
};

// computing CPU clock time
float compute_time(time_t start,time_t end)
{
   return ((float)((int)end - (int)start))/CLOCKS_PER_SEC;
};

// generating a squared matrix of size n and with elements equal to 0.0
double** matrix_zeros(size_t n)
{
   assert(n > 0UL);
   double** A = (double**)calloc(n,sizeof(double*));
   for (size_t i = 0; i < n; i++)  A[i] = (double*)calloc(n,sizeof(double)); 
   return A;
};

// generating a random squared matrix of size n with elements in [0,max]
double** matrix_random(size_t n )
{
   double** A = matrix_zeros(n);
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < n; j++)  A[i][j] = drand();
   };
   return A;
};

// printing a squared matrix
void matrix_print(size_t n,double **A)
{
   assert(n > 0UL);
   for (size_t i = 0; i < n; i++)
   {
      if (i == 0) printf("["); else printf(" ");
      for (size_t j = 0; j < n; j++)  printf("%8.4lf ",A[i][j]);
      if (i == n - 1)  printf("]");
      printf("\n");
   };
};

// matrix-by-matrix simple implementation
void mbm_simple(size_t n,double **A,double **B,double **C)
{
   assert(n > 0UL);
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < n; j++)
      {
         for (size_t k = 0; k < n; k++)
         {
            C[i][j] = C[i][j] + A[i][k]*B[k][j];
         };
      };
   };
};

// matrix-by-matrix by squared blocks
void mbm_blocks(size_t nblocks,double **A,double **B,double **C,size_t L,double **blockA,double **blockB,double **blockC)
{
   assert(nblocks > 0UL);
   assert(L > 0UL);

   // performing computations by blocks
   for (size_t iblock = 0; iblock < nblocks; iblock++)
   {
      for (size_t jblock = 0; jblock < nblocks; jblock++)
      {
         for (size_t i = 0; i < L; i++)  blockC[i] = &C[iblock*L + i][jblock*L];

         for (size_t kblock = 0; kblock < nblocks; kblock++)
         {
            for (size_t i = 0; i < L; i++)  blockA[i] = &A[iblock*L + i][kblock*L];
            for (size_t k = 0; k < L; k++)  blockB[k] = &B[kblock*L + k][jblock*L];
            mbm_simple(L,blockA,blockB,blockC);
         };
      };
   };
};

// main
int main()
{
   fprintf(stderr,"Matrix-by-matrix multiplication (N = %lu, NBLOCKS = %lu)\n",N,NBLOCKS);
   assert(N%NBLOCKS == 0UL);

   // preparing the matrices
   srand(999);
   double **A = matrix_random(N);
   double **B = matrix_random(N);
   double **C = matrix_zeros(N);
   if (N < 10)
   {
      printf("A =\n");
      matrix_print(N,A);
      printf("B =\n");
      matrix_print(N,B);
      printf("C =\n");
      matrix_print(N,C);
   };

   // matrix-by-matrix multiplication (simple)
   time_t start,end;
   fprintf(stderr,"Computing C = C + A*B with simple algorithm ... ");
   start = clock();
   mbm_simple(N,A,B,C);
   end = clock();
   fprintf(stderr,"done!\n");

   // printing the result
   float time = compute_time(start,end);
   if (N < 10)
   {
      printf("C =\n");
      matrix_print(N,C);
   };
   fprintf(stderr,"Clock time = %lf\n",time);

   // cleaning up the matrix C
   free(C);
   C = matrix_zeros(N);

   // matrix-by-matrix multiplication (by blocks)
   start = clock();
   fprintf(stderr,"Computing C = C + A*B with block algorithm ... ");
   size_t L = N/NBLOCKS;
   double **blockA = (double**)calloc(L,sizeof(double*));
   double **blockB = (double**)calloc(L,sizeof(double*));
   double **blockC = (double**)calloc(L,sizeof(double*));
   mbm_blocks(NBLOCKS,A,B,C,L,blockA,blockB,blockC);
   free(blockA);
   free(blockB);
   free(blockC);
   end = clock();
   fprintf(stderr,"done!\n");

   // printing the result
   time = compute_time(start,end);
   if (N < 10)
   {
      printf("C =\n");
      matrix_print(N,C);
   };
   fprintf(stderr,"Clock time (including extra memory allocation and freeing) = %lf\n",time);

   // ending
   free(A);
   free(B);
   free(C);
   return 0;
};

