
/* Hands-on lab session on OpenMP for beginners
 *
 * Please follow the instructions given to you when you compile 
 * and run the current version of this code.
 *
 * Compile with: gcc -o executable source.c -fopenmp
 *
 * Student version
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <omp.h>

// global variables
int N = 1000;
int NTHREADS = 4;

// exercise1
char maxchar(int n,char *clist)
{
   exit(0);
}

// exercise2
void matrix_by_vector(int n,double** A,double* b,double* c)
{
   exit(0);
}

// exercise3
bool are_all_elements_equal(int n,double *v)
{
   exit(0);
}

// exercise4
void philosophers()
{
   omp_lock_t *forks = (omp_lock_t*)calloc(NTHREADS,sizeof(omp_lock_t));
   for (int i = 0; i < NTHREADS; i++)  omp_init_lock(forks + i);

   #pragma omp parallel num_threads(NTHREADS)
   {
      unsigned short id = omp_get_thread_num();
      omp_set_lock(forks + id);
      printf("Philo %d has one fork\n",id);
      unsigned short other = (id + 1)%NTHREADS;
      omp_set_lock(forks + other);
      printf("Philo %d has two forks\n",id);
      omp_unset_lock(forks + other);
      omp_unset_lock(forks + id);
      printf("Philo %d finished eating\n",id);
   }

   for (int i = 0; i < NTHREADS; i++)  omp_destroy_lock(forks + i);
   free(forks);
};

// setting up a list of alphabetic characters with given length
char* gen_char_list(int n)
{
   int i;
   char *clist = (char*)calloc(n,sizeof(char));
   srand(999);
   for (i = 0; i < n; i++)  clist[i] = rand()%26 + 97;
   return clist;
}

// printing the char list (for debugging, if needed)
void print_char_list(int n,char *clist)
{
   for (int i = 0; i < n; i++)  printf("%c",clist[i]);
   printf("\n");
}

// initializing a vector of real numbers with zeros
double* init_vector(int n)
{
   int i;
   double *v = (double*)calloc(n,sizeof(double));
   for (int i = 0; i < n; i++)  v[i] = 0.0;
   return v;
}

// constructing a vector of real numbers with random elements
double* gen_vector(int n)
{
   int i;
   double *v = (double*)calloc(n,sizeof(double));
   for (int i = 0; i < n; i++)  v[i] = i%3; 
   return v;
}

// printing a vector a real numbers
void print_vector(int n,double *v)
{
   int i;
   for (i = 0; i < n; i++)  printf("%5.2lf ",v[i]);
   printf("\n");
}

// constructing a squared matrix with random elements
double** gen_matrix(int n)
{
   int i;
   double **M = (double**)calloc(n,sizeof(double*));
   for (i = 0; i < n; i++)  M[i] = gen_vector(n);
   return M;
}

// printing a squared matrix
void print_matrix(int n,double **M)
{
   int i,j;
   for (i = 0; i < n; i++)  print_vector(n,M[i]);
   printf("\n");
}

// freeing the matrix memory
void free_matrix(int n,double** M)
{
   int i;
   for (i = 0; i < n; i++)  free(M[i]);
   free(M);
}

// main
int main()
{
   // variables (in order of use)
   char *clist,max;
   double **A,*b,*c;

   // let's get started
   printf("Hands-on lab session on OpenMP for beginners\n\n");

   // exercise1
   printf("We have an array of alphabetic characters.\n");
   printf("We need to count the number of times each character appears.\n");
   printf("Because we want to know the character that appears more often.\n");
   printf("Please complete the function 'maxchar' to identify this character.\n");
   printf("We need to perform this task in parallel with OpenMP.\n");
   clist = gen_char_list(N);
   max = maxchar(N,clist);
   printf("The character that appears more often is '%c'.\n\n",max);
   free(clist);

   // exercise2
   int n = 4*NTHREADS;
   printf("We now need to multiply a matrix by a vector.\n");
   printf("Again, we need to do it in parallel with OpenMP.\n");
   A = gen_matrix(n);
   b = gen_vector(n);
   c = init_vector(n);
   matrix_by_vector(n,A,b,c);
   printf("The result is : ");
   print_vector(n,c);
   printf("\n");
   free(b);
   free_matrix(n,A);

   // exercise3
   printf("We can remark that all elements have the same value in the result.\n");
   printf("Instead of printing out and manually check, we can ask OpenMP to do that for us!\n");
   printf("The result is ... ");
   if (!are_all_elements_equal(n,c))  printf("NOT");
   printf(" OK\n\n");
   free(c);

   // exercise4
   printf("Finally, it is necessary to fix our last function.\n");
   printf("The function simulates the problem of the dining philosophers.\n");
   philosophers();
   printf("\nCongratulations, you were able to reach the end!\n");

   // we are done
   return 0;
}

