
/* C code for multiplying integers by 15
 *
 * performance version
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// we will be using the function
unsigned long computation(unsigned long x);

// computing CPU time
float compute_time(time_t start,time_t end)
{
   return ((float)((int)end - (int)start))/CLOCKS_PER_SEC;
};

// constants
static unsigned int N = 10000000;

int main()
{
   // measuring the starting time
   time_t start = clock();

   // our 64-bit register
   unsigned long x = 0UL;

   // loading 4 "4-bit" integers in one 64-bit register
   for (int i = 0; i < 64; i = i + 8)
   {
      unsigned long number = rand()%16;
      x = x << 8;  // we leave space for multiplying by 15 (+4 bits)
      x = x | number;
   };

   // we repeat the entire procedure N times
   for (int k = 0; k < N; k++)
   {
      // calling an external function to perform the computations
      unsigned long y = computation(x);
   };

   // measuring the ending time
   time_t end = clock();

   // reporting computational time
   printf("xfifteen N %u CPU time %lf\n",N,compute_time(start,end));

   // ending
   return 0;
};

