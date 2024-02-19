
/* C code for multiplying integers by 15
 *
 * main only
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>

// we will be using the function
unsigned long computation(unsigned long x);

int main()
{
   // our 64-bit register
   unsigned long x = 0UL;

   // loading 4 "4-bit" integers in one 64-bit register
   printf("Original integers: ");
   for (int i = 0; i < 64; i = i + 8)
   {
      unsigned long number = rand()%16;
      x = x << 8;  // we leave space for multiplying by 15 (+4 bits)
      x = x | number;
      printf("%lu ",number);
   };
   printf("\n");

   // calling an external function to perform the computations
   x = computation(x);

   // extracting and printing the result
   unsigned long mask = 0xff;  // 255 aka 1111 1111
   printf("Results of multiplication by 15: ");
   for (int i = 56; i >= 0; i = i - 8)
   {
      unsigned long result = (x >> i) & mask;
      printf("%lu ",result);
   };
   printf("\n");

   // ending
   return 0;
};

