
/* Main function for C version */

#include "figures.h"

// min function
double min(double x,double y)
{
   return (x < y) ? x : y;
};

// main
int main()
{
   // welcome message
   printf("Figures in C\n");
   printf("------------\n");

   // point_t
   point_t *p1 = point_new(1.5);
   point_print(p1);
   point_t *p2 = point_new(2.0);
   point_print(p2);
   point_t *p3 = point_intersect(p1,p2);
   if (p3 != NULL)
   {
      printf("The intersection is: ");
      point_print(p3);
   }
   else
   {
      printf("They have an empty intersection\n");
   };
   printf("------------\n");

   // segment_t
   segment_t *s1 = segment_new(1.0,2.0);
   segment_print(s1);
   segment_t *s2 = segment_new(1.5,2.5);  // change first to 2.0 to cause non-handled exception
   segment_print(s2);
   segment_t *s3 = segment_intersect(s1,s2);
   if (s3 != NULL)
   {
      printf("The intersection is: ");
      segment_print(s3);
   }
   else
   {
      printf("They have an empty intersection\n");
   };
   printf("------------\n");

   // mixed
   p3 = intersect_point_segment(p1,s1);
   printf("The intersection between the first point and the first segment is:\n");
   if (p3 != NULL)
      point_print(p3);
   else
      printf("Empty\n");

   // dont forget to free the memory!
   free(p1);
   free(p2);
   if (p3 != NULL)  free(p3);
   free(s1);
   free(s2);
   if (s3 != NULL)  free(s3);
   return 0;
};

