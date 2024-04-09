
/* segment_t struct */

#include "figures.h"

// creating a new segment_t
segment_t* segment_new(double x,double y)
{
   if (x == y)  abort();
   segment_t *news = (segment_t*)malloc(sizeof(segment_t));
   if (x < y)
   {
      news->x = x;
      news->d = y - x;
   }
   else
   {
      news->x = y;
      news->d = x - y;
   };
   return news;
};

// printing a segment_t
void segment_print(segment_t *s)
{
   printf("Segment(%7.4lf,%7.4lf)\n",s->x,s->x + s->d);
};

// intersecting two segment_t's
segment_t* segment_intersect(segment_t *s1,segment_t *s2)
{
   if (s1->x <= s2->x)
   {
      if (s1->x + s1->d < s2->x)
      {
         return NULL;
      }
      else
      {
         return segment_new(s2->x,min(s1->x + s1->d,s2->x + s2->d));
         // but segment_new will abort when the new segment is degenerate!
      };
   }
   else
   {
      if (s2->x + s2->d < s1->x)
      {
         return NULL;
      }
      else
      {
         return segment_new(s1->x,min(s2->x + s2->d,s1->x + s1->d));
         // again, segment_new will abort when the new segment is degenerate!
      };
   };
   return NULL;
};

