
/* point_t struct */

#include "figures.h"

// creating a new point_t
point_t* point_new(double x)
{
   point_t *newp = (point_t*)malloc(sizeof(point_t));
   newp->x = x;
   return newp;
};

// printing a point_t
void point_print(point_t *p)
{
   printf("Point(%7.4lf)\n",p->x);
};

// intersecting two point_t's
point_t* point_intersect(point_t *p1,point_t *p2)
{
   if (p1->x == p2->x)  return point_new(p1->x);
   return NULL;
};

