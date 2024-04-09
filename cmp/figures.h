
/* header file for all figures */

#include <stdio.h>
#include <stdlib.h>

// point_t structure definition
typedef struct
{
   double x;
}
point_t;

// point_t functions
point_t* point_new(double x);
void point_print(point_t* p);
point_t* point_intersect(point_t* p1,point_t* p2);

// segment_t structure definition
typedef struct
{
   double x;  // by convension, we store the lower bound x
   double d;  // and the distance from the lower bound to the upper bound
}
segment_t;

// segment_t functions
segment_t* segment_new(double x,double y);
void segment_print(segment_t* s);
segment_t* segment_intersect(segment_t* s1,segment_t* s2);

// point_t and segment_t join functions
point_t* intersect_point_segment(point_t* p,segment_t* s);
point_t* intersect_segment_point(segment_t *s,point_t *p);

// min function
double min(double x,double y);

