
/* mixed intersections */

#include "figures.h"

// intersection of a point with a segment
point_t* intersect_point_segment(point_t *p,segment_t *s)
{
   if (s->x <= p->x && p->x <= s->x + s->d)  return point_new(p->x);
   return NULL; 
};

// if we want the intersection of a segment with a point,
// we need a different function name in C
point_t* intersect_segment_point(segment_t *s,point_t *p)
{
   return intersect_point_segment(p,s);
};

