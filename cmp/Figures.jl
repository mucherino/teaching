
# All figures defined as structures in Julia
import Base.show;

# Point
struct Point
   x::Float64;
end

# Segment
struct Segment
   x::Float64;  # by convension, this is the lower bound
   d::Float64;  # and this is the distance (always positive) to the upper bound

   # to satisfy this convension, we need to write our own constructor
   function Segment(x::Float64,y::Float64)
      if (x == y) throw(ArgumentError("Rather use Point struct for a degenerate Segment")) end
      if x < y
         new(x,y - x);
      else
         new(y,x - y);
      end
   end

   # and override the default 'show' function
   function Base.show(io::IO,seg::Segment)
      print(io,"Segment(",seg.x,",",(seg.x + seg.d),")");
   end
end

## functions performing intersections between pairs of figures ##

# intersection between two Point structs
function intersect(p1::Point,p2::Point)
   if (p1.x != p2.x) return nothing end
   return Point(p1.x);
end

# intersection between a Point and a Segment structs
function intersect(p::Point,s::Segment)
   if (s.x <= p.x && p.x <= s.x + s.d) return Point(p.x) end
   return nothing;
end

# intersection between a Segment and a Point structs
function intersect(s::Segment,p::Point) intersect(p,s) end

# intersection between two Segment structs
function intersect(s1::Segment,s2::Segment)
   if s1.x <= s2.x
      if s1.x + s1.d == s2.x
         return Point(s2.x);
      elseif s1.x + s1.d > s2.x
         return Segment(s2.x,min(s1.x + s1.d,s2.x + s2.d));
      end
   else
      if s2.x + s2.d == s1.x
         return Point(s1.x);
      elseif s2.x + s2.d > s1.x
         return Segment(s1.x,min(s2.x + s2.d,s1.x + s1.d));
      end
   end
   return nothing;
end

