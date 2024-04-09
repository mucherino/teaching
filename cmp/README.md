
# Once upon a time, a Point, a Segment and ... a Square

This is the story of three simple geometrical figures that are looking
for the best way to be implemented. At the current time, three programming
languages have been tried out: the C programming language, Java, and finally
Julia. Have the three figures already found the best way to implement
themselves?

## The figures in C

The geometrical figures are not so ambitious. They only wish to be properly
initialized, printed out on the screen in a satisfying way, and intersect
themselves with other instances of their own group, or even with instances
of other groups, as long as the partner is still a geometrical figure.

### The point

The point decides to restrict himself to the one-dimensional case, so his
definition and initialization are rather trivial. In C, he defines himself 
through the following structure:

	typedef struct
	{
	   double x;
	}
	point_t; 

and writes the following function for its initialization:

	point_t* point_new(double x)
	{
	   point_t *newp = (point_t*)malloc(sizeof(point_t));
	   newp->x = x;
	   return newp;
	};

"Look, I've already set up my structure!", says the point to the segment. "I'll now
work on the other functions". But the segment is not so satisfied, he says "Careful, 
though. I can have access to your ```x``` attribute, and everybody else linked to you 
during the generation of the executable binaries will be able to do that!".

The point doesn't let the segment discourage himself and continues working on the 
other functions he needs. In order to print himself on the screen, he writes:

	void point_print(point_t *p)
	{
	   printf("Point(%7.4lf)\n",p->x);
	};

and to intersect itself with another point, he writes:

	point_t* point_intersect(point_t *p1,point_t *p2)
	{
	   if (p1->x == p2->x)  return point_new(p1->x);
	   return NULL;
	};

Except for the open access to its attribute, he's quite satisfied. "This solution in C
is not so bad after all", he says. This is the [full C file](./point.c).

### The segment

The segment is actually a little bit more ambitious. He doesn't want to be represented 
through the pair of lower and upper bounds on the Euclidean line (as the point had proposed
him to do), but rather through the lower bound alone, which he plans to couple then with the 
distance between the two bounds. In such a way, he thinks, he will be able to quickly retrieve 
the information about the distance between his lower and upper bounds, which, for some reasons 
we don't know, is very important for him.

Therefore, the structure in C concerning the segment finally looks like:

	typedef struct
	{
	   double x;  // by convention, we store the lower bound x
	   double d;  // and the distance from the lower bound to the upper bound
	}
	segment_t;

and the function for its initialization that he writes is:

	segment_t* segment_new(double x,double y)
	{
	   if (x == y)  abort();  // we cannot deal with exceptions here!
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

The structure definition and the function above do the job they were conceived for. 
However, he's a little angry with the point, because he had warned him about the
problem related to the open access to the attributes. After the efforts he made to
write his ```segment_new``` function, now he's concerned that somebody may change
his attribute values without telling him, and hence to break the order he tried to
create for himself...

Even worse, the segment realizes after a little while that he has even an additional 
problem. There is no native way in fact to deal with exceptions in C, and thus when 
by mistake there is an attempt to generate a degenerate segment, his function 
```segment_new``` cannot do anything smarter than aborting the entire program.

"Oh my god, why I listened to the point!?", he exclaims.

The same problem actually arises again when the segment tries to intersect himself
with another segment. See the [full C file](./segment.c). In fact, in some particular 
conditions, the intersection gives a point, and not a segment! This is another exception 
that finally the C implementation handles by simply making the program abort.

### Mixed intersections

In order to deal with non-mixed intersections, the point has already implemented 
the function:

	point_t* point_intersect(point_t *p1,point_t *p2);

available in [point.c](./point.c), and the segment has done the same in 
[segment.c](./segment.c):

	segment_t* segment_intersect(segment_t *s1,segment_t *s2);

We can notice that for each type, a new function, having a different name, is
necessary. But what about the functions dealing with mixed types? Well, in order
to have a clear reference to the types we are dealing with, every function will 
have to adapt its name to the types:

	point_t* intersect_point_segment(point_t *p,segment_t *s)
	{
	   if (s->x <= p->x && p->x <= s->x + s->d)  return point_new(p->x);
	   return NULL;
	};

And if we also want to have the function that takes as an argument first a segment,
and then a point, then we'll have to add another function with still another name:

	point_t* intersect_segment_point(segment_t *s,point_t *p)
	{
	   return intersect_point_segment(p,s);
	};

This is consequence of the fact that C does not support *function overloading*.
Notice that these last two functions use knowledge on how the point and the
segment are implemented, but they do not necessarily need to take part to the
set of functionalities characterizing the two geometrical figures. In fact, in
the C implementation, they are located in the external file named [mixed.c](./mixed.c).

## The figures in Java

The point and the segment spend quite a long time discussing the matter, and finally
they come to the conclusion that they should implement themselves in a more modern
language, in one of the languages supporting object-oriented programming. "What about
Java?", the point suggests.

First of all, the point says, I'll set up my ```x``` as a private attribute, and I'll
not include the getters and setters in my implementation. This will ensure me that
nobody will touch at my stuff:

	// my only attribute
	private double x;

	// constructor
	public Point(double x)
	{
	   this.x = x;
	};

But still, the point allows the others to look at its current state, through the standard 
```toString``` method:

	// toString
	public String toString()
	{
	   return "Point(" + this.x + ")";
	};

The segment is also happy about the idea to use Java. He remarks that in Java they can stop 
naming functions with several slightly different names, especially when these functions are 
actually supposed to perform the same task on a different set of variable types. The segment
goes a little farther with his thoughts, and proposes to define an interface, which they agree 
they should both implement:

	public interface Figure
	{
	   public Figure intersectWith(Figure F);
	};

"Great idea!", replies the point. 

However, a little of his initial enthusiasm fades away when he finds out that the interface 
introduces some new constraints for both of them. The method ```intersectWith``` defined in 
the interface, in fact, since it is generic for all geometrical figures, it takes a ```Figure``` 
type as an input argument, and returns a ```Figure``` type as well.  As a consequence, the very 
first implementation the point attempts for the method, where he was considering to use directly 
his own type:

	public Point intersectWith(Point other)
	{
	   if (this.x == other.x)  return new Point(this.x);
	   return null;
	};

gives the following error message:

	Point.java:4: error: Point is not abstract and does not override abstract method intersectWith(Figure) in Figure
	public class Point implements Figure
	       ^

The only way to make things work is to keep the method signature that appears in the 
interface. This doesn't not seem to be so bad, but it actually forces the point (and also 
the segment) to explicitly verify the type of ```Figure``` that is passed to their method,
and to cast accordingly thereafter:

	public Figure intersectWith(Figure other)
	{
	   if (other instanceof Point)
	   {
	      Point p = (Point) other;
              ...

Not really the ideal situation, maybe. The main idea behind using object-oriented programming 
was to get some benefits for their implementations, but it looks like our two heroes are quickly 
realizing that this paradigm also implies some new constraints.

While the point begins to have some doubts about the usefulness of implementing in Java, the 
segment is still quite happy because he can now easily handle exceptions. Instead of simply 
aborting the program when somebody tries to create a degenerate segment, he can now raise 
the following exception:

	IllegalArgumentException("Rather use Point object for a degenerate Segment object");

More precisely, the exception is raised by his constructor, so it's the final user that actually 
decides what to do with this exception. The user can decide to abort, as it was done in the C
implementation, or rather to catch the exception and jump to a piece of code that is supposed 
to deal properly with the situation.

Then the segment finds still another argument in favour of Java. "Look at my implementation of
```intersectWith```", he says to the point. "Since the returning value is a ```Figure```, I 
can now return any object that implements ```Figure```, and hence I can also return an object 
of your type!". All this excitement is due to the fact that, in the previous C implementation, 
the segment could not return references (or pointers) to types different from its own. For 
that reason, when his function attempted to create a degenerate segment in C, the program 
was doomed to abort.

But the point is still not sure. "I looked at your implementation of ```intersectWith```,
but now please give a look at mine: don't you see that we have a few problems in there??", 
he says to the segment. Let's give a look at the two Java files [Point.java](./Point.java)
and [Segment.java](./Segment.java). In summary, this is what the point is complaining about:

They both had to write additional code for the so-called *getters*, otherwise they would 
have not been able to see each other's attributes:

	public double getX()
	{
	   return this.x;
	}

And so what about their privacy? Well, at least who is reading cannot change the values 
of their attributes. Does this setup have anything to do with *immutable* types?

They both had to look at everybody else's attributes, learn how to properly use them,
and change their own code to adapt to the others. This seem to go quite a lot against 
the idea of "single-responsibility principle" the point had heard about some time ago. 
And what if another figure comes to play in the future? It is up to every single figure 
to update their own classes? Wouldn't it be better if an external actor could take care 
of the integration of other figures in their program?

"Hey, you're right", admits the segment, "we both ended up implementing the same piece of 
code!". "Exactly", the point replies. "And this will get worst when other figures will
join us. By the way, wasn't Java supposed to help us in avoiding duplicating code?".

## Julia

"What about implementing ourselves in Julia?", finally proposes the segment. But the point
is not really sure. In order to convince him, the segment claims that all problems they'd been 
discussing so far do not take place anymore when moving their implementations in Julia! 
Is this really true? And what is this *multi-dispatch* paradigm he's referring to? Let's 
give a look at [their code](./Figures.jl) (everything is in one file now).

## And the square?

Well, guys, I didn't like the idea to provide empty codes in my story above, i.e. functions 
or methods having only a given signature, and no code inside. But unfortunately the
implementation of the square figure would have taken too much of my time, and I think,
anyway, that the dialogues between the point and the segment already give us enough
insights about the limits and benefits of coding in these programming languages. The 
discussion can go on and touch other aspects of programming, of course, but this is 
independent on the inclusion of the square.

In case you'd like to add this third character, please go for it and do not hesitate to 
share your code (send the code to me by email, or simply make a push request on the 
repository). In Julia, the definition and initialization of the ```Square``` structure 
may look like:

	# Square
	struct Square
	   x::Float64;  # lowest x-coordinate
	   y::Float64;  # lowest y-coordinate
	   d::Float64;  # distance between lowest and highest (both x and y)

	   # we use the basic constructor but make sure that d is positive
	   function Square(x::Float64,y::Float64,d::Float64)
	      if (d <= 0.0) throw(ArgumentError("The sides of the square cannot be nonpositive")) end
	      new(x,y,d);
	   end

	   # we want the 'show' function to print all 4 vertices
	   function Base.show(io::IO,sq::Square)
	      print(io,"Square([",sq.x,",",sq.y,"] x ");
	      print(io,"[",(sq.x + sq.d),",",sq.y,"] x ");
	      print(io,"[",(sq.x + sq.d),",",(sq.y + sq.d),"] x ");
	      print(io,"[",sq.x,",",(sq.y + sq.d),"])");
	   end
	end

## Links

* [Back to Advanced Programming course](../Advanced.md)
* [Back to main repository page](../README.md)

