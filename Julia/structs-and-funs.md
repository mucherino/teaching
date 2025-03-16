
# Structures and functions in Julia

*Structures?* Why do we talk about "structures" here? If you have some experience
coding in C, then you may be used to the special keyword ```struct``` for the 
definition of new data structures, which you can obtain by combining native
types or other data structures. But when C++ and Java were introduced, then we
stopped talking about "structures" and we began making reference to *objects* and
*classes*. Why to talk about structures again in Julia, a programming language
introduced in 2012? Well, one of the reasons for this is that **Julia is not
object-oriented**.

## Definition of a structure

The keyword ```struct``` is the same as in C, the remaining syntax is rather 
different from C, but quite simple to understand:

	struct Kid
	   name::String
	   schoolYear::Int64
	   schoolName::String
	end

We focus our attention on what *we do not see* in the definition of this data 
structure. The lines between the two keywords```struct``` and ```end``` do not 
only define the attributes for representing a ```Kid``` type, but implicitly
define a basic constructor for the new data structure. For example, after the
definition of the structure, the line

	Ivan = Kid("Ivan",5,"primary")

constructs a new variable named ```Ivan``` of type ```Kid```. Julia's answer when
invoking the constructor is:

	Kid("Ivan", 5, "primary")

This is, in fact, the text representation of the variable ```Ivan```. And yes, if
you're wondering, you're right: Julia didn't only automatically generate the basic
constructor, but also the basic method that allows us to visualize the data structure
(the equivalent of ```toString``` in Java). 

But how to make access to the structure attributes? We can simply use classical member
access operator, aka *dot* operator. For example

	Ivan.name

produces 

	"Ivan"

The quotes around the name indicate that the type of the printed variable is ```String```.
Notice that the access is direct, there is no need to have *getters*. You may wonder, as
a consequence, whether we'd need to have *setters*. If not, probably you're concerned 
about the open access to these attributes. So, here's how Julia deals with the attributes 
of its data structures: by default, all attributes are **immutable**. In fact, if we try 
to modify the value of an attribute of ```Ivan``` with the line:

	Ivan.schoolYear = 6

then we get the following error message:

	ERROR: setfield!: immutable struct of type Kid cannot be changed
	Stacktrace:
	 [1] setproperty!(x::Kid, f::Symbol, v::Int64)
	   @ Base ./Base.jl:39
	 [2] top-level scope
	   @ REPL[7]:1

This is the way that Julia guarantees (at least partially) the encapsulation principle.

## Functions in Julia

In order to keep exploring Julia basic functionalities, we introduce now another data 
structure:

	struct Teacher
	   name::String
	   courses::Vector{String}
	end

Notice that the standard type for a *list* in Julia is a ```Vector```, the same
vector type that we have seen in our previous lesson.

*Functions* are determined only by their name in Julia. For example, the function
named ```add``` may be invoked for performing the sum of numbers, and also for 
inserting a new element in a collection such as a set. Naturally, there must be
two distinct implementations of the function ```add``` in order to deal with 
the two different situations. To this purpose, Julia introduces the *methods* of
a given function. Methods related to a given function have the same name of the
function, but each method has its own list of arguments. The way Julia selects
the method to invoke is the topic of the [next lesson](./multiple-dispatch.md).

In order to begin with a simple example, we define a new generic function and we
write one method for this function which simply counts the number of courses that 
a teacher is teaching:

	function numberOfCourses(teacher::Teacher) :: Int64
	   return length(teacher.courses)
	end

When we load the function in Julia, the REPL answers with the following output:

	numberOfCourses (generic function with 1 method)

indicating that ```numberOfCourses``` is recognized as a new function, and that 
it has (at the moment) only one implementation (one method).

In order to test our new function, we create a new instance of ```Teacher```:

	Antonio = Teacher("Antonio",["CSE","PO","HPC","ALG2","PA","PPAR"])

Julia answers by printing its auto-generated text representation:

	Teacher("Antonio", ["CSE", "PO", "HPC", "ALG2", "PA", "PPAR"])

And if we now invoke the function by passing the instance ```Antonio``` as an
argument:

	numberOfCourses(Antonio)

we get

	6

which is the expected answer.

## Several methods for one function

A little like the idea behind *method overloading* implemented in other languages,
Julia gives the possibility to provide several methods (i.e. several different 
implementations, involving different data types) for one generic function. Moreover,
not only we can write new methods for functions that we have created by ourselves
(such as the function named ```numberOfCourses```), but also for pre-defined functions. 

For example, the standard function in Julia that is in charge to provide the text
representation of data types is named ```show```, and it belongs to the ```Base```
package. As mentioned earlier, a new method for this function is automatically
generated by Julia as soon as we introduce a new data structure. However, the
simple text representation that it offers may not be convenient in all situations.
In fact, while the representation of ```Kid``` seems quite convenient, the
automatic representation for ```Teacher``` may be inadequate sometimes, especially
if when representing teachers in charge of several courses.

The following three lines allow us to provide our own method for the function ```show```
when its main argument is of type ```Teacher```:

	function Base.show(io::IO,teacher::Teacher)
	   print(io,"Teacher(",teacher.name,", #courses = ",numberOfCourses(teacher),")")
	end

Once loaded, this method will alter the text representation of all instances of ```Teacher```.
For example, the line:

	Antonio

would now print

	Teacher(Antonio, #courses = 6)

The same applies to the function named ```==``` in Julia. This is the equivalent of 
```equals``` in Java. Again, this is an automatically generated method, at the time
of definition of a new data structure. By default, it compares all attributes of the
two compared instances of a common data type, and it answers ```true``` if and only 
if all attributes are equal.

```==``` is a function that *looks like* an operator. These are special functions that
admit two different syntaxes. For example, we can write:

	==(Ivan,Antonio)

as we did above for the function ```numberOfCourses```, but we can also write:

	Ivan == Antonio

to perform the comparison, which results to be more natural. Naturally, in this
example, the answer is always ```false```, because the two compared variables 
have different type.

## Abstract types

We can remark that both ```Kid``` and ```Teacher``` structures have a common 
attribute named ```name```, which essentially represent the same kind of 
information in the two structures. In case we are interested in writing methods
that solely use this common attribute, there is a real risk of code duplication. In 
order to alleviate this issue, we can opt on defining *abstract types* in Julia, 
and to implement methods that take these abstract types as arguments. Consequently, 
all types that *inherit* from the original abstract type, for which methods may have 
been implemented, have the right to "appear" in the function call at the place of
the abstract types.

We develop a little example below to clarify the use of abstract types. We 
introduce the abstract type named ```Person```:

	abstract type Person end

No attributes can be specified in Julia for abstract types. But if we want that the
structures inheriting from ```Person``` all contain an attribute named ```name```, 
indicating the name of the ```Person```, then we can impose it as follows:

	function has_name(person::Person)
	   hasproperty(person,:name)
	end

We can now redefine the ```Kid```:

	struct Kid <: Person
	   name::String
	   schoolYear::Int64
	   schoolName::String
	end

as well as the ```Teacher```:

	struct Teacher <: Person
	   name::String
	   courses::Vector{String}
	end

We now create a new function for the abstract type, which uses only the 
attributes that we know all instances of ```Person``` need to include:

	function initials(person::Person)
	   return person.name[1:1]
	end

At this point, all types inheriting from ```Person``` can be passed as an 
argument of the function, so that to invoke the same method. The line:

	initials(Antonio) * initials(Ivan)

gives

	"AI"

because the ```*``` operator, when applied to the type ```String```, performs
the concatenation.

Notice that, in Julia, inheritance from a *concrete* data type (in order to define 
a more specific data type) is not allowed. For example, we cannot inherit from 
```Kid``` in order to define ```JuniorSoccerPlayer```. 

The presented Julia code is available in the file [structs-and-funs.jl](./structs-and-funs.jl).

## Links

* [Next: Multiple dispatch](./multiple-dispatch.md)
* [Summary](./README.md)

