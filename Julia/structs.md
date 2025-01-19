
# Structures in Julia

But wait! Julia is a modern language, why we talk about *structures* here,
while, since the introduction of languages such as C++ and Java, we are rather
used to talk about *objects*. Well, one of the reason why we refer to "structures",
exactly as we are used to do in standard C, is that Julia is **not** object-oriented.

## Definition of a structure

In Julia's prompt, we can define and subsequently manipulate a structure as follows:

	julia> struct Kid
	          name::String
	          schoolYear::Int64
	          schoolName::String
	       end
	
	julia> Kid
	Kid
	
	julia> Ivan = Kid("Ivan",5,"primary")
	Kid("Ivan", 5, "primary")
	
	julia> Ivan
	Kid("Ivan", 5, "primary")
	
	julia> Ivan.name
	"Ivan"
	
	julia> Ivan.schoolName
	"primary"

	julia> Ivan.schoolYear = 6
	ERROR: setfield!: immutable struct of type Kid cannot be changed
	Stacktrace:
	 [1] setproperty!(x::Kid, f::Symbol, v::Int64)
	   @ Base ./Base.jl:39
	 [2] top-level scope
	   @ REPL[7]:1
	
Some remarks:

* The definition of the structure (the lines between ```struct``` and ```end```)
  directly imply the definition of a basic constructor, the one that is employed
  a few lines later to define the variable ```Ivan``` of type ```Kid```.

* The access to the structure attributes can be done through the use of the typical 
  dot (```.```). The access is direct, there is no need to write getters.

* There is no *private* keyword, but the encapsulation principle is (at least partially)
  guaranteed by the immutability, by default, of all the attributes.

* Together with the standard constructor, other basic methods are "automatically" 
  generated, as for example:
  - the ```show``` method, the equivalent of ```toString``` in Java, that prints a text 
    representation of the ```Kid``` instance on the screen;
  - the ```==``` method, the equivalent of ```equals```, for comparing ```Kid``` instances 
    through the comparison of the value of all its attributes.

* All automatically generated methods (constructor and others) can be overridden if
  necessary.

Let's add create another structure now:

	julia> struct Teacher
	          name::String
	          courses::Vector{String}
	       end

	julia> Antonio = Teacher("Antonio",["CSE","PO","HPC","ALG2","PA","PPAR"])
	Teacher("Antonio", ["CSE", "PO", "HPC", "ALG2", "PA", "PPAR"])

	julia> Antonio.courses
	6-element Vector{String}:
	 "CSE"
	 "PO"
	 "HPC"
	 "ALG2"
	 "PA"
	 "PPAR"

Notice that the standard type for a *list* in Julia is a ```Vector```, the same 
vector type that we have seen in our previous lecture. 

For every new introduced structure, we can therefore write *functions* using
the information in, or acting on, the instances of the new structure. For example,
we can write a function that counts the number of courses that the teacher is
teaching:

	julia> function numberOfCourses(teacher::Teacher) :: Int64
	          return length(teacher.courses)
	       end
	numberOfCourses (generic function with 1 method)
	
	julia> numberOfCourses(Antonio)
	6

For the ```Teacher``` type, moreover, we may want to override the standard function
for text visualization of instances:

	julia> function Base.show(io::IO,teacher::Teacher)
	          print(io,"Teacher(",teacher.name,", #courses = ",numberOfCourses(teacher),")")
	       end

	julia> Antonio
	Teacher(Antonio, #courses = 6)

Now, we can remark that both ```Kid``` and ```Teacher``` have a common attribute
named ```name```, and therefore, there is a risk of code duplication in case we
are interested in writing functions using or acting on this common attribute.
Even if Julia is not object-oriented, it is possible to define *abstract types*
and to *inherit* from them to subsequently create new structures sharing similar 
properties. Inheritance from a structure to another is however not allowed in Julia.

Both a kid and a teacher are a ```Person```:

	julia> abstract type Person end

No attributes can be specified in Julia for abstract types. But if we want that the
structures inheriting from ```Person``` all contain an attribute named ```name```, 
indicating the name of the ```Person```, then we can impose it as follows:

	julia> function has_name(person::Person)
	           hasproperty(person,:name)
	       end
	has_name (generic function with 1 method)

We can now redefine the ```Kid```:

	julia> struct Kid <: Person
	          name::String
	          schoolYear::Int64
	          schoolName::String
	       end

as well as the ```Teacher```:

	julia> struct Teacher <: Person
	          name::String
	          courses::Vector{String}
	       end

If we write a function now taking as an argument the abstract type, we can
invoke it by passing as an argument instances of either types ```Kid``` or ```Teacher```:

	julia> function initials(person::Person)
	          return person.name[1:1]
	       end
	initials (generic function with 1 method)
	
	julia> typeof(Ivan)
	Kid
	
	julia> initials(Ivan)
	"I"
	
	julia> typeof(Antonio)
	Teacher
	
	julia> initials(Antonio)
	"A"

The full program in Julia is available in the file [structs.jl](./structs.jl).

## Links

* [Next: Multiple dispatch](./multiple-dispatch.md)
* [Summary](./README.md)

