
# Multiple dispatch in Julia

At first sight, it looks like Julia supports function *overloading*.
But actually, Julia goes much farther than simply supporting overloading.
In the example given in the [previous lecture](./structs.md), we have 
invoked functions through the type of the ```Person``` structure that
we have specified as an argument. In object-oriented languages such as 
Java, we would have written:

	String iKid = Ivan.initials();
	String iTeacher = Antonio.initials();

where the calling object is placed in evidence at the left side of the
expression. In Julia, this syntax is not valid: there is no such a thing
like a *calling* structure. But this is not because Julia is simpler:
actually **Julia selects the function to be invoked on the basis of
all function arguments, and not only by considering the type of the 
calling object**.

## Person structure

Let's consider again the example given in the previous lecture, to which
we now include another structure inheriting from the abstract type: 
the ```Scientist``` structure:

	# Person (abstract type)
	abstract type Person end
	
	# property for all instances of Person
	function has_name(person::Person)
	    hasproperty(person,:name)
	end
	
	# Kid
	struct Kid <: Person
	   name::String
	   schoolYear::Int64
	   schoolName::String
	end
	
	# Teacher
	struct Teacher <: Person
	   name::String
	   courses::Vector{String}
	end
	
	# Scientist
	struct Scientist <: Person
	   name::String
	   research_fields::Vector{String}
	end

## Computer structure

Similarly, we define an abstract type representing computer devices,
and three different kinds of computer machines represented by Julia
structures. Of course, the specified attributes only allow for an 
extremely simple representation of these objects:

	# Computer (abstract type)
	abstract type Computer end
	
	# property for all instances of Computer
	function has_power(computer::Computer)
	   hasproperty(computer,:power)
	end
	
	# GamingConsole
	struct GamingConsole <: Computer
	   power::Int64
	   battery_life::Int64
	end
	
	# Computer with GPU
	struct GPU <: Computer
	   power::Int64
	   number_of_threads::Int64
	end
	
	# Quantum computer
	struct Quantum <: Computer
	   power::Int64
	   number_of_qbits::Int64
	end

## Everybody uses a computer nowadays!

Suppose we want to write functions capable of simulating the use of 
one of the instances of ```Computer``` by each instance of ```Person```.

### Single-dispatch in Java

In object-oriented languages such as Java, we would add methods in 
every implementation of ```Person```, having a signature similar to:

	public void workWith(Computer computer);

We may even consider to add this method signature in the common
interface defining every ```Person```, so that every implementation
of ```Person``` will have to implement the method.

When the method is implemented for objects of type ```Kid```, ```Teacher```,
and ```Scientist```, the selection of the method to be invoked is performed
on the basis of the calling object. In this example, however, we have
several possible implementations also for ```Computer```, and not only
for ```Person```. In Java, there is no method selection on the basis of
the type of the "other" input arguments, and for this reason we nowadays
say that Java is an example of *single-dispatched* programming language.

### Multiple-dispatch in Julia

The main idea is that the selection of the methods should not be performed
solely on the basis of one input argument, but on the basis of all of them.

Let's begin with the simplest situation we can have in our example.
As long as ```Kid``` structures are concerned, it is not necessary to 
generalize, these "structures" are normally only allowed to use gaming
consoles:

	julia> Ivan = Kid("Ivan",5,"primary")
	Kid("Ivan", 5, "primary")
	
	julia> console = GamingConsole(1,100)
	GamingConsole(1, 100)
	
	julia> function work(kid::Kid,console::GamingConsole)
	          print(kid.name," is playing with a ",typeof(console))
	       end
	work (generic function with 1 method)
	
	julia> work(Ivan,console)
	Ivan is playing with a GamingConsole

Notice that, if there are no ```work``` functions involving at the same time 
```Kid``` and ```Computer``` structures other than ```GamingConsole```, the
kids are automatically not allowed to play with GPUs and quantum computers:

	julia> gpu = GPU(50,4096)
	GPU(50, 4096)
	
	julia> work(Ivan,gpu)
	ERROR: MethodError: no method matching work(::Kid, ::GPU)
	Closest candidates are:
	  work(::Teacher, ::Computer) at REPL[23]:2
	  work(::Kid, ::GamingConsole) at REPL[14]:1
	Stacktrace:
	 [1] top-level scope
	   @ REPL[27]:1

Differently, if we have a general interface in Java and we need to forbid some
potentially dangerous computer uses, we need to manually coded it (by raising exceptions 
for example).

Teachers, instead, may use different kinds of computer devices, for teaching purposes:

	julia> Antonio = Teacher("Antonio",["CSE","PO","HPC","ALG2","PA","PPAR"])
	Teacher("Antonio", ["CSE", "PO", "HPC", "ALG2", "PA", "PPAR"])
	
	julia> function work(teacher::Teacher,computer::Computer)
	          print(teacher.name," is teaching how to program a ",typeof(computer)," computer")
	       end
	work (generic function with 2 methods)
	
	julia> work(Antonio,gpu)
	Antonio is teaching how to program a GPU computer
	
	julia> quantum = Quantum(1000,1000)
	Quantum(1000, 1000)
	
	julia> work(Antonio,quantum)
	Antonio is teaching how to program a Quantum computer
	
We have a similar situation for the scientists. The main difference is that they
do not *teach*, but they rather *do research*:

	julia> Eistein = Scientist("Eistein",["Quantum Gravity","Faster-Than-Light Communication"])
	Scientist("Eistein", ["Quantum Gravity", "Faster-Than-Light Communication"])
	
	julia> function work(scientist::Scientist,computer::Computer)
	          print(scientist.name," is doing research on a ",typeof(computer)," computer")
	       end
	work (generic function with 3 methods)
	
	julia> work(Eistein,quantum)
	Eistein is doing research on a Quantum computer

But doing research may be quite different on the basis of the computing device that is
being used. The same applies when teaching to write programs for a specific computer system. 
We have here the very first glimpse of what multiple-dispatch is! In fact, the choise of the 
"function" to run in order to perform the ```work``` depends first of all on the ```Person```, 
but also on the ```Computer``` type that the person intends to use for the work!!

For example, teaching quantum computing is not really the same as teaching GPU programming!
We can add in Julia a specific implementation for this particular case:

	julia> function work(teacher::Teacher,computer::Quantum)
	          println(teacher.name, " is introducing the basis of quantum physics")
	          print("Programming on a ",typeof(computer)," computer requires this introduction")
	       end
	work (generic function with 4 methods)

	julia> work(Antonio,quantum)
	Antonio is introducing the basis of quantum physics
	Programming on a Quantum computer requires this introduction

Notice that the selection of the method to invoke may become very complex!
It's Julia that keeps track of the various variants of the methods sharing the same
name (here referred to as the *function*, having 4 *methods*, a.k.a. implementations
with different input arguments). 

## Question :-)

Wanna try to have to program a similar method selection strategy in programming languages 
which do not support multiple-dispatch?

## For a deeper understanding of multiple-dispatch

Please refer to this [lecture](./cmp/README.md).

## Links

* [Next: Linear Systems](./linear-systems.md)
* [Summary](./README.md)

