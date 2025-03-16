
# Multiple dispatch in Julia

From the [previous lesson](./structs-and-funs.md), it looks evident that
Julia supports *overloading*. But actually what Julia implements is much
more than simple method overloading. Julia supports *dynamic multiple-dispatch*.
This lesson is an attempt of introducing the concept of multiple-dispatching
in simple terms.

## Person structures

Recall, from the [previous lesson](./structs-and-funs.md), the abstract 
data type:

	# Person (abstract type)
	abstract type Person end
	
	# property for all instances of Person
	function has_name(person::Person)
	   hasproperty(person,:name)
	end

and two structures that inherit from ```Person```:

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

We will use in this lesson the automatic ```Base.show``` function for
all types, in order to avoid lines of code that are not strictly related 
to the topic of the lesson.

Oh, by the way, every time Julia encounters the symbol ```#```, it ignores 
the following characters. This is the way to leave single line comments in 
Julia code.

Now we introduce another data structure, which inherits from ```Person```:

	# Scientist
	struct Scientist <: Person
	   name::String
	   research_fields::Vector{String}
	end

## Computer structures

We define now an abstract type representing computer devices, and three 
different kinds of computer machines represented by Julia structures. Of 
course, our attributes only allow for extremely simplistic representations:

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

Our main aim is to show how easy is to write functions in Julia
(actually only one function, having as many methods as necessary) for
the simulation of the use of ```Computer``` machines by each of our
instances of ```Person```. The idea is that the selection of the method to
invoke (i.e. the choice on the specific implementation of the function that
Julia is going to run) is to be performed by taking into consideration *all*
function attributes, and not only one of them, as it generally happens in
other programming languages.

OK, you may reply, but this is just what overloading is supposed to do, isn't it?
Well, no, this is not only overloading! Let's try to understand the difference
by going through the following examples. 

### Kids playing with Gaming Consoles

To begin, we write the first method for our ```work``` function, which takes
as input arguments a ```Kid``` and a ```GamingConsole```:
	
	function work(kid::Kid,console::GamingConsole)
	   print(kid.name," is playing with a Gaming Console")
	end

After the function definition (with only 1 method at the moment), we can write
and execute in Julia the following lines:

	Ivan = Kid("Ivan",5,"primary");
	console = GamingConsole(1,100);
	work(Ivan,console)

which gives in output:

	Ivan is playing with a Gaming Console

As long as ```Kid``` structures are concerned, we may not want to generalize
the behavior of these "structures", because kids are normally only allowed to 
"work" with gaming consoles. And in fact, if we try to invoke the method above
with another kind of computer, we get an error message. The Julia code:

	gpu = GPU(50,4096);
	work(Ivan,gpu)

produces the output:

	ERROR: MethodError: no method matching work(::Kid, ::GPU)
	Closest candidates are:
	  work(::Kid, ::GamingConsole) at REPL[17]:1
	Stacktrace:
	 [1] top-level scope
	   @ REPL[22]:1

### Teachers and scientists have more rights

Differently from kids, the teacher can potentially teach to code on each of
the devices that we can represent. We can opt, in this case, to have a new method
that takes as an input argument a *generic* computer device, and not a specific one.
In this situation, we use the abstract type instead of one of its implementations:

	function work(teacher::Teacher,computer::Computer)
	   print(teacher.name," is teaching how to program a ",typeof(computer)," computer")
	end

Notice that ```typeof``` is a build-in function that allows us to get the type of
a given variable. The following Julia code:

	Antonio = Teacher("Antonio",["CSE","PO","HPC","ALG2","PA","PPAR"]);
	work(Antonio,gpu)
	work(Antonio,Quantum(1000,1000))

produces the input:

	Antonio is teaching how to program a GPU computer
	Antonio is teaching how to program a Quantum computer

The situation is similar for scientists. The main difference is that instead of teaching, 
but they do research:

	function work(scientist::Scientist,computer::Computer)
	   print(scientist.name," is doing research on a ",typeof(computer)," computer")
	end

Again, this method for the ```work``` function takes a specific instance of ```Person```,
but the abstract type ```Computer```. The Julia code:

	Eistein = Scientist("Eistein",["Quantum Gravity","Faster-Than-Light Communication"]);
	work(Eistein,quantum)

produces:

	Eistein is doing research on a Quantum computer

How many implementations of ```work``` do we have up to now? 3 methods!

### Dealing with particular situations

So far, we have dealt with rather generic situations. The methods concerning the 
teacher and the scientist, while specific for one of these two types, generally
take the abstract type ```Computer``` as an input argument. The only exception is the
method that we wrote for the ```Kid``` data structure, which can only consider the
```GamingConsole```. We have already observed that, if the kid tries to approach one 
of the other computer devices, then Julia outputs an error message. But what if we 
want to have a defined generic behavior instead of an error message, while keeping 
the possibility to specify a particular behavior for a given situation?

We can observe that teaching quantum computing is not really the same as teaching 
GPU programming. Let's add the following implementation of ```work```:

	function work(teacher::Teacher,computer::Quantum)
	   println(teacher.name, " is introducing the basis of quantum physics")
	   print("Programming on a ",typeof(computer)," computer requires this introduction")
	end

At this point we have 4 methods for the ```work``` function, and if we write in Julia:

	work(Antonio,Quantum(1000,1000))

we get the output:

	Antonio is introducing the basis of quantum physics
	Programming on a Quantum computer requires this introduction

So, to sum up, how many situations can be dealt with our methods?

- a kid that is playing with a gaming console;
- a teacher that is teaching to code on any kind of computer device (as long as they 
  inherit from ```Computer```, even devices that were not defined yet), except the 
  quantum computer;
- a teacher that is teaching to code on a quantum computer;
- a scientist that is doing research on any kind of computer devices.

The selection of the method to invoke is indeed done by Julia by taking the type
of all input arguments into consideration.

## Question :-)

Wanna try to have to program a similar method selection strategy in programming languages
which do not support multiple-dispatch? If yes, do not read (not yet) the next sections,
otherwise you may lose the will to try this out.

### Multiple-dispatch is not overloading

While it might look similar, Julia's multiple-dispatch is not the classical concept of
overloading, which is supported by many programming languages. In languages such as C++ 
and Java, we can define the same data structures (aka *classes* or *objects*) involved 
in our example above, and write 4 different methods sharing the same name but having 4
different lists of arguments. This is allowed because of method overloading.

However, method overloading is resolved, by the C++ and Java compilers, at ... compile 
time! This implies that, when a variable is declared by using a generic type (an abstract
class or an interface), even if it can make reference to any of its "children" classes
at run time, the various possible situations cannot be predicted by the compiler at 
compilation time.

**This is the key difference with multiple-dispatch**. When a given function is called,
Julia chooses the method to invoke on the basis of the current type (at run time!) of 
all its arguments.

### Single-dispatch in other languages

Other languages give the possibility to benefit from a similar dispatching mechanism,
but by selecting only one variable at a time. This is the case of languages, already 
mentioned above, such as Java and C++. Since these are object-oriented languages,
they focus the attention on the *object*. When a given object invokes one of its
methods (which may be implemented also in other classes, implementing for example 
a common interface), then the selection of the method to invoke is also performed at 
run time. 

Julia does the same! But with a big difference: since object-oriented languages focus
on the object, the dispatch is performed by taking only *the calling object* into 
consideration. The dispatching mechanism in Julia is instead more general, because it 
takes all function arguments into account, with equal importance.

## For a deeper understanding of multiple-dispatch

In this lesson, we have only introduced the basis of multiple-dispatch in Julia,
and proposed some basic comparisons with object-oriented programming. For a
wider comparison among non-dispatched, single and multiple-dispatched languages,
please refer to this [lesson](../cmp/README.md).

## Links

* [Next: Linear Systems](./linear-systems.md)
* [Summary](./README.md)

