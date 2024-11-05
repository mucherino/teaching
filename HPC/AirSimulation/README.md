
# AirSimulation

The director of an Airline company wishes to improve the performances of his small 
group of agents. The instructions that the agents execute are supposed to be specified 
in Java language. Every agent is supposed to be interacting mainly with the Java class 
named ```Aircraft```, which represents an aircraft through its map of seats and makes 
use the following Java files:

- ```Seat```, a common interface for the objects that can form the seat map of an aircraft;
- ```Customer```, an implementation of ```Seat``` representing a customer;
- ```Window```, an implementation of ```Seat``` which represents a window in the seat map;
- ```Aisle```, an implementation of ```Seat``` which represents an aisle in the seat map.

The director wants his agents to get used to work with a modern multithreading computer system.
In order to help the director to achieve this task, our initial task will consist in looking 
at the implementation of the provided Java classes, available here in the current directory 
of the repository. The director himself has coded these classes, and he says you should not 
modify his code without his explicit authorization. Don't hesitate to ask your teacher to
get in touch with him ;-)

Notice that, among the attributes of ```Customer```, you can find its age, its frequent
flyer level, and the information on whether the customer needs special assistance or not.
An object of the ```Aircraft``` class can be visualized through the ```toString``` method 
as follows:

![seatMap](seatMap.png)

where the red color is used to represent customers that need assistance, the yellow color 
indicates customer that are over 70, while the green color marks all other customers. The 
displayed numerical value is the flyer frequency level of the customer (the higher, the 
most frequent). In case your color system is not compatible with the one used by the 
director, you are authorized to modify the provided ```toString``` method.

## Simulating the behavior of a set of agents

### The two main agents

The director had asked each of his agents to write a Java class indicating the actions they
perform for the reservation of a customer seat in a given flight. The two main agents, *agent1*
and *agent2*, wrote their Java code immediately upon the director's request. To make things
easy (for themselves), they negleted some details, but the two codes seem nevertheless able to
work properly (notice that we don’t need to correct or improve their codes). Before continuing,
please give a look at the provided code.

### The third agent

The director has recently recruited a third agent, named *agent3*, who has not yet provided
his code. The director kindly asks you to help him write this new Java class. Basically, this new 
agent needs to work exactly like *agent1*, but instead of avoiding to place customers that are
over 70 near the security exits (as *agent1* does), he has to make sure that nobody needing 
special assistance is placed near the exits.

As you can remark, the overall approach of the group of agents is not optimal. They would need
much more coordination in order to make a proper job. But the confusion they are creating
will give us the perfect environment to test the new multithreading computer system!

Please write the code for *agent3* and test the system in sequential.

### The multithreading computer system

When the multithreading computer system is finally delivered, the director asks every agent (we 
have three agents right now) to use a seperated computing unit of the system. In other words, 
every agent will have to be associated to one single and independent computing resource. In 
Java, new threads can be invoked from a main running process with the ```start()``` method in 
```Thread``` (notice that all your agents already inherit from ```Thread```), It is your task 
to change the Java code in order to have the three agents running on three independent parallel 
threads. When done, please perform some tests. Does the system always work properly? If not, try 
to isolate the agent that may be causing the trouble.

### An expert suggestion

Very concerned for his new computer system, the director asks for an expert advise. The expert
explains to the director that, in this kind of computer systems, the memory is shared by the
agents and it is necessary to pay particular attention to the way every agent has access to it. 
In Java, he continues, ```Semaphores``` can be used for controlling the access to memory resources.
Even if the director had never heard about ```Semaphores``` before, after a quick online search, 
he finds some simple examples that allow him to fix the system! By using only one Semaphore, he 
makes sure that only one agent per time can have access to the seat map. Thanks to this 
modification, the system is now working again! 

### Doubts about the overall performances

After having successfully fixed the system, all director happiness fades away when he finds out 
that the use of his ```Semaphore``` makes his system much much slower. Actually, it looks like his
new computer system is even slower than the old one. Can you imagine a less restrictive use of 
```Semaphores``` that could allow the new computer system to work properly while providing good
performances? At the end, the director's happiness is totally in your hands!

## Links

* [Back to main repository page](../../README.md)

