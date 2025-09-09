
# The Node and its importance in developing collections

This exercise provides a practical introduction to data structures, using a 
ludic pedagogical approach to enhance learning. The used programming language 
is Java, but the same exercise can be easily adapted to other programming
languages supporting the object-oriented paradigm.

A [Node class](./Node.java) implementation is provided, as well as an 
initial implementation of another class named [Collection](./Collection.java). 
The latter will serve as a container for your ```Node``` objects. Your task 
will be to link these nodes together to represent different data structures, 
as specified below.

## The dot format

[Dot](https://en.wikipedia.org/wiki/Dot) is a graph description language. 
The ```Collection``` class has a ```toString``` method that generates a description 
of your data structure in this format. In order to visualize this structure as 
a graph:

1. Copy the output of the ```toString``` method into a text file (e.g., 
   "collection.dot").
2. On Linux, run the following command:

	dot -Tpng -o test.png collection.dot

This command will convert the ```dot``` file into a ```png``` image that 
you can thereafter open with an image viewer. The ```-Tpng``` option specifies 
the output format ```png``` and ```-o test.png``` indicates the output file name.

## Thomas is alone

Thomas has just arrived in Rennes and knows no one. In the ```Collection```
class, in fact, the ```Node``` object representing Thomas has been created by
using the constructor that takes only his name as an argument, hence without 
specifying a friend. Intrigued, Thomas attempts to establish a friendship with 
himself by calling the ```setFriend``` method and passing his own reference as 
an argument. To visualize the resulting network, use the ```toString``` method 
of the Collection class to generate the ```dot``` representation.

## Thomas meets Anne

A few days later, Thomas meets Anne, and they exchange their references. Thomas 
then uses the ```setFriend``` method to change the reference to his friend, hoping 
that Anne will do the same for him. But Thomas doesn't know that Anne is already 
friends with Gilles. Modify your collection to model this situation and visualize 
the result.

## and then he meets Marie

Model now the situation where Gilles is friends with Marie, and Marie, who has just
met Thomas, is friends with him.

By doing so, you have created a cycle in your "social network". Write a function to 
determine if a given node in your network is part of a cycle. In the current situation, 
in fact, all users should be in this cycle.

In order to make things easier for us, let's suppose that only two scenarios are possible:

- either the entire data structure *is* a cycle;
- or there are no cycles at all in the data structure.

## Matteo breaks the cycle

To break the cycle, make Marie friends with Matteo. Retest the method you developed 
in the previous exercise in this new configuration.

## Claire has some good suggestions for improvement

It is at this point that Claire, a very popular girl, comes in and suggests replacing 
our ```Collection``` class with another one. She thinks in fact it is unacceptable to 
have to manually add all our friends in the class constructor. A more efficient approach 
is undoubtedly necessary!

To facilitate this transition, she takes charge of creating a new class. She names the 
new class ```MySocialNetwork```, even if the class will only be limited to representing 
friendship lists, and not more complex networks. The ```MySocialNetwork``` class will 
have two attributes: the first node in the list, as well as a positive integer indicating 
the total number of friends in the list.

Please help Claire develop this new class. Start by defining the attributes, writing 
a constructor that creates an empty list, and implement the ```length``` method.

## Reading the friendship list from a text file

But how to pass the information about all friends in our "social network" ? To this aim,
let's implement a second constructor for our new class to load all friends into the data 
structure at once. We will assume that the name of each friend is stored on a separate line 
in a text file, and that the friendship link is automatically established between each 
consecutive pair of friends.

Next, based on the ```toString``` method of the ```Collection``` class, write the 
```toString``` method of ```MySocialNetwork``` to verify if your new constructor is 
capable of correctly building the simple friendship network that we had previously 
defined in the ```Collection``` class. Of course, you can still make use of the 
```toString``` method of ```Node```.

## Claire enters in the game

Finally, let's add Claire to our new collection! To do so, please write the ```add``` 
method in ```MySocialNetwork``` to include a friend to the end of the list, who will 
be automatically linked in friendship with the person who was previously the last.
Use ```toString``` to obtain a text representation of your data structure in dot
format, and visualize the result.

## Links

* [Back to main repository page](../README.md)

