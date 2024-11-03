
# The Node

This exercise provides a practical introduction to data structures, using a 
ludic pedagogical approach to enhance learning. The used programming language 
is Java, but the same exercise can be easily adapted to other programming
languages supporting the object-oriented paradigm.

A [Node class](./Node.java) implementation is provided, as well as an 
initial implementation of another class named [Collection](./Collection.java). 
The latter will serve as a container for your ```Node``` objects. Your task 
will be to link these nodes together to represent different data structures, 
as specified below.

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

By doing so, you have created a cycle in your social network. Write a function to 
determine if a given node in your network is part of a cycle. In the current situation, 
in fact, all users should be in this cycle.

## Matteo breaks the cycle

To break the cycle, make Marie friends with Matteo. Retest the method you developed 
in the previous exercise in this new scenario.

## Claire has more than one friend

Claire, being very popular, wants to have more than one friend. Create a new class, 
name it ```DoubleNode```, that inherits from ```Node``` and allows an object to have 
two friends. Implement this new class. 

Then, decomment all lines in ```Collection``` related to Claire, and link her to both 
Thomas and Remi. Finally, visualize the resulting friendship network.

## A social network with tree structure

You'll notice that your social network now resembles a tree. Create a function that 
counts the number of people in the sub-network of a given person (including that person). 
Use recursion to solve this problem. Be careful with Claire, because she has two friends!

## Links

* [Back to main repository page](../README.md)

