
# The MU-puzzle

The MU-puzzle was initially proposed by Douglas Hofstadter in his book entitled 
[*Gödel, Escher, Bach*](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach). 
It is about a system of symbols consisting of only the three letters **M**, **I** 
and **U**, that can be combined together to form various chains of symbols.

The puzzle asks the following question. Given the starting chain **MI**,
can we construct the chain **MU** by applying to the starting chain a specific 
sequence formed by exclusively the following 4 transformation rules:

- *rule 1* : if the chain ends with an **I**, then we can add **U** to the chain 
  (for example, **MII** becomes **MIIU**);
- *rule 2* : if the chain is of type **Mx**, then we can add another **x** at 
  the end (for example, **MUIU** becomes **MUIUUIU**, with x = 'UIU');
- *rule 3* : if the chain contains **III**, then we can replace this sub-chain 
  with one **U** (for example, **MUIII** becomes **MUU**);
- *rule 4* : if the chain contains **UU**, the we can remove this sub-chain
  (for example, **MUUII** becomes **MII**).

In order to perform the exploration of all possible chains, we are going
to develop a specific class in Java for the creation and transformation
of the chains of symbols by applying the 4 rules given above. The Java
class is supposed to implement an ad-hoc linked list internal structure
for storing the chains of symbols. The use of the standard Java classes
implementing lists is strictly forbidden in this exercise.

## The Java class

Let's give to our Java class the same name of the puzzle:

	public class MUpuzzle
	{
	   private class Node
	   {
	      ...

You can start by defining the attributes of the internal class, the ```Node```, 
as well as the appropriate constructors.

Then, you should define the attributes of the main class, the ```MUpuzzle```
class. Please consider that it is convenient for us to have a direct
access to both the beginning and the end of the chain of symbols.
For the main class, it is necessary to write only one constructor,
which is supposed to generate our starting chain of symbols **MI**.

## Implementing the rules

You can now implement the 4 rules of the MU-puzzle:

	public void rule1() { ... }
	public void rule2() { ... }
	public void rule3() { ... }
	public void rule4() { ... }

Please apply the rules 3 and 4 to all sub-chains of symbols for which
the rule can be applied. The sub-chains are to be searched by scanning
the main symbol chain from its beginning to its end (and not in the opposite
order; notice in fact that a different order may imply a different resulting
chain).

## Counting the symbols

To finalize this exercise, please implement the ```length``` method of the 
```MUpuzzle``` class. This method is supposed to count the total number of 
symbols in the current chain.

Notice that two possible implementations are actually possible.

* In the first implementation, the ```length``` method is supposed to count
  literally the symbols that are currently in the chain.

* In the second implementation, we want to avoid counting the symbols every
  time this is necessary. To this purpose, we include an extra attribute to 
  the ```MUpuzzle``` class, which is used to keep track of the current chain
  length. How many changes the introduction of this extra attribute is likely 
  to imply in our class?

Finally, compare your two implementations of this method in terms of 
computational complexity.

------------------------

