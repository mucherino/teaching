
# Chemical reactions in different languages

The main aim of this exercise is to explore the possible ways of implementation
for an application where it is matter of simulating the chemical reactions
between pairs of chemical substances. We will develop in Java in the first
part of this exercise, for moving then to Julia in the second part.

## Chemical reactions in Java

In Java, we suppose we have the following abstract class:

	public abstract class Substance
	{
	   // attributes
	   protected int quantity;
	
	   // getter for the quantity of substance
	   public int getQuantity()
	   {
	      return this.quantity;
	   }
	
	   // mixing the substance with another
	   public abstract String mix(Substance other);
	
	   @Override
	   public String toString()
	   {
	      return this.getClass().getName();
	   }
	}

As you can remark from the code above, all classes inheriting from ```Substance```
will have at least one attribute, indicating the quantity of the given substance.
A *getter* for this attribute is already implemented, as well as the ```toString```
method, which simply creates a ```String``` that contains the class name.

The ```mix``` method is not implemented in the abstract class. In fact, its 
implementation strongly depends on the specific class inheriting from this 
abstract class.

### Water, and other substances

The object-oriented paradigm suggests developing a distinct concrete class for 
every substance type that participates in the chemical reactions. For the 
```Water```, we have:

	public class Water extends Substance
	{
	   // constructor
	   public Water(int quantity)
	   {
	      this.quantity = quantity;
	   }
	
	   ...
	}

Similarly, we can include another class, named ```Sodium```.

### Mixing the substances

Since we now have two concrete classes representing two different substances,
we can begin with the implementations of the ```mix``` method. When the two
mixed substances are identical, the returning ```String``` of the method will
simply indicate what kind of substance it is. Otherwise, please check on the
Internet what kind of chemical reactions the two different substances will 
activate, and shortly describe it in the output ```String``` of ```mix```.

Recall that, in Java, it is possible to verify the type of a given object 
through the use of the keyword ```instanceof```.

### More substances!

Let's include now another substance in our Java project: the ```Acid``` class.
When necessary, please update the existing Java files, and then run the following 
```main``` method:

	public static void main(String[] args)
	{
	   Water W = new Water(100);
	   Water Z = new Water(200);
	   System.out.println("Water with water : " + W.mix(Z));
	   System.out.println("Water with water (swapped arguments) : " + Z.mix(W));
	   Sodium S = new Sodium(50);
	   System.out.println("Water and sodium : " + W.mix(S));
	   System.out.println("Sodium and water : " + S.mix(W));
	   Acid A = new Acid(10);
	   System.out.println("Acid and sodium : " + A.mix(S));
	   System.out.println("Sodium and acid : " + S.mix(A));
	}

## Chemical reactions in Julia

In Julia, we have instead the following abstract type:

	abstract type Substance end
	
	function has_name(substance::Substance)
	   hasproperty(person,:name)
	end

Recall that, in Julia, the function ```typeof``` allows us to obtain the
type of a given variable. For example, if ```W``` is of type ```Water```,
the syntax ```typeof(W)``` provides this information. Moreover, if we 
need to "embed" this information into a ```String```, then we can simply
use the syntax ```$(typeof(W))```.

Can you now reproduce the same implementation for the simulation of the
chemical reactions in Julia? During the process, you'll most likely
remark some important differences with the previous (object-oriented) 
approach!

## Links

* [Back to Julia programming](./README.md)
* [Back to main repository page](../README.md)

