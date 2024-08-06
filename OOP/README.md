
# Quaternions in Java

This exercise has as main aim of employing the object-oriented 
paradigm for implementing a Java class modeling the behavior of
*quaternions*. The exercise is conceived in such a way to give the
student the possibility to revise her knowledge in object-oriented
programming before beginning to study data structures such as lists,
maps, etc. In fact, quaternions can be seen as small collections 
consisting of 4 real numbers.

A little like complex numbers can extend the real numbers by 
introducing the imaginary unit $i$ (such that $i^2 = -1$) and 
can be used for representing points and rotations in the 
two-dimensional Euclidean space, quaternions are able to perform 
a similar extension to the three-dimensional space. Instead of
adding only another imaginary component $j$ to a complex number
(Sir William Rowan Hamilton, the inventor of quaternions, tried 
for long time to make his new algebra work only with the introduction 
of $j$, but unsuccessfully), a quaternion can be seen as a complex
number to which two additional imaginary components, $j$ and $k$, 
are added, such that:
\begin{displaymath}
i^2 = -1, \quad j^2 = -1, \quad k^2 = -1, \quad ijk = -1 . 
\end{displaymath}
The generic notation for a quaternion is:
\begin{displaymath}
(a + bi + cj + dk) ,
\end{displaymath}
where $a$, $b$, $c$ and $d$ are real numbers.

More information about quaternions can be found on this
[wikipedia page](https://en.wikipedia.org/wiki/Quaternion).

## Definition of the Quaternion class

Very first task in our exercise is to model our Quaternion class
by paying particular attention to the encapsulation principle. 
Moreover, we want all our Quaternion instances to be immutable. 
Please begin writing some initial code for this class, as well as 
the three following constructors:

- a generic constructor, taking four real numbers $a$, $b$, $c$ and $d$
  in input, and capable of creating the corresponding Quaternion instance;
- a constructor for a *pure* quaternion: this is a quaternion without 
  real part;
- a constructor for a *real* quaternion: this is a quaternion without
  imaginary parts.

Remember the possibility to use implemented constructors for writing 
a new one.

## Checking properties of Quaternion instances

Our constructors are able to create new quaternions having a given
property. Moreover, quaternions having specific properties may also 
be generated while performing calculations on quaternions. For this 
reason, it is important to include in our Java class the following 
three methods:

- ```isZero```, verifying whether the Quaternion instance has all its 
  elements set to 0:
- ```isPure```, verifying whether the Quaternion instance is pure or not;
- ```isReal```, verifying whether the Quaternion instance is real or not.

Please pay attention to the fact that round-off errors are likely 
to affect the values of $a$, $b$, $c$ and $d$ during the calculations.

## toString

In order to visualize our Quaternion instances, we override the
standard ```toString``` method. We want exactly the same format
as indicated below with these two examples:

	(1.0 + 2.0i + 3.0j + 4.0k)
	(1.0 - 2.0i - 3.0j - 4.0k)

## The main method

At this point, we are already able to construct instances of our
Quaternion class, and we are also able to print the details 
related to these instances on the screen. We can therefore begin
writing the ```main``` method to perform some basic tests. Then,
it is recommended that, for each new method we will include in
the class, we immediately write some new tests in this same 
main method to verify its correctness.

## Special quaternions

We want now to write methods for generating, for a given Quaternion
instance ```Q```, the following three special quaternions:

- the *conjugate* of ```Q```. Above, when giving the examples for
  the ```toString``` method, the two shown quaternions are one the
  conjugate of the other;
- the *versor* of ```Q```. The versor consists in a quaternion
  having the same components $a$, $b$, $c$ and $d$ of ```Q```
  divided by the norm of the original quaternion;
- the *reciprocal* of ```Q```. The reciprocal of a quaternion is
  the versor of its conjugate.

Notice that the norm of a quaternion (finding out what is the formula
for the computation of the norm is left to you) is necessary twice.
Moreover, please make your methods robust so that they can deal with
special situations where the versor and the reciprocal do not exist.

## Sums and multiplications

It is time now to write the methods for performing the three following
operations on quaternions:

- the sum of two quaternions;
- the product of a scalar by a quaternion;
- the product of two quaternions.

Recall that our Quaternion instances are supposed to be immutable.

## Final testing

If you followed the suggestion given below, you should have already 
implemented some basic tests for all the developed methods. These are 
some additional tests you may consider to include in your main method:

- it should be verified that the reciprocal of a quaternion ```Q```, when 
  multiplied by the original quaternion, actually gives the real quaternion
  with $a = 1$ (with a given tolerance for round-off errors);
- even if the multiplication of quaternions is not commutative, this property
  should still hold when multiplying a quaternion by its reciprocal;
- the non-commutativity can be verified by multiplying two random quaternions
  in the two possible orders.

## If you still have some time

In order to complete our Java class, we could override the two standard methods
```equals``` and ```hashCode```. Recall the signature of these two methods:

	public boolean equals(Object o)
	{
	   ...
	}
	
	public int hashCode()
	{
	   ...
	}

You can verify if the input ```Object``` instance is one of our quaternions 
with ```instanceof```.

-------------------

