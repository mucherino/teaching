
# Solving integer linear programs with Julia

In comparison with what you have seen in the previous lessons, 
the linear program that we are going to define now does not have
real variables, but rather integer variables.

## Using coins to buy Paris metro tickets

This is a true story (unfortunately). I was living in Paris and one day, 
before going out, I thought: "I have to take the metro, and I have no 
tickets. But I have several coins that I can use for buying a *carnet* 
of tickets! In this way, I'll finally get rid of them". 

So as planned, I went to the metro station with all my coins. Once in 
front of the machine, I selected the option "carnet of tickets", and 
started to insert my coins, one by one. Since the idea was to get
rid of as many coins as possible, I had started inserting the smallest
coins I had, the 5 and 10 cents. I was so proud of myself. “This is a 
very good way to recycle my coins!”, I thought. But I was about to discover 
something very bad...  

"You cannot use more than 20 coins per transaction. Transaction canceled",
the machine suddenly told me. **WHAT?** My enthusiasm faded out... I had 
inserted no more than 5 euros worth of coins so far, while I had to pay 11.40 
euros in total! And what was worst, is that the machine then started to spit 
back all my coins, and, during that tragic moment, my mind was looking for a 
solution...

## Solutions to my problem

What about mainly using 1 or 2 euro coins? The machine would accept my coins 
in this way, but... this is not a good method to get rid of the small ones. No, 
no, I don't think this is the optimal solution. I think I can come up with 
something smarter.

So, basically there are 6 different coins that I can use. The 5 cents, the
10 cents, and the 20 cents. These are the smallest, the ones I won't be able
to use anywhere else. Then the 50 cent coin, and the 1 and 2 euros, these
may be used for other purposes (at the bakery?), but I need to include them 
in my problem because otherwise I won't be able to sum up to 11.40.

Let $C$ be a vector containing the value of each of the 6 coin. In Julia, 
we can write:

	C = [0.05,0.10,0.20,0.50,1.0,2.0];

Now, I need to consider that I have a limited number of coins at my disposal.
How many 5 cent coins I actually have? It looks like I have 6 of them. And 
what about the 10 cents? Let's collect this piece of information in another 
vector:

	nC = [6,8,4,4,7,5];

Now all my data are available in Julia. The Julia file [coins.jl](./coins.jl) 
contains the details of the linear program with integer variables that can 
actually solve my problem. It's up to you to explore it and understand how 
this model is supposed to find solutions.

It is worth pointing out the need of using integer variables (instead of 
real variables, like in the previous lessons), because I cannot accept 
solutions where I'm advised to insert a "part" of a coin and a "part" of 
another. The coin is one small but indivisible piece of money, implying
to my variables the need to be constrained to hold only integers. 

Finally, notice the use a different solver (w.r.t the one used in the
previous lessons):

	using GLPK;

Differently from ```Clp```, the ```GLPK``` package also implements methods 
for solving integer programs with integer variables.

## References

The content of this lesson was adapted from a previous exercise proposed
at the Ecole Polytechnique in Palaiseau (France) in 2010. This is the
[link](http://www.antoniomucherino.it/download/ISC610A/td3.pdf)
to the slides used at that time. Notice however that the linear program 
was not written in Julia at that time, and the solver was also different.

The data used in the model have not been updated. The actual cost for 
a *carnet* of tickets for the Paris metro has certainly increased instead.

## Links

* [Back to math programming lectures](./math-prog.md)
* [Back to main repository page](../README.md)

