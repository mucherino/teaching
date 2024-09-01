
/* Customer class
 *
 * AirSimulation exercise on multi-threading
 *
 * AM
 */

import java.util.Random;

public class Customer implements Seat
{
   private int age;
   private int frequentFlyer;
   private int ticketNumber;
   private int flightCost;
   private boolean specialAssistence;
   private String type;

   // constructor for a random Customer
   public Customer()
   {
      Random rand = new Random();
      this.age = 20 + rand.nextInt(60);
      this.frequentFlyer = rand.nextInt(10);
      this.ticketNumber = 12345678 + rand.nextInt(87654321);
      this.flightCost = rand.nextInt(1500) + 500;
      double p = rand.nextDouble();
      this.specialAssistence = false;
      if (p < 0.2)  this.specialAssistence = true;
      this.type = "Customer";
   }

   // constructor from an existing Customer
   public Customer(Customer c)
   {
      try
      {
         if (c == null) throw new IllegalArgumentException("Customer: input Customer is null");
         this.age = c.age;
         this.frequentFlyer = c.frequentFlyer;
         this.ticketNumber = c.ticketNumber;
         this.flightCost = c.flightCost;
         this.specialAssistence = c.specialAssistence;
         this.type = "Customer";
      }
      catch (Exception e)
      {
         e.printStackTrace();
         System.exit(1);
      }
   }

   // getting Customer flyer level
   public int getFlyerLevel()
   {
      return this.frequentFlyer;
   }

   // checking whether the Customer is over 70
   public boolean isOver70()
   {
      return this.age > 70;
   }

   // checking whether the Customer needs special assistence
   public boolean needsAssistence()
   {
      return this.specialAssistence;
   }

   @Override
   public String type()
   {
      return this.type;
   }

   @Override
   public String symbol()
   {
      if (this.specialAssistence)  return "\033[31;1m" + this.frequentFlyer + "\033[0m";
      if (this.isOver70())  return "\033[33;1m" + this.frequentFlyer + "\033[0m";
      return "\033[32;1m" + this.frequentFlyer + "\033[0m";
   }

   @Override
   public boolean equals(Object o)
   {
      // pre-verification
      if (o == null)  return false;
      boolean isCustomer = (o instanceof Customer);
      if (!isCustomer)  return false;

      // comparing the two Customers
      Customer c = (Customer) o;
      boolean answer = this.age == c.age;
      answer = answer & (this.frequentFlyer == c.frequentFlyer);
      answer = answer & (this.ticketNumber == c.ticketNumber);
      answer = answer & (this.flightCost == c.flightCost);
      answer = answer & (this.specialAssistence == c.specialAssistence);
      return answer;
   }

   @Override
   public int hashCode()
   {
      return Integer.hashCode(this.age) + Integer.hashCode(this.frequentFlyer) + 
             Integer.hashCode(this.ticketNumber) + Integer.hashCode(this.flightCost) +
             Boolean.hashCode(this.specialAssistence);
   }

   // Printing
   public String toString()
   {
      String print;
      print = "(Customer age " + this.age;
      print = print + "; flyer level = " + this.frequentFlyer;
      print = print + "; ticket number = " + this.ticketNumber;
      print = print + "; flight cost = " + this.flightCost;
      if (this.specialAssistence)
         print = print + "; needs special assistence)";
      else
         print = print + ")";
      return print;
   }

   // main (simple testing)
   public static void main (String [] args)
   {
      System.out.println("Testing Customer class");
      Customer c1 = new Customer();
      System.out.println("Customer #1 : " + c1);
      System.out.println("Customer flyer level : " + c1.getFlyerLevel());
      System.out.println("Customer is over 70? " + c1.isOver70());
      System.out.println("Customer needs special assistence? " + c1.needsAssistence());
      System.out.println("Seat type : " + c1.type());
      System.out.println("Seat symbol : '" + c1.symbol() + "'");
      System.out.println("Customer #1 hash code : " + c1.hashCode());
      Customer c2 = new Customer();
      System.out.println("Customer #2 : " + c2);
      System.out.println("Testing 'c1.equals(c2)' : " + c1.equals(c2));
      Customer c3 = new Customer(c1);
      System.out.println("Customer #3 " + c3);
      System.out.println("Customer #3 hash code : " + c3.hashCode());
      System.out.println("Testing 'c1.equals(c3)' : " + c1.equals(c3));
      System.out.println();
   }
}

