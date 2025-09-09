
/* Collection class
 *
 * TO BE COMPLETED ...
 *
 * AM
 */

public class Collection
{
   // our actors
   private Node Thomas;
   private Node Anne;
   private Node Gilles;
   private Node Marie;
   private Node Matteo;

   // very simple constructor
   public Collection()
   {
      Thomas = new Node("Thomas");
      Anne = new Node("Anne");
      Gilles = new Node("Gilles");
      Marie = new Node("Marie");
      Matteo = new Node("Matteo");
   }

   // Thomas is alone
   public void question1()
   {
      // TO BE COMPLETED ...
   }

   // Thomas meets Anne
   public void question2()
   {
      // TO BE COMPLETED ...
   }

   // and then he meets Marie
   public void question3()
   {
      // TO BE COMPLETED ...
   }

   // cycle-detection method
   public boolean question4(Node node)
   {
      return false;  // TO BE COMPLETED ...
   }

   // Matteo breaks the cycle
   public void question5()
   {
      // TO BE COMPLETED ...
   }

   @Override
   public String toString()
   {
      String s = "digraph Collection\n{\n";
      if (Thomas.hasFriend())  s = s + Thomas.toString(3) + "\n";
      if (Anne.hasFriend())  s = s + Anne.toString(3) + "\n";
      if (Gilles.hasFriend())  s = s + Gilles.toString(3) + "\n";
      if (Marie.hasFriend())  s = s + Marie.toString(3) + "\n";
      if (Matteo.hasFriend())  s = s + Matteo.toString(3) + "\n";
      return s + "}\n";
   }

   // main
   public static void main(String[] args)
   {
      System.out.println("The Node\n");

      /*
      System.out.println("Question 1");
      Collection C = new Collection();
      C.question1();
      System.out.println(C);
      */

      /*
      System.out.println("Question 2");
      C.question2();
      System.out.println(C);
      */

      /*
      System.out.println("Question 3");
      C.question3();
      System.out.println(C);
      */

      /*
      System.out.println("Question 4");
      Node person = C.Thomas;
      boolean answer = C.question4(person);
      System.out.println(person.getName() + " is in a cycle : " + answer);
      System.out.println();
      */

      /*
      System.out.println("Question 5");
      C.question5();
      System.out.println(C);
      answer = C.question4(person);
      System.out.println(person.getName() + " is in a cycle : " + answer);
      System.out.println();
      */
   }
}

