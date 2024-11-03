
/* Collection class
 *
 * To be completed
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
   //private DoubleNode Claire;
   private Node Remi;

   // simple constructor
   public Collection()
   {
      Thomas = new Node("Thomas");
      Anne = new Node("Anne");
      Gilles = new Node("Gilles");
      Marie = new Node("Marie");
      Matteo = new Node("Matteo");
      //Claire = new DoubleNode("Claire");
      Remi = new Node("Remi");
   }

   // question 1
   public void question1()
   {
      // TO BE COMPLETED
   }

   // question 2
   public void question2()
   {
      // TO BE COMPLETED
   }

   // question 3
   public void question3()
   {
      // TO BE COMPLETED
   }

   // question 4
   public boolean question4(Node node)
   {
      // TO BE COMPLETED
      return false;
   }

   // question 5
   public void question5()
   {
      // TO BE COMPLETED
   }

   // question 6
   public void question6()
   {
      // TO BE COMPLETED
   }

   // question 7
   public int question7(Node node)
   {
      // TO BE COMPLETED
      return 0;
   }

   @Override
   // toString
   public String toString()
   {
      String s = "digraph Collection\n{\n";
      if (Thomas.hasFriend())  s = s + Thomas.toString(3) + "\n";
      if (Anne.hasFriend())  s = s + Anne.toString(3) + "\n";
      if (Gilles.hasFriend())  s = s + Gilles.toString(3) + "\n";
      if (Marie.hasFriend())  s = s + Marie.toString(3) + "\n";
      if (Matteo.hasFriend())  s = s + Matteo.toString(3) + "\n";
      //if (Claire.hasFriend())  s = s + Claire.toString(3) + "\n";
      if (Remi.hasFriend())  s = s + Remi.toString(3) + "\n";
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

      /*
      System.out.println("Question 6");
      C.question6();
      System.out.println(C);
      */

      /*
      System.out.println("Question 7");
      person = C.Claire;
      int nfriends = C.question7(person);
      System.out.println(person.getName() + " is root for a sub-tree containing " + nfriends + " friend(s)");
      System.out.println();
      */
   }
}

