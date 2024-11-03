
/* Node class
 *
 * AM
 */

public class Node
{
   // attributes
   protected String name;
   protected Node friend;

   // constructor w/out friends
   public Node(String name)
   {
      this.name = name;
      this.friend = null;
   }

   // getName
   public String getName()
   {
      return this.name;
   }

   // setting my friend
   public void setFriend(Node friend)
   {
      this.friend = friend;
   }

   // getting my friend
   public Node getFriend()
   {
      return this.friend;
   }

   // hasFriend
   public boolean hasFriend()
   {
      return this.friend != null;
   }

   // toString with indentation
   public String toString(int nlines)
   {
      String s = "";
      for (int i = 0; i < nlines; i++)  s = s + " ";
      s = s + this.name;
      if (this.friend == null)  return s + " -> null";
      return s + " -> " + this.friend.name;
   }

   @Override
   // toString
   public String toString()
   {
      return this.toString(0);
   }

   // main
   public static void main(String[] args)
   {
      Node A = new Node("Gael");
      Node B = new Node("Antonio");
      Node C = new Node("Matteo");
      A.setFriend(B);
      B.setFriend(C);
      System.out.println(A);
      System.out.println(B);
      System.out.println(C);
   }
}

