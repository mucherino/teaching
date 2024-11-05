
/* Aisle class
 *
 * AirSimulation exercise on multi-threading
 *
 * AM
 */

public class Aisle implements Seat
{
   private String type;

   // constructor
   public Aisle()
   {
      this.type = "Aisle";
   }

   @Override
   public String type()
   {
      return this.type;
   }

   @Override
   public String symbol()
   {
     return "-";
   }

   @Override
   public String toString()
   {
      return this.type;
   }

   // main (simple testing)
   public static void main (String [] args)
   {
      System.out.println("Testing Aisle class");
      Aisle a1 = new Aisle();
      System.out.println(a1);
      System.out.println("type : " + a1.type());
      System.out.println("Symbol : '" + a1.symbol() + "'");
      System.out.println();
   }
}

