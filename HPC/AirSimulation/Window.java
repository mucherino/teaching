
/* Window class
 *
 * AirSimulation exercise on multi-threading
 *
 * AM
 */

public class Window implements Seat
{
   private boolean emergencyExit;
   private String type;

   // constructor
   public Window(boolean isEmergencyExit)
   {
      this.emergencyExit = isEmergencyExit;
      this.type = "Window";
   }

   // constructor without arguments (no emergency exit)
   public Window()
   {
      this(false);
   }

   // is this an emergency exit?
   public boolean isEmergencyExit()
   {
      return this.emergencyExit;
   }

   @Override
   public String type()
   {
      return this.type;
   }

   @Override
   public String symbol()
   {
      if (this.emergencyExit)  return "|";
      return "=";
   }

   @Override
   public String toString()
   {
      String print = this.type;
      if (this.emergencyExit)  print = print + " (emergency exit)";
      return print;
   }

   // main (simple testing)
   public static void main (String [] args)
   {
      System.out.println("Testing Window class");
      Window w1 = new Window();
      System.out.println(w1);
      System.out.println("type : " + w1.type());
      System.out.println("Symbol : '" + w1.symbol() + "'");
      Window w2 = new Window(true);
      System.out.println(w2);
      System.out.println("type : " + w2.type());
      System.out.println("Symbol : '" + w2.symbol() + "'");
      System.out.println();
   }
}

