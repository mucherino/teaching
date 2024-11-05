
/* AirSimulation class
 *
 * AirSimulation exercise on multi-threading
 *
 * AM
 */

public class AirSimulation
{
   public static void main(String[] args)
   {
      Aircraft aircraft = new Aircraft();
      Agent1 ag1 = new Agent1(aircraft);
      Agent2 ag2 = new Agent2(aircraft);
      //Agent3 ag3 = new Agent3(aircraft);
      while (!aircraft.isFull())
      {
         ag1.run();
         ag2.run();
         //ag3.run();
         System.out.println(aircraft);
         System.out.println(aircraft.cleanString());
      }
      System.out.println(aircraft);
   }
}

