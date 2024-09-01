
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
      Agent agent1 = new Agent1(aircraft);
      Agent agent2 = new Agent2(aircraft);
      //Agent agent3 = new Agent3(aircraft);
      //Agent agent4 = new Agent4(aircraft);
      while (!aircraft.isFull())
      {
         agent1.run();
         agent2.run();
         //agent3.run();
         //agent4.run();
         System.out.println(aircraft);
         System.out.println(aircraft.cleanString());
      }
      System.out.println(aircraft);
   }
}

