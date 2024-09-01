
/* Agent1 class
 *
 * AirSimulation exercise on multi-threading
 *
 * AM
 */

public class Agent1 implements Agent
{
   // reference to the Aircraft
   private Aircraft aircraft;

   // Agent1 constructor
   public Agent1(Aircraft aircraft)
   {
      this.aircraft = aircraft;
   }

   // everytime it is invoked, it creates and places one Customer
   public void run()
   {
      if (this.aircraft.isFull())  return;
      Aircraft.SeatIterator seatIt = this.aircraft.iterator();
      try
      {
         while (seatIt.next() != null);
      }
      catch (Exception e)
      {
         return;  // nothing done
      }

      Customer c = new Customer();
      if (!c.isOver70() || !seatIt.isNearEmergencyExit())  seatIt.place(c);
   }
}

