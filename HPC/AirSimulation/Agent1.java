
/* Agent1 (extending Thread)
 *
 * AirSimulation exercise on multi-threading
 *
 * AM
 */

import java.util.NoSuchElementException;

public class Agent1 extends Thread
{
   // reference to the Aircraft
   private Aircraft aircraft;

   // Agent1 constructor
   public Agent1(Aircraft aircraft)
   {
      super();
      this.aircraft = aircraft;
   }

   @Override
   // everytime it is invoked, it creates and places one Customer
   public void run() throws NoSuchElementException
   {
      if (this.aircraft.isFull())  return;
      Aircraft.SeatIterator seatIt = this.aircraft.iterator();
      while (seatIt.hasNext())
      {
         while (seatIt.hasNext() && !seatIt.isNextFree())  seatIt.next();
         if (seatIt.hasNext())
         {
            Customer c = new Customer();
            if (!c.isOver70() || !seatIt.isNextNearEmergencyExit())
            {
               seatIt.placeAsNext(c);
               return;
            }
         }
      }
   }
}

