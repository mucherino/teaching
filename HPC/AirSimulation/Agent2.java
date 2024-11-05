
/* Agent2 (extending Thread)
 *
 * AirSimulation exercise on multi-threading
 *
 * AM
 */

import java.util.NoSuchElementException;

public class Agent2 extends Thread
{
   // reference to the Aircraft
   private Aircraft aircraft;

   // Agent2 constructor
   public Agent2(Aircraft aircraft)
   {
      super();
      this.aircraft = aircraft;
   }

   @Override
   // everytime it is invoked, it selects and moves one Customer
   public void run() throws NoSuchElementException
   {
      if (this.aircraft.isFull())  return;

      // looking for the Clustomer with higher frequency number
      Aircraft.SeatIterator seatIt = this.aircraft.iterator();
      Customer highest = null;
      while (seatIt.hasNext())
      {
         Customer current = seatIt.next();
         if (highest == null || (current != null && current.getFlyerLevel() > highest.getFlyerLevel()))
            highest = current;
      }
      if (highest == null)  return;

      // placing the "highest" Customer at the first seat
      seatIt = this.aircraft.iterator();
      Customer current = seatIt.extractNext();
      seatIt.placeAsNext(highest);
      seatIt.next();
      while (current != highest)
      {
         Customer next = seatIt.extractNext();
         seatIt.placeAsNext(current);
         seatIt.next();
         current = next;
      }
   }
}

