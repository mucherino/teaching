
/* Aircraft class
 *
 * AirSimulation exercise on multi-threading
 *
 * AM
 */

import java.util.Iterator;
import java.util.NoSuchElementException;

public class Aircraft implements Iterable<Customer>
{
   // attributs
   private Seat[][] seatMap;
   private int nfree;

   // constructor for a predefined aircraft model
   public Aircraft()
   {
      int n = 32;
      int m = 9;
      this.seatMap = new Seat[n][m];

      // windows and emergency exists
      for (int i = 0; i < n; i++)
      {
         boolean emergency = false;
         if (i == 0 || i == 12 || i == 31)  emergency = true;
         this.seatMap[i][0] = new Window(emergency);
         this.seatMap[i][m - 1] = new Window(emergency);
      }

      // aisles
      for (int i = 0; i < n; i++)  this.seatMap[i][4] = new Aisle();

      // counting the number of free seats (destinated to Customers)
      this.nfree = 0;
      for (int i = 0; i < n; i++)
      {
         for (int j = 0; j < m; j++)
         {
            if (this.seatMap[i][j] == null)  this.nfree++;
         }
      }
   }

   // is the aircraft full?
   public boolean isFull()
   {
      return this.nfree == 0;
   }

   @Override
   // seat iterator
   public SeatIterator iterator()
   {
      return new SeatIterator(this);
   }

   // internal SeatIterator class
   // -> it only iterates over Customers
   // -> it throws a NoSuchElementException to indicate that it's over
   public class SeatIterator implements Iterator<Customer>
   {
      private int i;  // row index
      private int j;  // column index
      private Aircraft aircraft;  // reference to the Aircraft

      // constructor
      public SeatIterator(Aircraft aircraft)
      {
         super();
         this.aircraft = aircraft;
         this.i = 0;
         this.j = -1;
      }

      // did we start with the iterations?
      private boolean isCurrentDefined()
      {
         try
         {
            if (this.j == -1) throw new IllegalStateException("Use method 'next' in order to select the first Custormer seat");
         }
         catch (Exception e)
         {
            e.printStackTrace();
            System.exit(1);
         }

         return true;
      }

      @Override
      public boolean hasNext()
      {
         try
         {
            throw new Exception("We won't use this method");
         }
         catch (Exception e)
         {
            e.printStackTrace();
            System.exit(1);
         }

         return false;
      }

      @Override
      public Customer next() throws NoSuchElementException
      {
         this.j++;
         if (this.j == this.aircraft.seatMap[0].length)
         {
            this.i++;
            if (this.i == this.aircraft.seatMap.length) throw new NoSuchElementException();
            this.j = 0;
         }
         if (this.aircraft.seatMap[i][j] == null)  return null;
         if (this.aircraft.seatMap[i][j] instanceof Customer)  return (Customer) this.aircraft.seatMap[i][j];
         return this.next();  // we skip windows and aisles
      }

      // is the current seat free?
      public boolean isFree()
      {
         if (this.isCurrentDefined())  return this.aircraft.seatMap[this.i][this.j] == null;
         return false;
      }

      // is the current seat near an emergency exit?
      public boolean isNearEmergencyExit()
      {
         if (!this.isCurrentDefined())  return false;

         // looking at the left-side
         if (this.j > 0)
         {
            Seat seat = this.aircraft.seatMap[this.i][this.j - 1];
            if (seat instanceof Window)
            {
               Window window = (Window) seat;
               if (window.isEmergencyExit())  return true;
            }
         }

         // looking at the right-side
         if (this.j < this.aircraft.seatMap[0].length - 1)
         {
            Seat seat = this.aircraft.seatMap[this.i][this.j + 1];
            if (seat instanceof Window)
            {
               Window window = (Window) seat;
               if (window.isEmergencyExit())  return true;
            }
         }

         // we are not near an emergency exit
         return false;
      }

      // placing a Customer in the current position
      public void place(Customer c)
      {
         if (!this.isCurrentDefined())  return;
         try
         {
            if (c == null) throw new IllegalArgumentException("In order to free seats, use instead the method 'seatFree'");
            if (this.aircraft.seatMap[this.i][this.j] != null) throw new IllegalStateException("The seat is not free!");
         }
         catch (Exception e)
         {
            e.printStackTrace();
            System.exit(1);
         }

         this.aircraft.seatMap[this.i][this.j] = c;
         this.aircraft.nfree--;
      }

      // removing a Customer from the current position
      public void remove()
      {
         if (!this.isCurrentDefined())  return;
         try
         {
            if (this.aircraft.seatMap[this.i][this.j] == null) throw new IllegalStateException("The current seat is already empty!");
         }
         catch (Exception e)
         {
            e.printStackTrace();
            System.exit(1);
         }

         this.aircraft.seatMap[this.i][this.j] = null;
         this.aircraft.nfree++;
      }
   }

   @Override
   public String toString()
   {
      int n = this.seatMap.length;
      int m = this.seatMap[0].length;
      String print = "";
      for (int j = 0; j < m; j++)
      {
         for (int i = 0; i < n; i++)
         {
            Seat seat = this.seatMap[i][j];
            if (seat != null)
               print = print + seat.symbol() + " ";
            else
               print = print + "  ";
         }
         print = print + "\n";
      }
      return print;
   }

   // cleaning up the last output from toString 
   public String cleanString()
   {
      String print = "";
      for (int i = 0; i < this.seatMap[0].length + 2; i++)  print = print + "\033[F";
      return print + "\r";
   }
}

