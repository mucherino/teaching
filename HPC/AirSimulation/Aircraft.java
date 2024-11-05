
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

      // counting the number of free seats (devoted to Customers)
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

   // internal SeatIterator class (it only iterates over Customers, and free seats)
   public class SeatIterator implements Iterator<Customer>
   {
      private int i;  // row index
      private int j;  // column index
      private Aircraft aircraft;  // reference to the Aircraft

      // constructor for the seat iterator
      public SeatIterator(Aircraft aircraft) throws IllegalStateException
      {
         super();
         this.aircraft = aircraft;
         Seat[][] map = this.aircraft.seatMap;
         this.i = 0;
         this.j = 0;
         while (map[this.i][this.j] != null && !(map[this.i][this.j] instanceof Customer))
         {
            this.j++;
            if (this.j == seatMap[0].length)
            {
               this.i++;
               if (this.i == seatMap.length) throw new IllegalStateException(); // no Customers on which iterating!
               this.j = 0;
            }
         }
      }

      @Override
      public boolean hasNext()
      {
         return this.i < this.aircraft.seatMap.length;
      }

      @Override
      public Customer next() throws NoSuchElementException
      {
         if (!this.hasNext()) throw new NoSuchElementException();
         Seat[][] map = this.aircraft.seatMap;
         Customer TheNext = null;
         if (map[this.i][this.j] != null)  TheNext = (Customer) map[this.i][this.j];

         // looking already for the next to TheNext
	 do {
            this.j++;
            if (this.j == map[0].length)
            {
               this.i++;
               if (this.i == map.length)  return TheNext; // no more Customers after TheNext!
               this.j = 0;
            }
	 }  while (map[this.i][this.j] != null && !(map[this.i][this.j] instanceof Customer));

         // returning TheNext
	 return TheNext;
      }

      // is the next seat free?
      public boolean isNextFree() throws NoSuchElementException
      {
         if (!this.hasNext()) throw new NoSuchElementException();
         return this.aircraft.seatMap[this.i][this.j] == null;
      }

      // is the next seat near an emergency exit?
      public boolean isNextNearEmergencyExit() throws NoSuchElementException
      {
         if (!this.hasNext()) throw new NoSuchElementException();

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

      // placing a Customer in the next free location
      public void placeAsNext(Customer c) throws NoSuchElementException
      {
         if (!this.hasNext()) throw new NoSuchElementException();
         try
         {
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

      // removing a Customer from the next position
      public void removeNext() throws NoSuchElementException
      {
         if (!this.hasNext()) throw new NoSuchElementException();
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

      // extracting a Customer from the next position
      public Customer extractNext() throws NoSuchElementException
      {
         if (!this.hasNext()) throw new NoSuchElementException();
         Customer extracted = null;
         Seat[][] map = this.aircraft.seatMap;
         if (map[this.i][this.j] != null)  extracted = (Customer) map[this.i][this.j];
         map[this.i][this.j] = null;
         this.aircraft.nfree++;
         return extracted;
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

