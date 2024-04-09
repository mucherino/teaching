
/* Segment class */

public class Segment implements Figure
{
   // my two attributes
   private double x;
   private double d;

   // constructor
   public Segment(double x,double y) throws IllegalArgumentException
   {
      if (x == y) throw new IllegalArgumentException("Rather use Point object for a degenerate Segment object");
      if (x < y)
      {
         this.x = x;
         this.d = y - x;
      }
      else
      {
         this.x = y;
         this.d = x - y;
      }
   }

   // getter for the lower bound
   public double getX()
   {
      return this.x;
   }

   // getter for the upper bound
   public double getY()
   {
      return this.x + this.d;
   }

   // intersecting with other Figure objects
   public Figure intersectWith(Figure other)
   {
      if (other instanceof Point)
      {
         Point p = (Point) other;
         double px = p.getX();
         if (this.x <= px && px <= this.x + this.d)  return new Point(px);
      }
      else if (other instanceof Segment)
      {
         Segment s = (Segment) other;
         if (this.x <= s.x)
         {
            if (this.x + this.d == s.x)
            {
               return new Point(s.x);
            }
            else if (this.x + this.d > s.x)
            {
               return new Segment(s.x,Math.min(this.x + this.d,s.x + s.d));
            }
         }
         else
         {
            if (s.x + s.d == this.x)
            {
               return new Point(this.x);
            }
            else if (s.x + s.d > this.x)
            {
               return new Segment(this.x,Math.min(s.x + s.d,this.x + this.d));
            }
         }
      }
      return null;
   }

   // toString
   public String toString()
   {
      return "Segment(" + this.x + "," + (this.x + this.d) + ")";
   }

   // main
   public static void main(String[] args) throws Exception
   {
      System.out.println("Segment class");
      Segment s1 = new Segment(1.0,2.0);
      System.out.println(s1);
      Segment s2 = new Segment(1.5,2.5);
      System.out.println(s2);
      Figure s3 = s1.intersectWith(s2);
      System.out.println("The intersection is: " + s3);
      System.out.println();

      Point p = new Point(1.5);
      System.out.println("Now we define: " + p);
      System.out.println("And the intersection of " + s1 + " with " + p + " is " + s1.intersectWith(p));
   }
}

