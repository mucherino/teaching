
/* Point class */

public class Point implements Figure
{
   // my only attribute
   private double x;

   // constructor
   public Point(double x)
   {
      this.x = x;
   }

   // getter for the only attribute
   public double getX()
   {
      return this.x;
   }

   // intersecting with other Figure objects
   public Figure intersectWith(Figure other)
   {
      if (other instanceof Point)
      {
         Point p = (Point) other;
         if (this.x == p.x)  return new Point(this.x);
      }
      else if (other instanceof Segment)
      {
         Segment s = (Segment) other;
         if (s.getX() <= this.x && this.x < s.getY())  return new Point(this.x);
      }  // other types?
      return null;
   }

   // toString
   public String toString()
   {
      return "Point(" + this.x + ")";
   }

   // main
   public static void main(String[] args)
   {
      System.out.println("Point class");
      Point p1 = new Point(1.5);
      System.out.println(p1);
      Point p2 = new Point(2.0);
      System.out.println(p2);
      Figure p3 = p1.intersectWith(p2);
      System.out.println("The intersection is: " + p3);
      System.out.println();

      Segment s = new Segment(1.0,2.0);
      System.out.println("Now we define: " + s);
      System.out.println("And the intersection of " + p1 + " with " + s + " is " + p1.intersectWith(s));
   }
}

