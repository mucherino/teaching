
/* An alternative way to multiply integers by 15
 *
 * 15x = 2*(4x+x) + (4x+x)
 *
 * AM
*/

unsigned long computation(unsigned long x)
{
   unsigned long four_x = x << 2;  // one shift
   unsigned long five_x = four_x + x;  // one sum
   unsigned long ten_x = five_x << 1;  // another shift
   return ten_x + five_x;  // another sum
};

