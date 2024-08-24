
/*
 * Conway's Game of Life
 *
 * basic sequential version
 *
 * AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

// torus structure
typedef struct
{
   size_t n;
   size_t m;
   bool* t;
}  torus_t;

// converting (x,y) indices into one unique z index
size_t torus_index(size_t n,size_t m,size_t x,size_t y,int dx,int dy)
{
   assert(n > 0UL);
   assert(m > 0UL);
   long i = x + dx;
   long j = y + dy;
   if (i < 0)  i = n + i;
   if (j < 0)  j = m + j;
   i = i%n;
   j = j%m;
   return (size_t) i*m + j;  // row by row organization of cells
};

// torus memory allocation
torus_t* torus_allocate(size_t n,size_t m)
{
   assert(n > 0UL);
   assert(m > 0UL);
   torus_t *torus = NULL;
   torus = (torus_t*)malloc(sizeof(torus_t));
   if (torus == NULL)  return NULL;
   torus->n = n;
   torus->m = m;
   torus->t = NULL;
   torus->t = (bool*)calloc(n*m,sizeof(bool));
   if (torus->t == NULL)
   {
      free(torus);
      return NULL;
   };
   return torus;
};

// writing a torus cell location
void torus_write(torus_t *torus,size_t x,size_t y,bool value)
{
   size_t z = torus_index(torus->n,torus->m,x,y,0,0);
   torus->t[z] = value;
};

// reading a torus cell location
bool torus_read(torus_t *torus,size_t x,size_t y,int dx,int dy)
{
   size_t z = torus_index(torus->n,torus->m,x,y,dx,dy);
   return torus->t[z];
};

// random generating a torus (includes allocation)
torus_t* torus_random(size_t n,size_t m,double p)
{
   assert(n > 0UL);
   assert(m > 0UL);
   assert(p >= 0.0 && p <= 1.0);
   torus_t *torus = torus_allocate(n,m);
   for (int z = 0; z < n*m; z++)
   {
      bool cell = false;
      if (rand()%10000/10000.0 < p)  cell = true;
      torus->t[z] = cell;
   };
   return torus;
};

// loading a torus model from a file (includes allocation)
torus_t* torus_load(char* filename,bool DOSformat)
{
   // if your text file is encoded in DOS format,
   // two chars appear at the end of each line
   short endofline = 1;
   if (DOSformat)  endofline++;

   // attempting to open the file
   FILE *input = fopen(filename,"r");
   if (input == NULL)  return NULL;

   // estimating the torus size
   fseek(input,0,SEEK_END);
   size_t filesize = ftell(input);
   rewind(input);
   char line[256];
   fgets(line,sizeof(line),input);
   size_t m = ftell(input) - endofline;
   size_t n = filesize/m;
   torus_t *torus = torus_allocate(n,m);
   rewind(input);

   // reading the file char by char
   char c;
   size_t z = 0;
   while (EOF != (c = fgetc(input)))
   {
      if (c != '\n')
      {
         bool cell = false;
         if (c == 'x')  cell = true;
         torus->t[z] = cell;
         z++;
      };
   };

   // ending
   fclose(input);
   return torus;
};

// counting the number of alive neighbours for (x,y)-cell
size_t torus_neighbours(torus_t *torus,size_t x,size_t y)
{
   size_t count = 0;
   for (int dx = -1; dx <= 1; dx++)
   {
      for (int dy = -1; dy <= 1; dy++)
      {
         if (dx != 0 || dy != 0)
         {
            if (torus_read(torus,x,y,dx,dy))  count++;
         };
      };
   };
   return count;
};

// creating the next generation
void torus_next(torus_t *src,torus_t *dst)
{
   bool cell;
   size_t nn;
   for (size_t x = 0; x < src->n; x++)
   {
      for (size_t y = 0; y < src->m; y++)
      {
         cell = torus_read(src,x,y,0,0);
         nn = torus_neighbours(src,x,y);
         if (cell)  // alive
         {
            if (nn < 2 || nn > 3)  cell = false;  // dies
         }
         else  // dead
         {
            if (nn == 3)  cell = true;  // gets born
         };
         torus_write(dst,x,y,cell);
      };
   };
};

// printing the torus
void torus_print(torus_t* torus)
{
   for (int x = 0; x < torus->n; x++)
   {
      for (int y = 0; y < torus->m; y++)
      {
         bool cell = torus_read(torus,x,y,0,0);
         if (cell)  printf("x");  else  printf(" ");
      };
      printf("\n");
   };
};

// freeing the torus
void torus_free(torus_t *torus)
{
   free(torus->t);
   free(torus);
};

// drawing a separation line on the screen
void draw_line(size_t n)
{
   assert(n > 0UL);
   for (size_t i = 0; i < n; i++)  printf("-");
   printf("\n");
};

// main
int main(int argc,char *argv[])
{
   size_t N = 20;  // predefined number of generations
   size_t SPEED = 200000;  // animation speed
   if (argc < 2)
   {
      fprintf(stderr,"%s: no input arguments; please provide input file name\n",argv[0]);
      return 1;
   };
   torus_t *torus1 = torus_load(argv[1],false);  // change to 'true' if you're working under Windows/DOS
   if (torus1 == NULL)
   {
      fprintf(stderr,"%s: a problem occurred while reading input file '%s'\n",argv[0],argv[1]);
      return 1;
   };
   torus_t *torus2 = torus_allocate(torus1->n,torus1->m);
   if (torus2 == NULL)
   {
      fprintf(stderr,"%s: impossible to allocate additional memory for second torus\n",argv[0]);
      return 1;
   };
   torus_t *tmp;
   torus_print(torus1);
   draw_line(torus1->m);
   usleep(SPEED);
   for (size_t i = 0; i < N; i++)
   {
      torus_next(torus1,torus2);
      torus_print(torus2);
      draw_line(torus2->m);
      usleep(SPEED);
      tmp = torus1;
      torus1 = torus2;
      torus2 = tmp;
   };
   torus_free(torus1);
   torus_free(torus2);
   return 0;
};

