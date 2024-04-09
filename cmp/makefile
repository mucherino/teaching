
# makefile for C version

figures: main.o point.o segment.o mixed.o
	gcc -o figures main.o point.o segment.o mixed.o

.c.o:
	gcc -c $<

clean:
	\rm -f *.o figures *.class  # also cleans Java class files

