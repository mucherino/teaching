
# My own absolute value function

It's a very peculiar absolute value function: it 
returns 0 when the input integer number is negative.

## C version

We consider the following simple function in C
(see file [myabs.c](./myabs.c)):

	int myabs(int a)
	{
	   if (a > 0)  return a;
	   return 0;
	};

In order to create an executable file, we can link it to 
the main function (available in the file [main-myabs.c](./main-myabs.c)).
The reason for separating these two functions in two
different files is that we'll compile only ```myabs```
down to llvm and assembly; we won't do the same for the
main function (for which you can directly generate its
*object* file).

In order to generate the llvm and assembly codes, we can 
use the [```clang```](https://clang.llvm.org/) compiler 
(the version ```6.0.0-1ubuntu2``` was used to generate the 
codes below). The command to generate the llvm version of 
the function from C is:

	clang -S -emit-llvm myabs.c -o myabs.ll

The command to generate the assembly version (from llvm,
or directly from C) is:

	clang -S [input file] -o myabs.asm

## llvm version

The resulting [llvm](https://llvm.org) code is in the
file [myabs.ll](./myabs.ll). The lines that are more 
interesting for us are given below:

	define i32 @myabs(i32) #0 {
	  %2 = alloca i32, align 4
	  %3 = alloca i32, align 4
	  store i32 %0, i32* %3, align 4
	  %4 = load i32, i32* %3, align 4
	  %5 = icmp sgt i32 %4, 0
	  br i1 %5, label %6, label %8

	; <label>:6:
	  %7 = load i32, i32* %3, align 4
	  store i32 %7, i32* %2, align 4
	  br label %9

	; <label>:8:
	  store i32 0, i32* %2, align 4
	  br label %9

	; <label>:9:
	  %10 = load i32, i32* %2, align 4
	  ret i32 %10
	}

This llvm version of the code allows us to *see* what is 
going on in the CPUs when our simple function is running.
In particular, we can remark that a lot of memory transfers
between the main memory (the RAM) and the core registers
are performed. But since our function is so small and does
not make use of large amounts of data, it is legitimate 
to wonder whether these memory transfers are really
necessary. 

Notice that the block of instructions under the label 6 may 
be shorten by making direct reference to ```%4``` (instead of 
reloading its value in ```%7```), but this will force the 
assembly code to use additional memory in the main memory
to save the variable and reload it thereafter.

## Assembly version

The full translation in assembly of the ```myabs``` function 
can be found in the file [myabs.asm](./myabs.asm). The lines 
of the code on which we focus our attention are the following:

	   pushq   %rbp
	   movq    %rsp, %rbp
	   movl    %edi, -8(%rbp)
	   cmpl    $0, -8(%rbp)
	   jle     .LBB0_2
	   movl    -8(%rbp), %eax
	   movl    %eax, -4(%rbp)
	   jmp     .LBB0_3
	.LBB0_2:
	   movl    $0, -4(%rbp)
	.LBB0_3:
	   movl    -4(%rbp), %eax
	   popq    %rbp
	   retq

Again, we can remark the data transfers between the RAM and
the core registers.
 
## Modifying the assembly code

So, what about changing a little bit the assembly code as 
follows?

	   cmpl    $0, %edi
	   jle     .MY_LABEL
	   movl    %edi, %eax
	   jmp     .MY_SECOND_LABEL
	.MY_LABEL:
	   movl    $0, %eax
	.MY_SECOND_LABEL:
	   retq

Can you see the difference? What interest in changing the 
assembly code? The complete assembly code, equivalent to the 
original C function (i.e. performing exactly the same task
of the C function), can be found in the file 
[myabs-handwritten.asm](./myabs-handwritten.asm).

## Links

* [Next: Multiplying 4-bit integers by 15](./xfifteen.md)
* [Back to low-level programming](./README.md)
* [Back to main repository page](../README.md)

