
# C programming

## A simple C function

We consider the following small function in C:

	int myabs(int a)
	{
	   if (a > 0)  return a;
	   return 0;
	};

The function, together with its header files, can also 
be found in the file [myabs.c](./myabs.c). In order to 
create an executable file, you can link it to the main 
function in the file [main-myabs.c](./main-myabs.c).
The reason for separating these two functions in two
different files is that we'll compile only ```myabs```
down to llvm and assembly; we won't do the same for the
main function (for which you can directly generate its
*object* file).

## Compiling to llvm

The compilation to llvm can be performed by using the 
```clang``` compiler. The resulting llvm code is in the
file [myabs.ll](./myabs.ll). The code snippets of the
code interesting us is given below:

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

## Compiling to assembly

The ```clang``` compiler also allows us to compile from
C code (or llvm code), to assembly. The full translation
in assembly of the ```myabs``` function can be found in
the file [myabs.asm](./myabs.asm). The lines of the code
on which we focus our attention are the following:

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
 
## Modifying the assembly code

What about changing a little bit the assembly code?

	   cmpl    $0, %edi
	   jle     .MY_LABEL
	   movl    %edi, %eax
	   jmp     .MY_SECOND_LABEL
	.MY_LABEL:
	   movl    $0, %eax
	.MY_SECOND_LABEL:
	   retq

Can you see the difference? What is the interest in changing
this assembly code? The complete assembly code, equivalent to 
the original C function (i.e. performing exactly the same job
of the C function), can be found in the file 
[myabs-handwritten.asm](./myabs-handwritten.asm).

## Links

* [Back to main repository page](../README.md)

