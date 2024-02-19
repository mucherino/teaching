	.text
	.file	"myabs.c"
	.globl	myabs                   # -- Begin function myabs
	.p2align	4, 0x90
	.type	myabs,@function
myabs:                                  # @myabs
	.cfi_startproc
# %bb.0:
###########################################
        cmpl    $0, %edi
        jle     .MY_LABEL
        movl    %edi, %eax
        jmp     .MY_SECOND_LABEL
.MY_LABEL:
        movl    $0, %eax
.MY_SECOND_LABEL:
        retq
###########################################
.Lfunc_end0:
	.size	myabs, .Lfunc_end0-myabs
	.cfi_endproc
                                        # -- End function

	.ident	"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"
	.section	".note.GNU-stack","",@progbits
