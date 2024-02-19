	.text
	.file	"xfifteen_computation.c"
	.globl	computation             # -- Begin function computation
	.p2align	4, 0x90
	.type	computation,@function
computation:                            # @computation
	.cfi_startproc
# %bb.0:                                computing 15x = 2*(4x+x) + (4x+x)
	movq	%rdi, %rcx		# rcx = x
	movq	%rdi, %rbx		# rbx = x
	shlq	$2, %rbx		# rbx = 4x
	addq	%rcx, %rbx		# rbx = 5x
	movq	%rbx, %rax		# rax = 5x
	shlq	$1, %rax		# rax = 10x
	addq	%rbx, %rax		# rax = 15x
	retq
.Lfunc_end0:
	.size	computation, .Lfunc_end0-computation
	.cfi_endproc
                                        # -- End function

	.ident	"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"
	.section	".note.GNU-stack","",@progbits
