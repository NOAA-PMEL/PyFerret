c
c	labels
C** 
C**    @(#)LABELS.INC	1.1    3/10/88
C** v5.51+ *acm* 3/03 Longer strings: changes for multi-line labels.
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
	COMMON/LABELS/ LAB1,XLAB,YLAB,LABS 
	CHARACTER LAB1*2048,XLAB*2048,YLAB*2048,LABS(NLABS)*2048

        CHARACTER fontprefix*3, penprefix*3, newfont*3, newpen*3
        INTEGER nlablines
        REAL line_spacing
        PARAMETER (nlablines=250,
     .             line_spacing=1.4)
        INTEGER lnbeg(nlablines), lnend(nlablines)
c
c	lab1	main label
c	xlab	x axis character label
c	ylab	y axis character label
c	labs()	moveable labels
c
c*******************************************************************
