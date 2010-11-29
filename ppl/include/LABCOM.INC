c
c	labcom
C** 
C**    @(#)LABCOM.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
	COMMON/LABCOM/ HLAB1,HXLAB,HYLAB,HLABS, 
     *	ICOUNT,NLAB,ICROSS,IGTLAB
c
c	hlab1	character height for main label (inches)
c	hxlab	character height for x axis character label (inches)
c	hylab	character height for y axis character label (inches)
c	hlabs	default character height for moveable labels (inches)
c	icount	number of plots produced
c	nlab	number of moveable labels
c	icross	=0 no cross
c		=1 draw through (0,0)
c		=2 horizontal line through each YOFF
c		=3 vertical line through each XOFF
c		=4 horizontal and vertical through each XOFF and YOFF
c	igtlab	=0 manual entry of graph labels
c		=1 get labels from data headers (EPIC and BIBO only)
c
c*******************************************************************
