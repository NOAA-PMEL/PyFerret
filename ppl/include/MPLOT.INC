c
c	mplot
C** 
C**    @(#)MPLOT.INC	1.1    3/10/88
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
	COMMON/MPLT/ MNX,MNY,IMULT,IMCNT,DXM(10),DYM(10), 
     *	XLENM(10),YLENM(10),IXM,IYM,MXBOT,MYLEF,HTM,WDM 
c
c	mnx	number of mult plots in x direction
c	mny	number of mult plots in y direction
c	imult	=0 mult plotting off
c		=1 mult plotting on
c	imcnt	current plot of mnx*mny mult plots
c	dxm(1:10)	spacing between plots x direction, 
c			bndry to col1, ...,  (inches)
c	dym(1:10)	spacing between plots y direction, 
c		row1 to row2,...,last row to bottom (inches)
c	xlenm(1:10)	length of x axis (inches)
c	ylenm(1:10)	length of y axis
c	ixm	current column -1
c	iym	current row
c	mxbot	=0 plot bottom x axis
c		=1 don't plot bottom x axis
c	mylef	=0 plot left y axis
c		=1 don't plot left y axis
c	htm	save height of plotting region (inches)
c	wdm	save width of plotting region (inches)
c
c*******************************************************************
