c
c	lines
C** 
C**    @(#)LINES.INC	1.2    5/31/88
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
	COMMON/LINES/ IMARK(NLINES),HMARK(NLINES),ITYPE(NLINES),
     *	XOFF(0:NLINES),YOFF(0:NLINES),XFCT(0:NLINES),YFCT(0:NLINES),
     *	DN1(NLINES),UP1(NLINES),DN2(NLINES),UP2(NLINES),
     *	LLENG(NLINES),LNUM,IWIND 
c
c	xoff(0), yoff(0), etc are the offsets for the contour and view
c	plots.
c
c	imark()	type of mark at each point  (0-88)
c	hmark()	height of mark (inches)
c	itype()	=0 no mark
c		=1 mark data points
c		=2 mark end points only
c		=3 only mark  (no connecting line)
c		=4 dashes
c		=5 dashes with mark at end points
c	xoff()	x offset
c	yoff()	y offset
c	xfct()	multiplicative factor for x
c	yfct()	multiplicative factor for y
c	dn1()	first down stroke for dashes (inches)
c	up1()	first up stroke for dashes (inches)
c	dn2()	second down stroke for dashes (inches)
c	up2()	second up stroke for dashes (inches)
c	lleng()	number of points in line
c	lnum	total number of lines for buffer
c	iwind	=0 windowing of data off
c		=1 window data to XLO, XHI and YLO, YHI
c
c*******************************************************************
