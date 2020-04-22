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

* V510 *sh* 3/00 - dimension lleng() using the larger value, NPOLYGON
* V696 *acm*  6/15 Move ribbon_fast setting here.
* v696 *acm*	gap_var to define seperate line segments on PLOT/VS plots
*                  (LAS ticket 1597 and others)
* V698  4/16 *acm* New PLOT/NTH=



	COMMON/LINES/ IMARK(NLINES),HMARK(NLINES),ITYPE(NLINES),
     *	XOFF(0:NLINES),YOFF(0:NLINES),XFCT(0:NLINES),YFCT(0:NLINES),
     *	DN1(NLINES),UP1(NLINES),DN2(NLINES),UP2(NLINES),
     *	LLENG(NPOLYGON),LNUM,IWIND,gap_var,nskpsym(NLINES),
     *	ribbon_fast, ribbon_on
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
c	ribbon_fast FAST for one color per point, default is interpolate
c                       colors between points
c	ribbon_on flag when RIBBON is turned on (allows for RIBBON/SET; ppl plot)
c	gap_var pointer to variable on PLOT/VS plots for adding gaps
c
c*******************************************************************
