c
c	cont
C** 
C**    @(#)CONT.INC	1.1    3/10/88
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
	COMMON/CONT/ ZLEV(ICNLEV),LDIG(ICNLEV),LWGT(ICNLEV),LCLR(ICNLEV),
     *	NLEV,HGT,NARC,IAUTOC,NLEV2,NSIGC,CANGLE,IVCOMP,
     *  CENTERED,SET_DELTA,DELTA

c
c	zlev()	array of levels to be contoured (user units)
c	ldig()	number opf decimal places in the contour labels
c		=0  no decimal
c		=-2 no label
c	lwgt()	=1 standard line
c		=2 heavy line
c		=3 dashed line
c       lclr()  index of color to use
c	nlev	number of levels to use
c	hgt	height of contour labels (inches)
c	narc	number of subsegments to replace each line segment of 
c		contour.
c	iautoc	=0  use supplied levels
c		=1  compute levels from input data and nlev
c	nlev2	actual number of levels to use 
c	nsigc	number of significant digits for labels if iautoc=1
c	cangle	angle to rotate the contour plot about its center.
c	ivcomp	vector component to contour or surf
c
c 	These added 8.3.95 *jd*
c	centered 	True if levels are to be centered around zero
c	set_delta	True if DZ will be specified by the user
c	delta		The specified DZ value

c*******************************************************************

