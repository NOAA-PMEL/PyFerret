c
c	xylabp
C** 
C**    @(#)XYLABP.INC	1.1    3/10/88
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
	COMMON/XYLABP/XLABS(NLABS),YLABS(NLABS),RLABS(NLABS),
     *  LTYPE(NLABS),XLLABS(NLABS),YLLABS(NLABS),LLTYPE(NLABS),
     *	RHLABS(NLABS),USRLBS(NLABS)
	LOGICAL USRLBS
c
c	xlabs()	x position of moveable label (user units)
c	ylabs() y position of moveable label (user units)
c	rlabs()	angle to draw label (degrees)
c	ltype()	=-1 x,y is at lower left of label
c		=0  x,y is at lower center of label
c		=1  x,y is at lower right of label
c	xllabs()x position of line start (user units)
c	yllabs()y position of line start (user units)
c	lltype()=0 no line
c		=1 normal line
c		=2 fancy line
c	rhlabs()character height for moveable label (inches)
c	usrlbs()user units if true  (system units if false)
c
c*******************************************************************
